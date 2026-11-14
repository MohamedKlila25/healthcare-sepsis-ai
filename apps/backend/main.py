"""Versioned research API. Public mode serves synthetic records only."""
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from healthcare_sepsis_ai.documents import DocumentRequest, extract_document
from healthcare_sepsis_ai.evidence import retrieve, search_pubmed
from healthcare_sepsis_ai.graph import build_graph
from healthcare_sepsis_ai.serving import ResearchService

REQUESTS = Counter("healthcare_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("healthcare_request_duration_seconds", "HTTP latency", ["route"])
hits = defaultdict(deque)
rate_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    key = os.environ.get("HEALTHCARE_API_KEY", "")
    allow_demo = os.environ.get("HEALTHCARE_ALLOW_DEMO", "true").lower() == "true"
    if not key and not allow_demo:
        raise RuntimeError("Set HEALTHCARE_API_KEY or explicitly enable synthetic demo mode")
    app.state.service = ResearchService()
    app.state.graph = build_graph(app.state.service)
    yield


app = FastAPI(title="Healthcare SepsisAI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])


@app.middleware("http")
async def monitoring(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    REQUESTS.labels(request.method, route, str(response.status_code)).inc()
    LATENCY.labels(route).observe(time.perf_counter() - started)
    return response


@app.exception_handler(ValueError)
async def invalid_value(request, exc):
    return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": str(exc)}})


@app.exception_handler(KeyError)
async def missing_patient(request, exc):
    return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "Unknown demo patient"}})


def authorize(request: Request, x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("HEALTHCARE_API_KEY", "")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "Invalid API key")
    # Bounded per-IP limiter, local-process only. Proxy headers are deliberately not trusted.
    identity = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with rate_lock:
        for key in list(hits):
            if not hits[key] or now - hits[key][-1] > 60:
                del hits[key]
        bucket = hits[identity]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= 60:
            raise HTTPException(429, "Demo request limit reached; retry in a minute", headers={"Retry-After": "60"})
        bucket.append(now)


class PredictionRequest(BaseModel):
    patient_id: str = Field(pattern=r"^DEMO-\d{3}$")
    hour: int | None = Field(default=None, ge=0, le=1000)


class AgentRequest(PredictionRequest):
    question: str = Field(min_length=3, max_length=2000)


protected = [Depends(authorize)]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "project": "Healthcare SepsisAI", "synthetic_demo": True,
            "model_ready": request.app.state.service.bundle is not None}


@app.get("/metrics", dependencies=protected)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/patients", dependencies=protected)
def patients(request: Request):
    return request.app.state.service.patients()


@app.get("/api/v1/patients/{patient_id}/timeline", dependencies=protected)
def timeline(patient_id: str, request: Request):
    return request.app.state.service.json_timeline(patient_id)


@app.post("/api/v1/predict", dependencies=protected)
@app.post("/api/v1/explain", dependencies=protected)
def predict(body: PredictionRequest, request: Request):
    return request.app.state.service.predict(body.patient_id, body.hour)


@app.post("/api/v1/agent/query", dependencies=protected)
def agent_query(body: AgentRequest, request: Request):
    return request.app.state.graph.invoke(body.model_dump(), config={"recursion_limit": 12})


@app.get("/api/v1/evidence", dependencies=protected)
def evidence(q: str = "sepsis prediction"):
    if len(q) > 2000:
        raise HTTPException(422, "Query too long")
    return retrieve(q)


@app.get("/api/v1/evidence/pubmed", dependencies=protected)
def pubmed(q: str = "sepsis prediction"):
    if not 3 <= len(q) <= 500:
        raise HTTPException(422, "Query length must be 3–500 characters")
    try:
        return search_pubmed(q)
    except Exception as exc:
        raise HTTPException(502, "PubMed is unavailable; retry later") from exc


@app.post("/api/v1/documents/analyze", dependencies=protected)
def analyze(body: DocumentRequest):
    return extract_document(body)


@app.get("/api/v1/models", dependencies=protected)
def models(request: Request):
    bundle = request.app.state.service.bundle
    return [] if bundle is None else [{"version": bundle["version"], "family": bundle["family"],
        "synthetic": True, "clinical_validation": False,
        "metrics": {k: v for k, v in bundle["report"].items() if k != "splits"}}]
