import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app, hits
from healthcare_sepsis_ai.data import VITALS
from healthcare_sepsis_ai.documents import DocumentRequest, extract_document
from healthcare_sepsis_ai.evidence import SOURCES, retrieve


@pytest.fixture
def client(monkeypatch, trained_bundle):
    monkeypatch.setenv("HEALTHCARE_MODEL_PATH", "nonexistent.joblib")
    monkeypatch.delenv("HEALTHCARE_API_KEY", raising=False)
    hits.clear()
    with TestClient(app) as client:
        app.state.service.bundle = trained_bundle
        yield client


def test_prediction_and_timeline_do_not_expose_training_labels(client):
    timeline = client.get("/api/v1/patients/DEMO-001/timeline").json()
    assert "SepsisLabel" not in timeline[0]
    result = client.post("/api/v1/predict", json={"patient_id": "DEMO-001", "hour": 12}).json()
    assert result["hour"] == 12 and len(result["trajectory"]) == 13
    assert result["synthetic"] and result["clinical_validation"] is False
    assert 0 <= result["risk_score"] <= 1


def test_missing_data_causes_abstention_and_conditional_graph_route(client):
    app.state.service.records.loc[:, VITALS] = float("nan")
    response = client.post("/api/v1/agent/query", json={"patient_id": "DEMO-001", "question": "Explain risk"})
    result = response.json()
    assert result["prediction"]["risk_score"] is None
    assert result["prediction"]["abstained"]
    assert [t["node"] for t in result["trace"]] == ["patient_data", "risk_prediction", "verification"]


def test_missing_model_returns_no_invented_score(client):
    app.state.service.bundle = None
    result = client.post("/api/v1/predict", json={"patient_id": "DEMO-001"}).json()
    assert result["abstained"] and result["risk_score"] is None


def test_graph_report_uses_known_citations(client):
    result = client.post("/api/v1/agent/query", json={"patient_id": "DEMO-001", "question": "Explain sepsis prediction and missingness"}).json()
    assert result["verification_passed"]
    assert result["requires_human_review"]
    known = {s["id"] for s in SOURCES}
    assert all(e["id"] in known for e in result["evidence"])
    assert "Research demo only" in result["final_report"]


def test_auth_validation_and_rate_limit(client, monkeypatch):
    monkeypatch.setenv("HEALTHCARE_API_KEY", "test-key")
    assert client.get("/api/v1/patients").status_code == 401
    headers = {"X-API-Key": "test-key"}
    assert client.get("/api/v1/patients", headers=headers).status_code == 200
    assert client.post("/api/v1/predict", headers=headers, json={"patient_id": "../../secret"}).status_code == 422
    assert client.get("/api/v1/patients/DEMO-999/timeline", headers=headers).status_code == 404
    for _ in range(60):
        response = client.get("/api/v1/patients", headers=headers)
    assert response.status_code == 429


def test_documents_require_units_and_reject_nonsynthetic():
    result = extract_document(DocumentRequest(text="Lactate: 2.4 mmol/L; WBC: 9; Temp: 99 C"))
    assert [m["field"] for m in result["measurements"]] == ["Lactate"]
    assert result["rejected"][0]["field"] == "Temp"
    with pytest.raises(ValueError, match="synthetic"):
        extract_document(DocumentRequest(text="Lactate: 2 mmol/L", synthetic=False))


def test_retrieval_never_returns_unmatched_notes():
    assert retrieve("zzzxxyy") == []
    assert retrieve("label shifted six hours")[0]["id"] == "physionet-2019"
