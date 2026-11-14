"""Conditional, traceable tool workflow; no LLM credentials required."""
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .evidence import SOURCES, retrieve


class ClinicalAgentState(TypedDict, total=False):
    patient_id: str
    hour: int | None
    question: str
    prediction: dict[str, Any]
    evidence: list[dict[str, Any]]
    trace: list[dict[str, str]]
    verification_passed: bool
    requires_human_review: bool
    final_report: str
    error: str


def build_graph(service):
    def patient_node(state):
        try:
            service.timeline(state["patient_id"], state.get("hour"))
            return {"trace": [{"node": "patient_data", "status": "validated"}]}
        except (KeyError, ValueError):
            return {"error": "Patient or requested history unavailable.",
                    "trace": [{"node": "patient_data", "status": "blocked"}]}

    def risk_node(state):
        result = service.predict(state["patient_id"], state.get("hour"))
        return {"prediction": result, "trace": state["trace"] + [
            {"node": "risk_prediction", "status": "abstained" if result["abstained"] else "completed"}]}

    def explain_node(state):
        return {"trace": state["trace"] + [{"node": "explanation", "status": "occlusion_computed"}]}

    def research_node(state):
        evidence = retrieve(state["question"])
        return {"evidence": evidence, "trace": state["trace"] + [
            {"node": "research", "status": "retrieved" if evidence else "no_matching_evidence"}]}

    def verify_node(state):
        known = {s["id"]: s["url"] for s in SOURCES}
        evidence = state.get("evidence", [])
        valid = all(known.get(item["id"]) == item["url"] for item in evidence)
        prediction = state.get("prediction", {})
        if state.get("error"):
            report = state["error"]
        elif prediction.get("abstained"):
            report = prediction["reason"]
        else:
            score = prediction["risk_score"]
            trajectory = [p for p in prediction["trajectory"] if p["risk"] is not None]
            prior = next((p for p in reversed(trajectory) if p["hour"] <= prediction["hour"] - 12), None)
            trend = (f" Change over 12 hours: {(score - prior['risk']) * 100:+.1f} percentage points."
                     if prior else " Less than 12 hours of comparable history available.")
            names = ", ".join(p["feature"] for p in prediction["contributions"][:3])
            report = (f"Synthetic patient {state['patient_id']}, hour {prediction['hour']}: "
                      f"model score {score:.1%}.{trend} Largest model sensitivities: {names}. "
                      "Sensitivities are not evidence of clinical causation.")
        if evidence and valid:
            report += " References: " + "; ".join(f"[{e['id']}] {e['title']}" for e in evidence) + "."
        elif not state.get("error"):
            report += " No matching evidence was retrieved for this question."
        report += " Research demo only. Not a diagnosis or treatment recommendation; human review required."
        return {"verification_passed": valid and not bool(state.get("error")),
                "requires_human_review": True, "final_report": report,
                "trace": state["trace"] + [{"node": "verification", "status": "passed" if valid else "blocked"}]}

    graph = StateGraph(ClinicalAgentState)
    for name, node in [("patient_data", patient_node), ("risk_prediction", risk_node),
                       ("explanation", explain_node), ("research", research_node), ("verification", verify_node)]:
        graph.add_node(name, node, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_edge(START, "patient_data")
    graph.add_conditional_edges("patient_data", lambda s: "verification" if s.get("error") else "risk_prediction")
    graph.add_conditional_edges("risk_prediction", lambda s: "verification" if s["prediction"]["abstained"] else "explanation")
    graph.add_edge("explanation", "research")
    graph.add_edge("research", "verification")
    graph.add_edge("verification", END)
    return graph.compile()

