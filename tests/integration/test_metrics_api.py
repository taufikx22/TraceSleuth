"""Integration tests for Metrics, Alerts, and Model-Assisted Analysis APIs (Milestones 9 & 10)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.api.main import app
from src.ingestion.service import IngestionService
from src.storage.database import DatabaseManager, reset_database


@pytest.fixture(autouse=True)
def client():
    reset_database()
    import src.storage.database as db_module
    db_module._default_db = DatabaseManager(database_url="sqlite:///:memory:")
    db_module._default_db.init_db()

    with TestClient(app) as test_client:
        yield test_client
    reset_database()


def _seed_traces(client) -> tuple[str, str]:
    import src.storage.database as db_module
    db = db_module._default_db

    agent = InstrumentedResearchAgent()
    # 1. Seed healthy trace
    _, tr1 = agent.run(AgentRequest(query="Healthy query 1"))
    # 2. Seed failure trace
    _, tr2 = agent.run(AgentRequest(query="Failure query 2", failure_mode=FailureMode.POLICY_BYPASS))

    ingestion = IngestionService(db)
    ingestion.ingest_trace(tr1, auto_classify=True)
    inc2 = ingestion.ingest_trace(tr2, auto_classify=True)

    inc_id = inc2[0].incident_id if inc2 else ""
    return tr2.trace_id, inc_id


class TestMetricsAndAlertsAPI:
    """Integration tests for /metrics and /alerts endpoints (Milestone 9)."""

    def test_get_metrics_summary(self, client):
        """GET /metrics/summary returns high-level SLO metrics."""
        _seed_traces(client)

        resp = client.get("/metrics/summary")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_traces"] == 2
        assert "p50_latency_ms" in data
        assert "p95_latency_ms" in data
        assert "availability_rate" in data
        assert "quality_pass_rate" in data
        assert "estimated_cost_usd" in data

    def test_get_latency_breakdown_endpoint(self, client):
        """GET /metrics/latency/{trace_id} returns stage latency decomposition."""
        trace_id, _ = _seed_traces(client)

        resp = client.get(f"/metrics/latency/{trace_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == trace_id
        assert data["total_latency_ms"] > 0
        assert "router_ms" in data
        assert "retrieval_ms" in data
        assert "model_ms" in data

    def test_get_quality_and_cost_endpoints(self, client):
        """GET /metrics/quality and GET /metrics/cost return domain analytics."""
        _seed_traces(client)

        q_resp = client.get("/metrics/quality")
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        assert q_data["total_traces"] == 2
        assert q_data["availability_rate"] == 1.0

        c_resp = client.get("/metrics/cost")
        assert c_resp.status_code == 200
        c_data = c_resp.json()
        assert c_data["total_tokens"] > 0
        assert c_data["total_cost_usd"] > 0.0

    def test_get_alerts_endpoint(self, client):
        """GET /alerts detects P0 policy violations and fires alerts."""
        _seed_traces(client)

        resp = client.get("/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        assert len(alerts) >= 1
        # P0 policy bypass was seeded
        policy_alert = next((a for a in alerts if a["name"] == "CriticalPolicyBypass"), None)
        assert policy_alert is not None
        assert policy_alert["severity"] == "P0"
        assert policy_alert["status"] == "firing"


class TestModelAssistedAnalysisAPI:
    """Integration tests for model-assisted hypothesis endpoints (Milestone 10)."""

    def test_post_incident_model_hypothesis(self, client):
        """Milestone 10 exit criterion: POST /incidents/{id}/model-hypothesis produces traceable hypothesis."""
        trace_id, incident_id = _seed_traces(client)

        resp = client.post(f"/incidents/{incident_id}/model-hypothesis")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == trace_id
        assert data["incident_id"] == incident_id
        assert data["primary_root_cause"] == "policy_failure"
        assert data["confidence"] >= 0.9
        assert len(data["evidence_citations"]) >= 1
        assert "span_id" in data["evidence_citations"][0]
        assert "evidence_key" in data["evidence_citations"][0]
        assert "remediation" in data["recommended_remediation"].lower() or len(data["recommended_remediation"]) > 0

    def test_post_trace_model_hypothesis(self, client):
        """POST /traces/{id}/model-hypothesis produces grounded analysis directly from trace."""
        trace_id, _ = _seed_traces(client)

        resp = client.post(f"/traces/{trace_id}/model-hypothesis")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == trace_id
        assert data["primary_root_cause"] == "policy_failure"
        assert len(data["evidence_citations"]) >= 1
