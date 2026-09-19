"""Integration tests for Incident View, Timeline, Similar Incidents, and Web UI (Milestone 6)."""

import uuid
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.ingestion.service import IngestionService
from src.storage.database import DatabaseManager, get_database, reset_database
from src.storage.repository import TraceRepository


@pytest.fixture(autouse=True)
def setup_test_db():
    reset_database()
    import src.storage.database as db_module
    db_module._default_db = DatabaseManager(database_url="sqlite:///:memory:")
    db_module._default_db.init_db()
    yield
    reset_database()


@pytest.fixture
def client():
    return TestClient(app)


def _seed_incident_trace(failure_mode: FailureMode = FailureMode.RETRIEVAL_MISS) -> tuple[str, str]:
    """Runs an agent with failure mode and ingests trace, returning (trace_id, incident_id)."""
    agent = InstrumentedResearchAgent()
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
        failure_mode=failure_mode,
    )
    resp, trace = agent.run(req)

    db = get_database()
    ingestion = IngestionService(db)
    incidents = ingestion.ingest_trace(trace, auto_classify=True)

    inc_id = incidents[0].incident_id if incidents else ""
    return trace.trace_id, inc_id


class TestIncidentTimelineEndpoint:

    def test_incident_timeline_has_first_suspicious_flag(self, client):
        """Milestone 6: timeline flags the earliest suspicious event."""
        trace_id, inc_id = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)

        resp = client.get(f"/incidents/{inc_id}/timeline")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == trace_id
        assert data["first_suspicious_span_id"] is not None

        entries = data["entries"]
        assert len(entries) > 0

        # At least one entry must have is_first_suspicious=True
        suspicious_entries = [e for e in entries if e["is_first_suspicious"]]
        assert len(suspicious_entries) >= 1

    def test_incident_timeline_not_found(self, client):
        resp = client.get("/incidents/nonexistent/timeline")
        assert resp.status_code == 404


class TestSimilarIncidentsEndpoint:

    def test_similar_incidents_search(self, client):
        """Milestone 6: similar incidents matching failure signature."""
        # Seed two incidents of the same category
        _, inc_id_1 = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)
        _, inc_id_2 = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)

        resp = client.get(f"/incidents/{inc_id_1}/similar")
        assert resp.status_code == 200
        data = resp.json()

        assert data["target_incident_id"] == inc_id_1
        assert data["total"] >= 1
        assert data["similar_incidents"][0]["incident_id"] == inc_id_2
        assert data["similar_incidents"][0]["similarity_score"] >= 0.8


class TestEvidenceGraphEndpoint:

    def test_evidence_graph_structure(self, client):
        """Milestone 6: causal evidence graph returns nodes and edges."""
        trace_id, inc_id = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)

        resp = client.get(f"/incidents/{inc_id}/evidence-graph")
        assert resp.status_code == 200
        data = resp.json()

        assert data["incident_id"] == inc_id
        assert len(data["nodes"]) >= 2
        assert len(data["edges"]) >= 1

        node_types = {n["type"] for n in data["nodes"]}
        assert "hypothesis" in node_types
        assert "evidence" in node_types


class TestWebUIEndpoints:

    def test_incident_view_ui_renders(self, client):
        """Milestone 6 exit criterion: reviewer can explain an incident from the UI."""
        trace_id, inc_id = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)

        resp = client.get(f"/ui/incidents/{inc_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text

        assert "Incident Review" in body
        assert inc_id[:12] in body
        assert "First Suspicious Event" in body
        assert "Supporting Evidence Items" in body
        assert "Similar Historical Incidents" in body

    def test_dashboard_ui_renders(self, client):
        trace_id, inc_id = _seed_incident_trace(FailureMode.RETRIEVAL_MISS)

        resp = client.get("/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Reliability &amp; Forensics Dashboard" in resp.text or "Reliability & Forensics Dashboard" in resp.text
