"""Integration tests for the Replay Pipeline (Milestone 7).

Verifies the end-to-end replay workflow:
Trace with failure -> Incidents persisted -> POST /replay API -> Fixed verdict & metrics diff.
"""

import pytest
from fastapi.testclient import TestClient

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.api.main import app
from src.ingestion.service import IngestionService
from src.storage.database import DatabaseManager, get_database, reset_database


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


def _seed_failure_trace(failure_mode: FailureMode = FailureMode.RETRIEVAL_MISS) -> str:
    agent = InstrumentedResearchAgent()
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
        failure_mode=failure_mode,
    )
    _, trace = agent.run(req)
    ingestion = IngestionService(get_database())
    ingestion.ingest_trace(trace, auto_classify=True)
    return trace.trace_id


class TestReplayPipelineAPI:

    def test_post_replay_controlled_fix(self, client):
        """Milestone 7: POST /replay fixes retrieval failure with retriever override."""
        trace_id = _seed_failure_trace(FailureMode.RETRIEVAL_MISS)

        payload = {
            "source_trace_id": trace_id,
            "replay_type": "controlled",
            "retriever_mode": "fixed",
        }

        resp = client.post("/replay", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["source_trace_id"] == trace_id
        assert data["replayed_trace_id"] is not None
        assert data["verdict"] == "fixed"
        assert "FIXED" in data["summary"]

        diff = data["metrics_diff"]
        assert diff is not None
        assert "retrieval_failure" in diff["resolved_incident_categories"]
        assert diff["replayed_outcome"] == "success"

        # Check replay retrieval endpoint
        replay_id = data["replay_id"]
        get_resp = client.get(f"/replay/{replay_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["replay_id"] == replay_id

        # Check list trace replays endpoint
        list_resp = client.get(f"/traces/{trace_id}/replays")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1

    def test_post_replay_invalid_trace(self, client):
        payload = {
            "source_trace_id": "nonexistent_trace_id",
            "replay_type": "controlled",
        }
        resp = client.post("/replay", json=payload)
        assert resp.status_code == 404

    def test_post_replay_invalid_type(self, client):
        payload = {
            "source_trace_id": "some_id",
            "replay_type": "invalid_type",
        }
        resp = client.post("/replay", json=payload)
        assert resp.status_code == 400
