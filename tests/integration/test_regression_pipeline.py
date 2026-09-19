"""Integration tests for Regression Conversion API endpoints (Milestone 8).

Verifies creating regression cases, executing tests, and exporting CI datasets via FastAPI.
"""

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


def _seed_failure_incident(client) -> tuple[str, str]:
    import src.storage.database as db_module
    db = db_module._default_db

    agent = InstrumentedResearchAgent()
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR?",
        failure_mode=FailureMode.RETRIEVAL_MISS,
    )
    _, trace = agent.run(req)

    ingestion = IngestionService(db)
    incidents = ingestion.ingest_trace(trace, auto_classify=True)
    return trace.trace_id, incidents[0].incident_id


class TestRegressionAPI:
    """Integration tests for /regression and /incidents/{id}/regression endpoints."""

    def test_convert_incident_to_regression_case(self, client):
        """Milestone 8 exit criterion: POST /incidents/{id}/regression creates a test case."""
        trace_id, incident_id = _seed_failure_incident(client)

        resp = client.post(
            f"/incidents/{incident_id}/regression",
            json={"owner": "qa-platform", "introduced_version": "1.3.0"},
        )
        assert resp.status_code == 201
        data = resp.json()

        assert data["id"].startswith("reg_")
        assert data["incident_id"] == incident_id
        assert data["source_trace_id"] == trace_id
        assert data["failure_type"] == "retrieval_failure"
        assert data["owner"] == "qa-platform"
        assert data["status"] == "active"
        assert len(data["assertions"]) >= 1

    def test_list_and_get_regression_cases(self, client):
        """GET /regression/cases and GET /regression/cases/{id} retrieve cases."""
        _, incident_id = _seed_failure_incident(client)
        create_resp = client.post(f"/incidents/{incident_id}/regression")
        case_id = create_resp.json()["id"]

        list_resp = client.get("/regression/cases")
        assert list_resp.status_code == 200
        cases = list_resp.json()
        assert len(cases) >= 1
        assert any(c["id"] == case_id for c in cases)

        get_resp = client.get(f"/regression/cases/{case_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == case_id

    def test_run_regression_case_endpoint(self, client):
        """POST /regression/cases/{id}/run executes test and updates status."""
        _, incident_id = _seed_failure_incident(client)
        create_resp = client.post(f"/incidents/{incident_id}/regression")
        case_id = create_resp.json()["id"]

        run_resp = client.post(f"/regression/cases/{case_id}/run")
        assert run_resp.status_code == 200
        run_data = run_resp.json()

        assert run_data["case_id"] == case_id
        assert run_data["passed"] is True
        assert len(run_data["assertion_results"]) >= 1
        assert "PASSED" in run_data["summary"]

    def test_export_ci_dataset_yaml_and_json(self, client):
        """GET /regression/export returns formatted CI datasets."""
        _, incident_id = _seed_failure_incident(client)
        client.post(f"/incidents/{incident_id}/regression")

        # YAML format
        yaml_resp = client.get("/regression/export?format=yaml")
        assert yaml_resp.status_code == 200
        assert "application/x-yaml" in yaml_resp.headers["content-type"]
        assert "failure_type" in yaml_resp.text

        # JSON format
        json_resp = client.get("/regression/export?format=json")
        assert json_resp.status_code == 200
        assert "application/json" in json_resp.headers["content-type"]
        data = json_resp.json()
        assert "regression_cases" in data
        assert len(data["regression_cases"]) >= 1
