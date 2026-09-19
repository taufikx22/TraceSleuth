"""Tests for Regression Conversion and CI Dataset Export (Milestone 8).

Verifies the failure-to-regression loop:
  Confirmed Incident -> Regression Case -> CI Dataset -> Continuous Release Protection.
"""

from __future__ import annotations

import yaml
import pytest

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.domain.incident import FailureCategory, IncidentStatus
from src.domain.regression import (
    AssertionOperator,
    RegressionAssertion,
    RegressionStatus,
)
from src.ingestion.service import IngestionService
from src.regression.service import RegressionService
from src.storage.database import DatabaseManager, reset_database
from src.storage.repository import TraceRepository


@pytest.fixture(autouse=True)
def test_db():
    reset_database()
    import src.storage.database as db_module
    db_module._default_db = DatabaseManager(database_url="sqlite:///:memory:")
    db_module._default_db.init_db()
    yield db_module._default_db
    reset_database()


@pytest.fixture
def regression_service(test_db):
    return RegressionService(test_db)


def _seed_incident_with_failure(test_db, failure_mode: FailureMode = FailureMode.RETRIEVAL_MISS) -> str:
    agent = InstrumentedResearchAgent()
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
        failure_mode=failure_mode,
    )
    resp, trace = agent.run(req)

    ingestion = IngestionService(test_db)
    incidents = ingestion.ingest_trace(trace, auto_classify=True)
    assert len(incidents) > 0, f"Expected incidents for {failure_mode.value}"
    return incidents[0].incident_id


class TestRegressionConversion:
    """Unit tests for converting incidents into persistent regression cases."""

    def test_create_regression_case_from_incident(self, regression_service, test_db):
        """Milestone 8 exit criterion: Confirmed incident becomes a test case."""
        inc_id = _seed_incident_with_failure(test_db, FailureMode.RETRIEVAL_MISS)

        case = regression_service.create_case_from_incident(
            incident_id=inc_id,
            owner="ai-reliability-team",
            introduced_version="1.2.0",
        )

        assert case is not None
        assert case.id.startswith("reg_")
        assert case.incident_id == inc_id
        assert case.failure_type == "retrieval_failure"
        assert case.owner == "ai-reliability-team"
        assert case.introduced_version == "1.2.0"
        assert case.status == RegressionStatus.ACTIVE
        assert len(case.assertions) >= 1
        assert "retrieval" in case.assertions[0].target_metric

        # Verify saved in database
        session = test_db.get_session()
        repo = TraceRepository(session)
        loaded = repo.get_regression_case(case.id)
        session.close()

        assert loaded is not None
        assert loaded.id == case.id
        assert loaded.query == case.query

    def test_run_regression_case_passes_on_healthy_execution(self, regression_service, test_db):
        """Regression test passes when the agent runs cleanly without the failure mode."""
        inc_id = _seed_incident_with_failure(test_db, FailureMode.RETRIEVAL_MISS)
        case = regression_service.create_case_from_incident(inc_id)

        # Run test case against healthy agent
        result = regression_service.run_case(case.id)

        assert result is not None
        assert result.case_id == case.id
        assert result.passed is True
        assert len(result.assertion_results) >= 1
        assert all(a.passed for a in result.assertion_results)

        # Case status should be updated to PASSED
        session = test_db.get_session()
        repo = TraceRepository(session)
        updated_case = repo.get_regression_case(case.id)
        session.close()

        assert updated_case.status == RegressionStatus.PASSED

    def test_run_regression_case_fails_if_defect_reproduced(self, regression_service, test_db):
        """Regression test catches the defect and fails if the agent repeats the failure."""
        inc_id = _seed_incident_with_failure(test_db, FailureMode.RETRIEVAL_MISS)
        case = regression_service.create_case_from_incident(inc_id)

        # Mock an agent that repeats the retrieval miss defect
        failing_agent = InstrumentedResearchAgent()
        original_run = failing_agent.run

        def failing_run(req):
            req.failure_mode = FailureMode.RETRIEVAL_MISS
            return original_run(req)

        failing_agent.run = failing_run

        result = regression_service.run_case(case.id, agent=failing_agent)

        assert result is not None
        assert result.passed is False
        assert any(not a.passed for a in result.assertion_results)

    def test_export_ci_dataset_yaml_schema(self, regression_service, test_db):
        """Exports regression cases matching the schema in spec Section 13."""
        inc_id = _seed_incident_with_failure(test_db, FailureMode.RETRIEVAL_MISS)
        regression_service.create_case_from_incident(inc_id)

        yaml_content = regression_service.export_ci_dataset(format="yaml")
        assert yaml_content is not None
        parsed = yaml.safe_load(yaml_content)

        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        entry = parsed[0]

        # Spec Section 13 fields:
        assert "id" in entry
        assert "source_trace" in entry
        assert "failure_type" in entry
        assert "expected_evidence" in entry
        assert "assertions" in entry
        assert "owner" in entry
        assert "introduced_version" in entry
        assert "status" in entry
