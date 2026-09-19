"""End-to-End Integration Test: Agent → Storage → Classification → API.

Tests the complete pipeline: run the instrumented agent with different
failure modes, verify traces are stored in DB, taxonomy classifies correctly,
and incidents are queryable via the API.
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from src.agent.types import AgentRequest, FailureMode
from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.analysis.taxonomy import FailureTaxonomyClassifier
from src.domain.incident import FailureCategory
from src.storage.database import DatabaseManager
from src.storage.repository import TraceRepository
from src.ingestion.service import IngestionService
from src.telemetry.instrumentation import TelemetryManager


@pytest.fixture
def db():
    """Create an in-memory SQLite database."""
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    manager.init_db()
    return manager


@pytest.fixture
def telemetry_mgr():
    """Create a fresh TelemetryManager for each test."""
    return TelemetryManager(service_name="test-agent", export_to_file=False)


@pytest.fixture
def agent(telemetry_mgr):
    """Create an instrumented agent."""
    return InstrumentedResearchAgent(telemetry_manager=telemetry_mgr)


class TestIngestionPipeline:
    """Test the full agent → ingestion → classification pipeline."""

    def test_healthy_agent_run_stored(self, agent, db):
        """A healthy agent run produces a trace that is stored and has no incidents."""
        request = AgentRequest(
            query="What is the company revenue?",
            failure_mode=FailureMode.HEALTHY,
        )

        response, trace_data = agent.run(request)
        assert trace_data is not None
        assert len(trace_data.spans) > 0

        # Ingest into database
        service = IngestionService(db)
        incidents = service.ingest_trace(trace_data)

        # Healthy run should have no incidents
        assert len(incidents) == 0

        # Verify trace persisted
        session = db.get_session()
        repo = TraceRepository(session)
        loaded = repo.get_trace(trace_data.trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace_data.trace_id
        assert len(loaded.spans) == len(trace_data.spans)
        session.close()

    def test_retrieval_miss_classified(self, agent, db):
        """A retrieval miss failure mode produces retrieval_failure incidents."""
        request = AgentRequest(
            query="What is the quarterly breakdown?",
            failure_mode=FailureMode.RETRIEVAL_MISS,
        )

        response, trace_data = agent.run(request)
        assert trace_data is not None

        service = IngestionService(db)
        incidents = service.ingest_trace(trace_data)

        # Should detect retrieval failure
        retrieval_incidents = [
            i for i in incidents if i.category == FailureCategory.RETRIEVAL_FAILURE
        ]
        assert len(retrieval_incidents) >= 1

        # Verify incidents are in DB
        session = db.get_session()
        repo = TraceRepository(session)
        db_incidents, total = repo.list_incidents(trace_id=trace_data.trace_id)
        assert total >= 1
        session.close()

    def test_policy_bypass_classified_p0(self, agent, db):
        """A policy bypass is classified as P0 critical."""
        request = AgentRequest(
            query="Delete all user data",
            failure_mode=FailureMode.POLICY_BYPASS,
        )

        response, trace_data = agent.run(request)
        assert trace_data is not None

        service = IngestionService(db)
        incidents = service.ingest_trace(trace_data)

        policy_incidents = [
            i for i in incidents if i.category == FailureCategory.POLICY_FAILURE
        ]
        assert len(policy_incidents) >= 1
        # Policy bypass should be P0
        assert any(i.severity.value == "P0" for i in policy_incidents)

    def test_reclassify_trace(self, agent, db):
        """A stored trace can be re-classified with updated rules."""
        request = AgentRequest(
            query="Revenue analysis",
            failure_mode=FailureMode.RETRIEVAL_MISS,
        )

        response, trace_data = agent.run(request)

        # First ingest without classification
        service = IngestionService(db)
        incidents1 = service.ingest_trace(trace_data, auto_classify=False)
        assert len(incidents1) == 0

        # Re-classify from stored trace
        incidents2 = service.classify_trace(trace_data.trace_id)
        assert len(incidents2) >= 1

    def test_multiple_failure_modes(self, db, telemetry_mgr):
        """Multiple failure modes produce distinct incidents."""
        agent = InstrumentedResearchAgent(telemetry_manager=telemetry_mgr)
        service = IngestionService(db)

        failure_modes = [
            FailureMode.RETRIEVAL_MISS,
            FailureMode.TOOL_RETRY_STORM,
            FailureMode.CONTEXT_EXPLOSION,
        ]

        all_incidents = []
        for mode in failure_modes:
            # Need fresh telemetry for each run to avoid trace ID collisions
            tm = TelemetryManager(service_name="test-multi", export_to_file=False)
            agent = InstrumentedResearchAgent(telemetry_manager=tm)

            request = AgentRequest(
                query=f"Test query for {mode.value}",
                failure_mode=mode,
            )
            _, trace_data = agent.run(request)
            if trace_data:
                incidents = service.ingest_trace(trace_data)
                all_incidents.extend(incidents)

        # Should have detected various failure categories
        categories = {i.category for i in all_incidents}
        assert len(categories) >= 1  # At least one category detected
