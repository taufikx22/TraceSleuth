"""Tests for Trace Replay Engine (Milestone 7).

Verifies exact replay, controlled replay with configuration overrides,
metrics diff calculations, and verdict determination.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
import pytest

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.domain.trace import TraceOutcome
from src.ingestion.service import IngestionService
from src.replay.service import ReplayService
from src.replay.types import ReplayConfig, ReplayType, ReplayVerdict
from src.storage.database import DatabaseManager, get_database, reset_database
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
def replay_service(test_db):
    return ReplayService(test_db)


def _seed_trace_with_failure(test_db, failure_mode: FailureMode = FailureMode.RETRIEVAL_MISS) -> str:
    agent = InstrumentedResearchAgent()
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
        failure_mode=failure_mode,
    )
    resp, trace = agent.run(req)

    ingestion = IngestionService(test_db)
    ingestion.ingest_trace(trace, auto_classify=True)
    return trace.trace_id


class TestReplayService:

    def test_exact_replay_reproduces_failure(self, replay_service, test_db):
        """Exact replay with same configuration should reproduce the failure."""
        trace_id = _seed_trace_with_failure(test_db, FailureMode.RETRIEVAL_MISS)

        config = ReplayConfig(replay_type=ReplayType.EXACT)
        run = replay_service.execute_replay(trace_id, config=config)

        assert run.replay_id.startswith("rpl_")
        assert run.source_trace_id == trace_id
        assert run.replayed_trace_id is not None
        assert run.verdict in (ReplayVerdict.IDENTICAL, ReplayVerdict.STILL_FAILING)

    def test_controlled_replay_fixes_retrieval_failure(self, replay_service, test_db):
        """Milestone 7 exit criterion: testing a configuration change against a historical trace.

        Given a trace with a retrieval failure, a controlled replay with a fixed retriever
        resolves the failure and produces a 'FIXED' verdict.
        """
        trace_id = _seed_trace_with_failure(test_db, FailureMode.RETRIEVAL_MISS)

        # Apply controlled fix: healthy retriever
        config = ReplayConfig(
            replay_type=ReplayType.CONTROLLED,
            retriever_mode="fixed",
        )
        run = replay_service.execute_replay(trace_id, config=config)

        assert run.verdict == ReplayVerdict.FIXED
        assert run.metrics_diff is not None
        assert "retrieval_failure" in run.metrics_diff.resolved_incident_categories
        assert run.metrics_diff.replayed_outcome == "success"

    def test_controlled_replay_persistence(self, replay_service, test_db):
        """Replay runs are persisted to database and retrievable."""
        trace_id = _seed_trace_with_failure(test_db, FailureMode.RETRIEVAL_MISS)

        config = ReplayConfig(retriever_mode="fixed")
        run = replay_service.execute_replay(trace_id, config=config)

        session = test_db.get_session()
        repo = TraceRepository(session)
        loaded = repo.get_replay(run.replay_id)
        assert loaded is not None
        assert loaded.replay_id == run.replay_id
        assert loaded.source_trace_id == trace_id
        assert loaded.verdict == ReplayVerdict.FIXED

        # List replays for trace
        replays = repo.list_replays(source_trace_id=trace_id)
        assert len(replays) == 1
        session.close()

    def test_replay_nonexistent_trace_raises(self, replay_service):
        with pytest.raises(ValueError, match="Source trace nonexistent not found"):
            replay_service.execute_replay("nonexistent")
