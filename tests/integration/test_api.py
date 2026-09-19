"""Tests for TraceSleuth API — Trace Explorer (Milestone 3).

Uses FastAPI TestClient to verify all trace and incident endpoints.
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.domain.incident import (
    Evidence,
    FailureCategory,
    Incident,
    IncidentSeverity,
)
from src.domain.trace import (
    SpanData,
    SpanStatus,
    SpanType,
    TraceData,
    TraceEvent,
    TraceOutcome,
)
from src.storage.database import DatabaseManager, get_database, reset_database
from src.storage.repository import TraceRepository


@pytest.fixture(autouse=True)
def setup_test_db():
    """Set up an in-memory test database for each test."""
    reset_database()
    # Override the database URL to use in-memory SQLite
    import src.storage.database as db_module
    db_module._default_db = DatabaseManager(database_url="sqlite:///:memory:")
    db_module._default_db.init_db()
    yield
    reset_database()


@pytest.fixture
def client():
    return TestClient(app)


def _seed_trace(
    trace_id: str = None,
    outcome: TraceOutcome = TraceOutcome.SUCCESS,
    with_failure: bool = False,
) -> TraceData:
    """Create and persist a test trace."""
    t_id = trace_id or uuid.uuid4().hex[:32]
    now = datetime.now(timezone.utc)

    spans = [
        SpanData(
            span_id="span_root",
            trace_id=t_id,
            name="agent.run",
            span_type=SpanType.AGENT_RUN,
            start_time=now,
            end_time=now + timedelta(seconds=1),
            duration_ms=1000.0,
            status=SpanStatus.OK,
            attributes={"tracesleuth.request_id": f"req_{t_id[:8]}"},
        ),
        SpanData(
            span_id="span_retrieval",
            trace_id=t_id,
            parent_span_id="span_root",
            name="retrieval.search",
            span_type=SpanType.RETRIEVAL,
            start_time=now + timedelta(milliseconds=10),
            end_time=now + timedelta(milliseconds=100),
            duration_ms=90.0,
            status=SpanStatus.OK,
            attributes={"retrieval.candidate_count": 0 if with_failure else 5},
            events=[TraceEvent(name="retrieval_empty")] if with_failure else [],
        ),
        SpanData(
            span_id="span_model",
            trace_id=t_id,
            parent_span_id="span_root",
            name="model.generate",
            span_type=SpanType.MODEL,
            start_time=now + timedelta(milliseconds=100),
            end_time=now + timedelta(milliseconds=500),
            duration_ms=400.0,
            status=SpanStatus.OK,
            attributes={
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 200,
            },
        ),
    ]

    trace = TraceData(
        trace_id=t_id,
        request_id=f"req_{t_id[:8]}",
        environment="test",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        duration_ms=1000.0,
        outcome=outcome,
        root_span_id="span_root",
        spans=spans,
    )

    db = get_database()
    session = db.get_session()
    repo = TraceRepository(session)
    repo.save_trace(trace)
    session.close()

    return trace


class TestHealthEndpoint:

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestTraceListEndpoint:

    def test_list_empty(self, client):
        resp = client.get("/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["traces"] == []

    def test_list_with_traces(self, client):
        _seed_trace(trace_id="list_test_001")
        _seed_trace(trace_id="list_test_002")

        resp = client.get("/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["traces"]) == 2

    def test_list_filter_by_outcome(self, client):
        _seed_trace(trace_id="ok_trace", outcome=TraceOutcome.SUCCESS)
        _seed_trace(trace_id="fail_trace", outcome=TraceOutcome.FAILURE)

        resp = client.get("/traces?outcome=failure")
        data = resp.json()
        assert data["total"] == 1
        assert data["traces"][0]["trace_id"] == "fail_trace"

    def test_list_pagination(self, client):
        for i in range(5):
            _seed_trace(trace_id=f"page_{i:03d}")

        resp = client.get("/traces?limit=2&offset=0")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["traces"]) == 2


class TestTraceDetailEndpoint:

    def test_get_trace_full(self, client):
        """Milestone 3 exit criterion: engineer can inspect a complete trace."""
        _seed_trace(trace_id="detail_test_001")

        resp = client.get("/traces/detail_test_001")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == "detail_test_001"
        assert data["span_count"] == 3
        assert len(data["spans"]) == 3

        # Verify span details are present
        span_names = {s["name"] for s in data["spans"]}
        assert "agent.run" in span_names
        assert "retrieval.search" in span_names
        assert "model.generate" in span_names

        # Verify span attributes preserved
        model_span = next(s for s in data["spans"] if s["name"] == "model.generate")
        assert model_span["attributes"]["gen_ai.usage.input_tokens"] == 500

    def test_get_trace_not_found(self, client):
        resp = client.get("/traces/nonexistent")
        assert resp.status_code == 404


class TestTimelineEndpoint:

    def test_trace_timeline(self, client):
        """Timeline shows chronological span starts, ends, and events."""
        _seed_trace(trace_id="timeline_test", with_failure=True)

        resp = client.get("/traces/timeline_test/timeline")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == "timeline_test"
        entries = data["entries"]
        assert len(entries) > 0

        # Should have span_start, span_end, and event entries
        event_types = {e["event_type"] for e in entries}
        assert "span_start" in event_types
        assert "span_end" in event_types
        assert "event" in event_types  # retrieval_empty event

    def test_timeline_not_found(self, client):
        resp = client.get("/traces/nonexistent/timeline")
        assert resp.status_code == 404


class TestAnalyzeEndpoint:

    def test_analyze_failure_trace(self, client):
        """Analyzing a trace with failures creates incidents."""
        _seed_trace(trace_id="analyze_test", with_failure=True)

        resp = client.post("/traces/analyze_test/analyze")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trace_id"] == "analyze_test"
        assert data["incidents_created"] >= 1
        assert len(data["incidents"]) >= 1

        # Verify incident structure
        inc = data["incidents"][0]
        assert "incident_id" in inc
        assert "category" in inc
        assert "hypothesis_score" in inc
        assert "evidence" in inc

    def test_analyze_healthy_trace(self, client):
        """A healthy trace should produce no incidents."""
        _seed_trace(trace_id="healthy_test")

        resp = client.post("/traces/healthy_test/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert data["incidents_created"] == 0


class TestIncidentEndpoints:

    def test_list_incidents(self, client):
        """Incidents created by analysis are queryable."""
        _seed_trace(trace_id="inc_list_test", with_failure=True)
        client.post("/traces/inc_list_test/analyze")

        resp = client.get("/incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_get_incident_detail(self, client):
        """Individual incident can be inspected with full evidence."""
        _seed_trace(trace_id="inc_detail_test", with_failure=True)
        analyze_resp = client.post("/traces/inc_detail_test/analyze")
        incidents = analyze_resp.json()["incidents"]

        if incidents:
            inc_id = incidents[0]["incident_id"]
            resp = client.get(f"/incidents/{inc_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["incident_id"] == inc_id
            assert "evidence" in data

    def test_update_incident_status(self, client):
        """Incident status can be updated through lifecycle."""
        _seed_trace(trace_id="inc_update_test", with_failure=True)
        analyze_resp = client.post("/traces/inc_update_test/analyze")
        incidents = analyze_resp.json()["incidents"]

        if incidents:
            inc_id = incidents[0]["incident_id"]

            # Confirm the incident
            resp = client.patch(f"/incidents/{inc_id}", json={"status": "confirmed"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "confirmed"

            # Resolve it
            resp = client.patch(f"/incidents/{inc_id}", json={"status": "resolved"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "resolved"

    def test_invalid_status_update(self, client):
        """Invalid status value returns 400."""
        _seed_trace(trace_id="inc_invalid_test", with_failure=True)
        analyze_resp = client.post("/traces/inc_invalid_test/analyze")
        incidents = analyze_resp.json()["incidents"]

        if incidents:
            inc_id = incidents[0]["incident_id"]
            resp = client.patch(f"/incidents/{inc_id}", json={"status": "invalid_status"})
            assert resp.status_code == 400

    def test_filter_incidents_by_category(self, client):
        """Incidents can be filtered by failure category."""
        _seed_trace(trace_id="inc_filter_test", with_failure=True)
        client.post("/traces/inc_filter_test/analyze")

        resp = client.get("/incidents?category=retrieval_failure")
        assert resp.status_code == 200

    def test_incident_not_found(self, client):
        resp = client.get("/incidents/nonexistent")
        assert resp.status_code == 404
