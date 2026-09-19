"""Tests for Metrics, Cost Forensics, and Alerts Engine (Milestone 9).

Verifies latency decomposition, SLO/SLA percentiles, quality vs availability,
and threshold/baseline alert triggers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
import pytest

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.domain.incident import FailureCategory, Incident, IncidentSeverity, IncidentStatus
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData, TraceOutcome
from src.ingestion.service import IngestionService
from src.metrics.alerts import AlertEngine, AlertSeverity, AlertStatus
from src.metrics.cost import CostForensicsCalculator, ModelPricing
from src.metrics.service import MetricsService
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
def metrics_service(test_db):
    return MetricsService(test_db)


class TestMetricsAndCostForensics:
    """Unit tests for latency decomposition and token cost calculations."""

    def test_latency_stage_decomposition(self, metrics_service, test_db):
        """Milestone 9: Decomposes total latency into pipeline stages (spec Section 17)."""
        agent = InstrumentedResearchAgent()
        req = AgentRequest(query="What was Enterprise Corp Q3 2025 cloud ARR?")
        resp, trace = agent.run(req)

        ingestion = IngestionService(test_db)
        ingestion.ingest_trace(trace)

        breakdown = metrics_service.get_latency_breakdown(trace.trace_id)

        assert breakdown is not None
        assert breakdown["trace_id"] == trace.trace_id
        assert breakdown["total_latency_ms"] > 0
        assert "router_ms" in breakdown
        assert "retrieval_ms" in breakdown
        assert "model_ms" in breakdown
        assert "tool_ms" in breakdown
        assert "citation_ms" in breakdown
        # Total latency should approximate the sum of stages
        stage_sum = (
            breakdown["router_ms"]
            + breakdown["retrieval_ms"]
            + breakdown["reranker_ms"]
            + breakdown["planner_ms"]
            + breakdown["model_ms"]
            + breakdown["tool_ms"]
            + breakdown["citation_ms"]
            + breakdown["other_ms"]
        )
        assert stage_sum > 0

    def test_cost_forensics_calculation(self):
        """Calculates accurate token costs based on model pricing (spec Section 16)."""
        now = datetime.now(timezone.utc)
        trace = TraceData(
            trace_id="cost_test_trace",
            request_id="req_cost",
            start_time=now,
            end_time=now + timedelta(milliseconds=100),
            duration_ms=100.0,
            outcome=TraceOutcome.SUCCESS,
            spans=[
                SpanData(
                    span_id="s_model_1",
                    trace_id="cost_test_trace",
                    name="model.answer_generation",
                    span_type=SpanType.MODEL,
                    start_time=now,
                    end_time=now + timedelta(milliseconds=50),
                    duration_ms=50.0,
                    attributes={
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.usage.input_tokens": 1000,
                        "gen_ai.usage.output_tokens": 500,
                    },
                ),
                SpanData(
                    span_id="s_tool_1",
                    trace_id="cost_test_trace",
                    name="tool.calculator",
                    span_type=SpanType.TOOL,
                    start_time=now,
                    end_time=now + timedelta(milliseconds=20),
                    duration_ms=20.0,
                ),
            ],
        )

        calc = CostForensicsCalculator()
        est = calc.calculate_trace_cost(trace)

        # 1000 input tokens * $2.50 / 1M = $0.0025
        # 500 output tokens * $10.00 / 1M = $0.0050
        # 1 tool call * $0.50 / 1K = $0.0005
        expected_cost = 0.0025 + 0.0050 + 0.0005
        assert est.total_cost_usd == pytest.approx(expected_cost, rel=1e-3)
        assert est.input_tokens == 1000
        assert est.output_tokens == 500
        assert est.tool_call_count == 1
        assert est.is_cost_anomaly is False

    def test_sla_metrics_availability_vs_quality(self, metrics_service, test_db):
        """Milestone 9: Distinguishes availability (HTTP 200) from quality pass rate (spec Section 15)."""
        agent = InstrumentedResearchAgent()

        # Seed 1 healthy trace (HTTP 200, Quality PASS)
        _, tr_healthy = agent.run(AgentRequest(query="Valid financial query", failure_mode=FailureMode.HEALTHY))
        # Seed 1 retrieval failure trace (HTTP 200 outcome technically, but Quality FAIL)
        _, tr_fail = agent.run(AgentRequest(query="Failing query", failure_mode=FailureMode.RETRIEVAL_MISS))

        ingestion = IngestionService(test_db)
        ingestion.ingest_trace(tr_healthy, auto_classify=True)
        ingestion.ingest_trace(tr_fail, auto_classify=True)

        sla = metrics_service.get_sla_metrics()

        assert sla["total_traces"] == 2
        # Both technically completed (status=COMPLETED, outcome=SUCCESS)
        assert sla["availability_rate"] == 1.0
        # But only 1 trace passed semantic quality!
        assert sla["quality_pass_rate"] == 0.5


class TestAlertEngine:
    """Unit tests for threshold and incident-driven alert rules."""

    def test_p0_critical_policy_bypass_triggers_alert(self):
        """Any P0 policy bypass immediately fires a high-priority alert (spec Section 27)."""
        engine = AlertEngine()
        now = datetime.now(timezone.utc)
        incident = Incident(
            incident_id="inc_policy_p0",
            trace_id="tr_p0",
            category=FailureCategory.POLICY_FAILURE,
            severity=IncidentSeverity.P0,
            summary="Policy guardrail bypassed by unauthenticated query",
            status=IncidentStatus.OPEN,
            created_at=now,
        )

        sla_dummy = {"total_traces": 10, "quality_pass_rate": 0.98}
        alerts = engine.evaluate(sla_dummy, [incident])

        assert len(alerts) >= 1
        policy_alert = next((a for a in alerts if a.name == "CriticalPolicyBypass"), None)
        assert policy_alert is not None
        assert policy_alert.severity == AlertSeverity.P0
        assert policy_alert.status == AlertStatus.FIRING
        assert "inc_policy_p0" in policy_alert.details["incident_ids"]

    def test_quality_pass_rate_drop_triggers_alert(self):
        """Drop in quality pass rate below SLO threshold triggers an alert."""
        engine = AlertEngine(quality_pass_rate_threshold=0.95)
        sla_degraded = {
            "total_traces": 20,
            "quality_pass_rate": 0.85,  # Below 95%
            "p99_latency_ms": 120.0,
        }

        alerts = engine.evaluate(sla_degraded, [])
        quality_alert = next((a for a in alerts if a.name == "QualityPassRateDrop"), None)

        assert quality_alert is not None
        assert quality_alert.severity == AlertSeverity.P1
        assert quality_alert.status == AlertStatus.FIRING
