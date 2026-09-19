"""Tests for TraceSleuth Root-Cause Analysis (RCA) Engine (Milestone 5).

Verifies the 8+ forensic rules, evidence graph construction,
earliest-suspicious-span isolation, and confidence scoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
import pytest

from src.analysis.engine import RootCauseEngine
from src.analysis.rules import (
    ContextExplosionRule,
    EvaluationBlindSpotRule,
    GenerationContradictionRule,
    PolicyBypassRule,
    RankingAnomalyRule,
    RetrievalMissRule,
    TimeoutSLABreachRule,
    ToolRetryStormRule,
)
from src.domain.incident import FailureCategory, IncidentSeverity
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData, TraceEvent, TraceOutcome


@pytest.fixture
def rca():
    return RootCauseEngine()


def _make_base_trace(trace_id: str = None) -> TraceData:
    t_id = trace_id or uuid.uuid4().hex[:32]
    now = datetime.now(timezone.utc)
    return TraceData(
        trace_id=t_id,
        request_id=f"req_{t_id[:8]}",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        duration_ms=1000.0,
        outcome=TraceOutcome.SUCCESS,
        spans=[],
    )


def _add_span(
    trace: TraceData,
    name: str,
    span_type: SpanType,
    parent_span_id: str = None,
    status: SpanStatus = SpanStatus.OK,
    attributes: dict = None,
    events: list = None,
    duration_ms: float = 20.0,
) -> SpanData:
    now = datetime.now(timezone.utc)
    span = SpanData(
        span_id=f"span_{len(trace.spans):04d}",
        trace_id=trace.trace_id,
        parent_span_id=parent_span_id,
        name=name,
        span_type=span_type,
        start_time=now,
        end_time=now + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        status=status,
        attributes=attributes or {},
        events=events or [],
    )
    trace.spans.append(span)
    return span


class TestForensicRules:
    """Unit tests for each of the 8 deterministic forensic diagnosis rules."""

    def test_retrieval_miss_rule(self):
        trace = _make_base_trace()
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL, attributes={"retrieval.candidate_count": 0})
        rule = RetrievalMissRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.RETRIEVAL_FAILURE
        assert res.confidence >= 0.9
        assert len(res.evidence) == 1
        assert "retrieval_empty" in res.evidence[0].type

    def test_ranking_anomaly_rule(self):
        trace = _make_base_trace()
        _add_span(
            trace,
            "reranker.filter",
            SpanType.RERANKER,
            attributes={"reranker.input_count": 5, "reranker.output_count": 0},
        )
        rule = RankingAnomalyRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.RANKING_FAILURE
        assert res.confidence == 0.90
        assert res.evidence[0].type == "reranker_zero_output"

    def test_generation_contradiction_rule(self):
        trace = _make_base_trace()
        _add_span(
            trace,
            "citation.attribution",
            SpanType.CITATION,
            attributes={"citation.invalid_count": 2},
        )
        rule = GenerationContradictionRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.GENERATION_FAILURE
        assert "citation_hallucination" in res.evidence[0].type

    def test_tool_retry_storm_rule(self):
        trace = _make_base_trace()
        _add_span(
            trace,
            "tool.calculator",
            SpanType.TOOL,
            events=[
                TraceEvent(name="retry", attributes={"attempt": 1}),
                TraceEvent(name="retry", attributes={"attempt": 2}),
            ],
        )
        rule = ToolRetryStormRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.RETRY_FAILURE
        assert res.confidence >= 0.85
        assert res.evidence[0].type == "retry_amplification"

    def test_policy_bypass_rule(self):
        trace = _make_base_trace()
        _add_span(
            trace,
            "policy.check",
            SpanType.POLICY,
            attributes={"policy.is_bypassed": True, "policy.allowed": False},
            events=[TraceEvent(name="policy_bypass")],
        )
        rule = PolicyBypassRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.POLICY_FAILURE
        assert res.severity == IncidentSeverity.P0
        assert res.confidence >= 0.95

    def test_context_explosion_rule(self):
        trace = _make_base_trace()
        _add_span(
            trace,
            "model.gen",
            SpanType.MODEL,
            attributes={"gen_ai.usage.input_tokens": 3500},
            events=[TraceEvent(name="context_bloated")],
        )
        rule = ContextExplosionRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.CONTEXT_FAILURE
        assert res.confidence == 0.85

    def test_evaluation_blind_spot_rule(self):
        trace = _make_base_trace()
        trace.outcome = TraceOutcome.SUCCESS
        _add_span(
            trace,
            "citation.verify",
            SpanType.CITATION,
            attributes={"citation.total_count": 0},
            events=[TraceEvent(name="citation_missing")],
        )
        rule = EvaluationBlindSpotRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.EVALUATION_FAILURE

    def test_timeout_sla_breach_rule(self):
        trace = _make_base_trace()
        _add_span(trace, "tool.slow", SpanType.TOOL, duration_ms=6200.0)
        rule = TimeoutSLABreachRule()
        res = rule.evaluate(trace)
        assert res.matched is True
        assert res.category == FailureCategory.TIMEOUT_FAILURE


class TestRootCauseEngine:
    """Tests for causal graph traversal, earliest suspicious span isolation, and ranking."""

    def test_causal_graph_construction(self, rca):
        trace = _make_base_trace()
        root = _add_span(trace, "agent.run", SpanType.AGENT_RUN)
        child1 = _add_span(trace, "retrieval.search", SpanType.RETRIEVAL, parent_span_id=root.span_id)
        child2 = _add_span(trace, "model.generate", SpanType.MODEL, parent_span_id=root.span_id)

        graph = rca.build_causal_graph(trace)
        assert len(graph) == 3
        assert graph[root.span_id].depth == 0
        assert graph[child1.span_id].depth == 1
        assert graph[child2.span_id].depth == 1
        assert child1.span_id in graph[root.span_id].children_span_ids
        assert child2.span_id in graph[root.span_id].children_span_ids

    def test_earliest_suspicious_span_isolated(self, rca):
        trace = _make_base_trace()
        _add_span(trace, "agent.run", SpanType.AGENT_RUN)
        s_retrieval = _add_span(
            trace, "retrieval.search", SpanType.RETRIEVAL, attributes={"retrieval.candidate_count": 0}
        )
        _add_span(trace, "model.generate", SpanType.MODEL)

        earliest = rca.find_earliest_suspicious_span(trace)
        assert earliest is not None
        assert earliest.span_id == s_retrieval.span_id

    def test_root_cause_precedence_over_downstream(self, rca):
        """Milestone 5 exit criterion: when retrieval fails early, and generation later contradicts,

        retrieval_failure is ranked as the PRIMARY root cause, with generation as downstream symptom.
        """
        trace = _make_base_trace()
        _add_span(trace, "agent.run", SpanType.AGENT_RUN)
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL, attributes={"retrieval.candidate_count": 0})
        _add_span(
            trace, "citation.verify", SpanType.CITATION, attributes={"citation.invalid_count": 1}
        )

        rankings = rca.analyze(trace)
        assert len(rankings) >= 2
        # Primary root cause must be retrieval_failure
        assert rankings[0].is_primary_root_cause is True
        assert rankings[0].category == FailureCategory.RETRIEVAL_FAILURE
        assert len(rankings[0].downstream_symptoms) > 0

    def test_diagnose_to_incidents_structure(self, rca):
        trace = _make_base_trace()
        _add_span(
            trace,
            "policy.check",
            SpanType.POLICY,
            attributes={"policy.is_bypassed": True},
            events=[TraceEvent(name="policy_bypass")],
        )

        incidents = rca.diagnose_to_incidents(trace)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.category == FailureCategory.POLICY_FAILURE
        assert inc.severity == IncidentSeverity.P0
        assert inc.hypothesis_score >= 0.95
        assert "remediation" in inc.metadata
