"""Tests for TraceSleuth Failure Taxonomy Classifier (Milestone 4).

Verifies that all 13 failure categories are correctly detected from
crafted trace data, and that healthy traces produce no incidents.
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from src.analysis.taxonomy import FailureTaxonomyClassifier
from src.domain.incident import FailureCategory
from src.domain.trace import (
    SpanData,
    SpanStatus,
    SpanType,
    TraceData,
    TraceEvent,
    TraceOutcome,
)


@pytest.fixture
def classifier():
    return FailureTaxonomyClassifier()


def _make_base_trace(trace_id: str = None, outcome: TraceOutcome = TraceOutcome.SUCCESS) -> TraceData:
    """Create a minimal trace shell."""
    t_id = trace_id or uuid.uuid4().hex[:32]
    now = datetime.now(timezone.utc)
    return TraceData(
        trace_id=t_id,
        request_id=f"req_{t_id[:8]}",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        outcome=outcome,
        spans=[],
    )


def _add_span(
    trace: TraceData,
    name: str,
    span_type: SpanType,
    status: SpanStatus = SpanStatus.OK,
    error_message: str = None,
    attributes: dict = None,
    events: list = None,
    duration_ms: float = 10.0,
) -> SpanData:
    """Add a span to a trace and return it."""
    now = datetime.now(timezone.utc)
    span = SpanData(
        span_id=f"span_{len(trace.spans):04d}",
        trace_id=trace.trace_id,
        name=name,
        span_type=span_type,
        start_time=now,
        end_time=now + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        status=status,
        error_message=error_message,
        attributes=attributes or {},
        events=events or [],
    )
    trace.spans.append(span)
    return span


class TestHealthyTrace:
    """A clean trace should produce no incidents."""

    def test_no_incidents_for_healthy_trace(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "agent.run", SpanType.AGENT_RUN)
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   attributes={"retrieval.candidate_count": 5})
        _add_span(trace, "reranker.filter", SpanType.RERANKER,
                   attributes={"reranker.input_count": 5, "reranker.output_count": 3})
        _add_span(trace, "model.generate", SpanType.MODEL,
                   attributes={"gen_ai.usage.input_tokens": 500, "gen_ai.usage.output_tokens": 200})
        _add_span(trace, "citation.verify", SpanType.CITATION,
                   attributes={"citation.total_count": 3, "citation.valid_count": 3, "citation.invalid_count": 0})

        incidents = classifier.classify(trace)
        assert len(incidents) == 0


class TestRetrievalFailure:
    """Detect when retrieval returns 0 candidates."""

    def test_empty_retrieval(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   attributes={"retrieval.candidate_count": 0},
                   events=[TraceEvent(name="retrieval_empty", attributes={"query": "test"})])

        incidents = classifier.classify(trace)
        retrieval = [i for i in incidents if i.category == FailureCategory.RETRIEVAL_FAILURE]
        assert len(retrieval) == 1
        assert retrieval[0].hypothesis_score >= 0.8
        assert len(retrieval[0].evidence) >= 1


class TestRankingFailure:
    """Detect aggressive filtering by reranker."""

    def test_reranker_eliminates_all(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "reranker.filter", SpanType.RERANKER,
                   attributes={"reranker.input_count": 5, "reranker.output_count": 0})

        incidents = classifier.classify(trace)
        ranking = [i for i in incidents if i.category == FailureCategory.RANKING_FAILURE]
        assert len(ranking) == 1
        assert ranking[0].hypothesis_score >= 0.8


class TestContextFailure:
    """Detect context bloat or truncation."""

    def test_context_bloated_event(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   events=[TraceEvent(name="context_bloated", attributes={"chunk_count": 50})])

        incidents = classifier.classify(trace)
        context = [i for i in incidents if i.category == FailureCategory.CONTEXT_FAILURE]
        assert len(context) == 1
        assert context[0].hypothesis_score >= 0.7


class TestGenerationFailure:
    """Detect model contradicting evidence."""

    def test_invalid_citations(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "citation.verify", SpanType.CITATION,
                   attributes={"citation.total_count": 3, "citation.valid_count": 1, "citation.invalid_count": 2},
                   events=[TraceEvent(name="citation_invalid", attributes={"claim": "wrong"})])

        incidents = classifier.classify(trace)
        gen = [i for i in incidents if i.category == FailureCategory.GENERATION_FAILURE]
        assert len(gen) == 1
        assert gen[0].hypothesis_score >= 0.7

    def test_model_error(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "model.generate", SpanType.MODEL,
                   status=SpanStatus.ERROR, error_message="Model inference failed")

        incidents = classifier.classify(trace)
        gen = [i for i in incidents if i.category == FailureCategory.GENERATION_FAILURE]
        assert len(gen) == 1
        assert gen[0].hypothesis_score >= 0.8


class TestToolSelectionFailure:
    """Detect missing tool execution when planned."""

    def test_planned_tools_not_executed(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "planner.plan", SpanType.PLANNER,
                   attributes={"planner.step_count": 3, "planner.suggested_tools": "['calc']"})
        # No tool spans added → tool selection failure

        incidents = classifier.classify(trace)
        tool_sel = [i for i in incidents if i.category == FailureCategory.TOOL_SELECTION_FAILURE]
        assert len(tool_sel) == 1


class TestToolArgumentFailure:
    """Detect tool called with invalid arguments."""

    def test_argument_error(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "tool.calculator", SpanType.TOOL,
                   status=SpanStatus.ERROR,
                   error_message="Invalid argument type: expected int")

        incidents = classifier.classify(trace)
        tool_arg = [i for i in incidents if i.category == FailureCategory.TOOL_ARGUMENT_FAILURE]
        assert len(tool_arg) == 1


class TestPolicyFailure:
    """Detect policy bypass — the most critical failure type."""

    def test_policy_bypass(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "policy.evaluate", SpanType.POLICY,
                   attributes={"policy.decision": "deny", "policy.is_bypassed": True},
                   events=[TraceEvent(name="policy_bypass", attributes={})])

        incidents = classifier.classify(trace)
        policy = [i for i in incidents if i.category == FailureCategory.POLICY_FAILURE]
        assert len(policy) == 1
        assert policy[0].severity.value == "P0"  # Critical
        assert policy[0].hypothesis_score >= 0.9

    def test_policy_denial_without_bypass(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "policy.evaluate", SpanType.POLICY,
                   attributes={"policy.decision": "deny", "policy.is_bypassed": False})

        incidents = classifier.classify(trace)
        policy = [i for i in incidents if i.category == FailureCategory.POLICY_FAILURE]
        assert len(policy) == 1
        assert policy[0].severity.value == "P1"  # Not critical since not bypassed


class TestStateFailure:
    """Detect session/memory state corruption."""

    def test_state_corruption_event(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "agent.run", SpanType.AGENT_RUN,
                   events=[TraceEvent(name="state_corruption", attributes={"field": "memory"})])

        incidents = classifier.classify(trace)
        state = [i for i in incidents if i.category == FailureCategory.STATE_FAILURE]
        assert len(state) == 1


class TestRoutingFailure:
    """Detect wrong model/provider routing."""

    def test_router_error(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "router.select", SpanType.ROUTER,
                   status=SpanStatus.ERROR, error_message="Provider unavailable")

        incidents = classifier.classify(trace)
        routing = [i for i in incidents if i.category == FailureCategory.ROUTING_FAILURE]
        assert len(routing) == 1


class TestRetryFailure:
    """Detect retry amplification storms."""

    def test_retry_storm(self, classifier):
        trace = _make_base_trace()
        for i in range(4):
            _add_span(trace, "tool.calculator", SpanType.TOOL,
                       attributes={"tool.retry_count": i},
                       events=[TraceEvent(name="retry")] if i > 0 else [])

        incidents = classifier.classify(trace)
        retry = [i for i in incidents if i.category == FailureCategory.RETRY_FAILURE]
        assert len(retry) == 1
        assert retry[0].hypothesis_score >= 0.7


class TestTimeoutFailure:
    """Detect timeout-related failures."""

    def test_tool_timeout(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "tool.api", SpanType.TOOL,
                   attributes={"tool.result_status": "timeout"}, duration_ms=6000)

        incidents = classifier.classify(trace)
        timeout = [i for i in incidents if i.category == FailureCategory.TIMEOUT_FAILURE]
        assert len(timeout) == 1
        assert timeout[0].hypothesis_score >= 0.8


class TestDataQualityFailure:
    """Detect upstream data quality issues."""

    def test_stale_data_event(self, classifier):
        trace = _make_base_trace()
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   events=[TraceEvent(name="stale_data", attributes={"age_days": 90})])

        incidents = classifier.classify(trace)
        dq = [i for i in incidents if i.category == FailureCategory.DATA_QUALITY_FAILURE]
        assert len(dq) == 1


class TestEvaluationFailure:
    """Detect outputs incorrectly marked as acceptable."""

    def test_success_with_invalid_citations(self, classifier):
        """Trace marked success but citations reveal problems."""
        trace = _make_base_trace(outcome=TraceOutcome.SUCCESS)
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   attributes={"retrieval.candidate_count": 5})
        _add_span(trace, "citation.verify", SpanType.CITATION,
                   attributes={"citation.total_count": 3, "citation.valid_count": 1, "citation.invalid_count": 2})

        incidents = classifier.classify(trace)
        eval_fail = [i for i in incidents if i.category == FailureCategory.EVALUATION_FAILURE]
        assert len(eval_fail) == 1
        assert eval_fail[0].hypothesis_score >= 0.6


class TestIncidentRanking:
    """Verify that incidents are ranked by hypothesis score."""

    def test_incidents_sorted_by_score(self, classifier):
        trace = _make_base_trace(outcome=TraceOutcome.SUCCESS)
        # Retrieval failure (high score ~0.9)
        _add_span(trace, "retrieval.search", SpanType.RETRIEVAL,
                   attributes={"retrieval.candidate_count": 0},
                   events=[TraceEvent(name="retrieval_empty")])
        # Citation issues (medium score)
        _add_span(trace, "citation.verify", SpanType.CITATION,
                   attributes={"citation.total_count": 2, "citation.valid_count": 0, "citation.invalid_count": 2},
                   events=[TraceEvent(name="citation_missing")])

        incidents = classifier.classify(trace)
        assert len(incidents) >= 2
        # Verify sorted descending by score
        for i in range(len(incidents) - 1):
            assert incidents[i].hypothesis_score >= incidents[i + 1].hypothesis_score
