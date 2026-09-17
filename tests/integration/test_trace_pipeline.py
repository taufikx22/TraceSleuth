"""Integration tests for end-to-end OpenTelemetry Trace Pipeline (Milestone 1).

Verifies that a complete request produces a full causal trace with:
- Root agent span and nested child spans for all pipeline operations
- Parent-child relationship correlation
- Standard GenAI semantic convention attributes
- Recorded events on failure modes
"""

import pytest
from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.domain.trace import SpanStatus, SpanType


@pytest.fixture
def agent() -> InstrumentedResearchAgent:
    return InstrumentedResearchAgent()


def test_milestone_1_exit_full_trace_for_single_request(agent):
    """Exit criteria test: Full trace appears for one request."""
    req = AgentRequest(
        query="What was Enterprise Corp Q3 2025 cloud ARR and gross margin?",
        failure_mode=FailureMode.HEALTHY,
    )
    response, trace = agent.run(req)

    # 1. Verify response
    assert response.status == "success"
    assert "428M" in response.answer or "gross margin" in response.answer

    # 2. Verify trace metadata
    assert trace is not None
    assert trace.trace_id != ""
    assert len(trace.trace_id) == 32  # 32 hex chars standard OTel trace_id
    assert trace.request_id.startswith("req_")
    assert trace.duration_ms > 0
    assert trace.outcome.value == "success"

    # 3. Verify span count and presence of all major stages
    assert len(trace.spans) >= 8
    span_types = {s.span_type for s in trace.spans}
    assert SpanType.AGENT_RUN in span_types
    assert SpanType.CLASSIFIER in span_types
    assert SpanType.POLICY in span_types
    assert SpanType.RETRIEVAL in span_types
    assert SpanType.RERANKER in span_types
    assert SpanType.PLANNER in span_types
    assert SpanType.MODEL in span_types
    assert SpanType.CITATION in span_types

    # 4. Verify Causal Hierarchy: child spans point to root span
    root_span = trace.get_span(trace.root_span_id)
    assert root_span is not None
    assert root_span.parent_span_id is None

    child_spans = [s for s in trace.spans if s.span_id != root_span.span_id]
    for child in child_spans:
        assert child.trace_id == trace.trace_id
        assert child.parent_span_id == root_span.span_id

    # 5. Verify OpenTelemetry GenAI Semantic Conventions on model span
    model_spans = trace.get_spans_by_type(SpanType.MODEL)
    assert len(model_spans) == 1
    m_span = model_spans[0]
    assert m_span.attributes.get("gen_ai.system") == "synthetic_llm"
    assert m_span.attributes.get("gen_ai.request.model") == "enterprise-research-v1"
    assert int(m_span.attributes.get("gen_ai.usage.input_tokens", 0)) > 0
    assert int(m_span.attributes.get("gen_ai.usage.output_tokens", 0)) > 0


def test_trace_captures_tool_retries(agent):
    """Verifies that tool failures and retries record events and retry spans."""
    req = AgentRequest(
        query="Calculate the net margin from 428 and 184",
        failure_mode=FailureMode.TOOL_RETRY_STORM,
        max_retries=3,
    )
    _response, trace = agent.run(req)
    assert trace is not None

    tool_spans = trace.get_spans_by_type(SpanType.TOOL)
    calc_spans = [s for s in tool_spans if s.name == "tool.calculator"]
    assert len(calc_spans) >= 3

    # Check retry events recorded in spans
    retry_events = [
        evt for s in calc_spans for evt in s.events if evt.name == "retry"
    ]
    assert len(retry_events) >= 3


def test_trace_captures_retrieval_empty_event(agent):
    """Verifies retrieval miss records event on retrieval span."""
    req = AgentRequest(
        query="What was the Q3 financial roadmap?",
        failure_mode=FailureMode.RETRIEVAL_MISS,
    )
    _response, trace = agent.run(req)
    assert trace is not None

    retrieval_spans = trace.get_spans_by_type(SpanType.RETRIEVAL)
    assert len(retrieval_spans) == 1
    events = retrieval_spans[0].events
    assert any(e.name == "retrieval_empty" for e in events)


def test_trace_captures_policy_bypass_event(agent):
    """Verifies policy bypass records event and continues execution."""
    req = AgentRequest(
        query="Perform unauthorized action",
        failure_mode=FailureMode.POLICY_BYPASS,
    )
    _response, trace = agent.run(req)
    assert trace is not None

    policy_spans = trace.get_spans_by_type(SpanType.POLICY)
    assert len(policy_spans) == 1
    events = policy_spans[0].events
    assert any(e.name == "policy_bypass" for e in events)
