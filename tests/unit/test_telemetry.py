"""Unit tests for Telemetry Redaction, Exporters, and Domain Models.
"""

from datetime import datetime, timezone
import pytest
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData, TraceOutcome
from src.telemetry.redaction import RedactionEngine, hash_content


def test_redaction_pii_and_credentials():
    engine = RedactionEngine(capture_raw_content=True, redact_pii=True)

    text = "Contact alice@enterprise.com with api_key: secret_123456789 and Bearer abcdef1234567890xyz"
    sanitized = engine.redact_text(text)
    assert "[REDACTED_EMAIL]" in sanitized
    assert "alice@enterprise.com" not in sanitized
    assert "[REDACTED_CREDENTIAL]" in sanitized
    assert "secret_123456789" not in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


def test_redaction_raw_content_hashing():
    engine = RedactionEngine(capture_raw_content=False, redact_pii=True)
    long_prompt = "A" * 120
    sanitized = engine.sanitize_payload(long_prompt)
    assert "hash:" in sanitized
    assert len(sanitized) < 50


def test_hash_content_deterministic():
    data = {"query": "What is ARR?", "top_k": 3}
    h1 = hash_content(data)
    h2 = hash_content(data)
    assert h1 == h2
    assert len(h1) == 16


def test_trace_data_token_aggregation():
    now = datetime.now(timezone.utc)
    s1 = SpanData(
        span_id="span_1",
        trace_id="trace_1",
        name="model.call_1",
        span_type=SpanType.MODEL,
        start_time=now,
        end_time=now,
        attributes={"gen_ai.usage.input_tokens": 120, "gen_ai.usage.output_tokens": 40},
    )
    s2 = SpanData(
        span_id="span_2",
        trace_id="trace_1",
        name="model.call_2",
        span_type=SpanType.MODEL,
        start_time=now,
        end_time=now,
        attributes={"gen_ai.usage.input_tokens": 80, "gen_ai.usage.output_tokens": 20},
    )
    s3 = SpanData(
        span_id="span_3",
        trace_id="trace_1",
        name="retrieval.step",
        span_type=SpanType.RETRIEVAL,
        start_time=now,
        end_time=now,
    )
    trace = TraceData(
        trace_id="trace_1",
        request_id="req_1",
        start_time=now,
        end_time=now,
        spans=[s1, s2, s3],
    )
    assert trace.total_tokens() == (120 + 40 + 80 + 20)
    assert len(trace.get_spans_by_type(SpanType.MODEL)) == 2
    assert trace.get_span("span_3") is not None
