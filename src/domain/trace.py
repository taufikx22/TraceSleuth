"""TraceSleuth Domain Models.

Defines the core schema for Traces, Spans, Events, and Telemetry metadata
aligned with OpenTelemetry GenAI Semantic Conventions and TraceSleuth Forensics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SpanType(str, Enum):
    """Categorization of agentic spans in AI pipelines."""
    MODEL = "model"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    TOOL = "tool"
    POLICY = "policy"
    PLANNER = "planner"
    CLASSIFIER = "classifier"
    CITATION = "citation"
    ROUTER = "router"
    AGENT_RUN = "agent_run"
    INTERNAL = "internal"


class SpanStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"


class TraceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"


class TraceEvent(BaseModel):
    """A notable occurrence inside a span (e.g. retry, policy block, missing citation)."""
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SpanData(BaseModel):
    """Structured representation of an executed operation within a trace."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    span_type: SpanType
    start_time: datetime
    end_time: datetime
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.OK
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[TraceEvent] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.duration_ms == 0.0 and self.end_time and self.start_time:
            diff = (self.end_time - self.start_time).total_seconds() * 1000.0
            self.duration_ms = max(0.0, round(diff, 3))


class TraceData(BaseModel):
    """An end-to-end request execution trace containing all causal spans and metadata."""
    trace_id: str
    request_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    application_version: str = "0.1.0"
    model_configuration_version: str = "v1.0.0"
    prompt_version: str = "v1.0.0"
    environment: str = "development"
    start_time: datetime
    end_time: datetime
    duration_ms: float = 0.0
    outcome: TraceOutcome = TraceOutcome.SUCCESS
    status: str = "COMPLETED"
    root_span_id: Optional[str] = None
    spans: List[SpanData] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.duration_ms == 0.0 and self.end_time and self.start_time:
            diff = (self.end_time - self.start_time).total_seconds() * 1000.0
            self.duration_ms = max(0.0, round(diff, 3))

    def get_span(self, span_id: str) -> Optional[SpanData]:
        for s in self.spans:
            if s.span_id == span_id:
                return s
        return None

    def get_span_by_id(self, span_id: str) -> Optional[SpanData]:
        return self.get_span(span_id)

    def get_spans_by_type(self, span_type: SpanType) -> List[SpanData]:
        return [s for s in self.spans if s.span_type == span_type]

    def total_tokens(self) -> int:
        total = 0
        for s in self.spans:
            if s.span_type == SpanType.MODEL:
                input_tok = s.attributes.get("gen_ai.usage.input_tokens", 0)
                output_tok = s.attributes.get("gen_ai.usage.output_tokens", 0)
                total += int(input_tok) + int(output_tok)
        return total
