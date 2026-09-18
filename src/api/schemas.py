"""TraceSleuth API — Request/Response Schemas.

Pydantic models for API input validation and output serialization.
Includes schemas for traces, timelines, incidents, evidence graphs,
similar incident search, and replay executions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Trace Schemas ─────────────────────────────────────────────


class SpanEventResponse(BaseModel):
    """API representation of a span event."""

    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SpanResponse(BaseModel):
    """API representation of a single span."""

    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    span_type: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[SpanEventResponse] = Field(default_factory=list)


class TraceResponse(BaseModel):
    """Full trace detail with all spans."""

    trace_id: str
    request_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    application_version: str
    model_configuration_version: str
    prompt_version: str
    environment: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    outcome: str
    status: str
    root_span_id: Optional[str] = None
    spans: List[SpanResponse] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    span_count: int = 0


class TraceListItem(BaseModel):
    """Compact trace summary for list views."""

    trace_id: str
    request_id: str
    environment: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    outcome: str
    status: str
    span_count: int = 0


class TraceListResponse(BaseModel):
    """Paginated list of traces."""

    traces: List[TraceListItem]
    total: int
    limit: int
    offset: int


class TimelineEntry(BaseModel):
    """A single entry in the chronological timeline view."""

    timestamp: datetime
    span_id: str
    span_name: str
    span_type: str
    event_type: str  # "span_start", "span_end", "event"
    event_name: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    is_first_suspicious: bool = False


class TimelineResponse(BaseModel):
    """Chronological timeline of a trace."""

    trace_id: str
    entries: List[TimelineEntry]
    first_suspicious_span_id: Optional[str] = None


# ── Incident Schemas ──────────────────────────────────────────


class EvidenceResponse(BaseModel):
    """API representation of evidence."""

    evidence_id: str
    span_id: Optional[str] = None
    type: str
    key: str
    value_hash: str
    description: str
    strength: float


class IncidentResponse(BaseModel):
    """Full incident detail."""

    incident_id: str
    trace_id: str
    category: str
    severity: str
    summary: str
    first_suspicious_span_id: Optional[str] = None
    hypothesis_score: float
    evidence: List[EvidenceResponse] = Field(default_factory=list)
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentListItem(BaseModel):
    """Compact incident summary for list views."""

    incident_id: str
    trace_id: str
    category: str
    severity: str
    summary: str
    hypothesis_score: float
    status: str
    created_at: datetime


class IncidentListResponse(BaseModel):
    """Paginated list of incidents."""

    incidents: List[IncidentListItem]
    total: int
    limit: int
    offset: int


class IncidentStatusUpdate(BaseModel):
    """Request body for updating an incident's status."""

    status: str  # "open", "investigating", "confirmed", "resolved", "false_positive"


class SimilarIncidentItem(BaseModel):
    """An incident that shares a failure signature with similarity score."""

    incident_id: str
    trace_id: str
    category: str
    severity: str
    summary: str
    similarity_score: float
    created_at: datetime


class SimilarIncidentsResponse(BaseModel):
    """Response containing similar historical incidents."""

    target_incident_id: str
    target_category: str
    similar_incidents: List[SimilarIncidentItem]
    total: int


# ── Evidence Graph Schemas (Milestone 6) ──────────────────────


class GraphNode(BaseModel):
    """Node in an incident causal evidence graph."""

    id: str
    label: str
    type: str  # "span", "evidence", "hypothesis", "anomaly"
    is_suspicious: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Directed edge in an incident causal evidence graph."""

    source: str
    target: str
    label: str
    strength: Optional[float] = None


class EvidenceGraphResponse(BaseModel):
    """Serialized causal evidence graph for rendering."""

    incident_id: str
    trace_id: str
    first_suspicious_span_id: Optional[str] = None
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ── Replay Schemas (Milestone 7) ──────────────────────────────


class ReplayRequest(BaseModel):
    """Request payload to initiate a replay."""

    source_trace_id: str
    replay_type: str = "controlled"  # "exact", "controlled", "counterfactual"
    retriever_mode: Optional[str] = None  # "fixed", "default", "dense_expanded"
    vector_threshold: Optional[float] = None
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    bypass_policy: Optional[bool] = None
    tool_failure_rate: Optional[float] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class ReplayMetricsDiffResponse(BaseModel):
    """Metrics difference between source and replayed traces."""

    original_duration_ms: float
    replayed_duration_ms: float
    duration_diff_ms: float
    original_tokens: int
    replayed_tokens: int
    tokens_diff: int
    original_span_count: int
    replayed_span_count: int
    original_outcome: str
    replayed_outcome: str
    resolved_incident_categories: List[str]
    new_incident_categories: List[str]


class ReplayResponse(BaseModel):
    """Replay execution result."""

    replay_id: str
    source_trace_id: str
    replayed_trace_id: Optional[str] = None
    replay_type: str
    verdict: str  # "fixed", "still_failing", "regressed", "identical", "inconclusive"
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float
    metrics_diff: Optional[ReplayMetricsDiffResponse] = None
    summary: str


class ReplayListResponse(BaseModel):
    """List of replay executions."""

    replays: List[ReplayResponse]
    total: int


# ── Analysis Schemas ──────────────────────────────────────────


class AnalyzeTraceResponse(BaseModel):
    """Response from running taxonomy analysis on a trace."""

    trace_id: str
    incidents_created: int
    incidents: List[IncidentResponse]


# ── Regression Schemas (Milestone 8) ─────────────────────────


class CreateRegressionCaseRequest(BaseModel):
    """Request payload to convert an incident into a regression test case."""

    owner: str = "ai-platform"
    introduced_version: Optional[str] = None


class RegressionAssertionResponse(BaseModel):
    """API model for a regression assertion."""

    assertion_id: str
    name: str
    target_metric: str
    operator: str
    expected_value: Any
    description: str = ""


class RegressionCaseResponse(BaseModel):
    """API model for a regression case."""

    id: str
    incident_id: str
    source_trace_id: str
    failure_type: str
    query: str
    input_hash: str
    expected_evidence: List[str] = Field(default_factory=list)
    assertions: List[RegressionAssertionResponse] = Field(default_factory=list)
    owner: str
    introduced_version: str
    fixed_version: Optional[str] = None
    status: str
    created_at: datetime


class AssertionResultResponse(BaseModel):
    """API model for a single assertion check."""

    assertion_id: str
    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""


class RegressionRunResponse(BaseModel):
    """API model for a regression run outcome."""

    case_id: str
    passed: bool
    replayed_trace_id: Optional[str] = None
    assertion_results: List[AssertionResultResponse] = Field(default_factory=list)
    duration_ms: float
    summary: str


# ── Metrics & Alerts Schemas (Milestone 9) ───────────────────


class LatencyBreakdownResponse(BaseModel):
    """Decomposed latency breakdown by execution stage."""

    trace_id: str
    total_latency_ms: float
    router_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    planner_ms: float = 0.0
    model_ms: float = 0.0
    tool_ms: float = 0.0
    citation_ms: float = 0.0
    other_ms: float = 0.0


class SLAMetricsResponse(BaseModel):
    """Service Level Objective metrics for availability and quality."""

    total_traces: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    availability_rate: float
    quality_pass_rate: float
    error_rate: float
    tool_failure_rate: float
    total_tokens: int
    estimated_cost_usd: float


class CostBreakdownResponse(BaseModel):
    """Token usage and estimated financial cost summary."""

    total_cost_usd: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cost_per_request_usd: float
    model_pricing_version: str


class AlertResponse(BaseModel):
    """Active or historic alert item."""

    alert_id: str
    name: str
    severity: str  # "P0", "P1", "P2"
    status: str  # "firing", "resolved"
    triggered_at: datetime
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ── Model Hypothesis Schemas (Milestone 10) ──────────────────


class EvidenceCitationResponse(BaseModel):
    """Traceable citation linking a model hypothesis claim to specific span telemetry."""

    span_id: str
    evidence_key: str
    description: str
    strength: float


class ModelHypothesisResponse(BaseModel):
    """Grounded, model-assisted root-cause hypothesis with evidence citations."""

    trace_id: str
    incident_id: Optional[str] = None
    primary_root_cause: str
    confidence: float
    explanation: str
    evidence_citations: List[EvidenceCitationResponse] = Field(default_factory=list)
    recommended_remediation: str
    engine: str
    generated_at: datetime
