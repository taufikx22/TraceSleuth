"""TraceSleuth API — Request/Response Schemas.

Pydantic models for API input validation and output serialization.
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


class TimelineResponse(BaseModel):
    """Chronological timeline of a trace."""

    trace_id: str
    entries: List[TimelineEntry]


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


# ── Analysis Schemas ──────────────────────────────────────────


class AnalyzeTraceResponse(BaseModel):
    """Response from running taxonomy analysis on a trace."""

    trace_id: str
    incidents_created: int
    incidents: List[IncidentResponse]
