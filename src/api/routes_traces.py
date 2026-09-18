"""TraceSleuth API — Trace Explorer Routes.

Endpoints for listing, searching, inspecting, and analyzing traces.
Implements the trace explorer described in spec Section 28.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import (
    AnalyzeTraceResponse,
    EvidenceResponse,
    IncidentResponse,
    SpanEventResponse,
    SpanResponse,
    TimelineEntry,
    TimelineResponse,
    TraceListItem,
    TraceListResponse,
    TraceResponse,
)
from src.domain.trace import TraceData
from src.storage.database import get_database
from src.storage.repository import TraceRepository
from src.ingestion.service import IngestionService

router = APIRouter(prefix="/traces", tags=["traces"])


def _get_repo() -> TraceRepository:
    db = get_database()
    return TraceRepository(db.get_session())


def _trace_to_response(trace: TraceData) -> TraceResponse:
    """Convert a TraceData to an API TraceResponse."""
    spans = [
        SpanResponse(
            span_id=s.span_id,
            trace_id=s.trace_id,
            parent_span_id=s.parent_span_id,
            name=s.name,
            span_type=s.span_type.value,
            start_time=s.start_time,
            end_time=s.end_time,
            duration_ms=s.duration_ms,
            status=s.status.value,
            error_message=s.error_message,
            attributes=s.attributes,
            events=[
                SpanEventResponse(
                    name=e.name,
                    timestamp=e.timestamp,
                    attributes=e.attributes,
                )
                for e in s.events
            ],
        )
        for s in trace.spans
    ]

    return TraceResponse(
        trace_id=trace.trace_id,
        request_id=trace.request_id,
        session_id=trace.session_id,
        tenant_id=trace.tenant_id,
        application_version=trace.application_version,
        model_configuration_version=trace.model_configuration_version,
        prompt_version=trace.prompt_version,
        environment=trace.environment,
        start_time=trace.start_time,
        end_time=trace.end_time,
        duration_ms=trace.duration_ms,
        outcome=trace.outcome.value,
        status=trace.status,
        root_span_id=trace.root_span_id,
        spans=spans,
        metadata=trace.metadata,
        span_count=len(trace.spans),
    )


@router.get("", response_model=TraceListResponse)
def list_traces(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    outcome: Optional[str] = Query(default=None),
    environment: Optional[str] = Query(default=None),
    start_after: Optional[datetime] = Query(default=None),
    start_before: Optional[datetime] = Query(default=None),
) -> TraceListResponse:
    """List traces with optional filters (spec Section 28: Trace explorer)."""
    repo = _get_repo()
    traces, total = repo.list_traces(
        limit=limit,
        offset=offset,
        outcome=outcome,
        environment=environment,
        start_after=start_after,
        start_before=start_before,
    )

    items = [
        TraceListItem(
            trace_id=t.trace_id,
            request_id=t.request_id,
            environment=t.environment,
            start_time=t.start_time,
            end_time=t.end_time,
            duration_ms=t.duration_ms,
            outcome=t.outcome.value,
            status=t.status,
            span_count=len(t.spans),
        )
        for t in traces
    ]

    return TraceListResponse(traces=items, total=total, limit=limit, offset=offset)


@router.get("/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str) -> TraceResponse:
    """Get a complete trace with all spans and events (spec Section 29)."""
    repo = _get_repo()
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return _trace_to_response(trace)


@router.get("/{trace_id}/timeline", response_model=TimelineResponse)
def get_trace_timeline(trace_id: str) -> TimelineResponse:
    """Get a chronological timeline of a trace's execution (spec Section 14).

    Returns all span starts, ends, and events in chronological order,
    enabling the engineer to reconstruct the execution sequence.
    """
    repo = _get_repo()
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    entries: list[TimelineEntry] = []

    for span in trace.spans:
        # Span start entry
        entries.append(
            TimelineEntry(
                timestamp=span.start_time,
                span_id=span.span_id,
                span_name=span.name,
                span_type=span.span_type.value,
                event_type="span_start",
                details={"parent_span_id": span.parent_span_id},
            )
        )

        # Inline events
        for event in span.events:
            entries.append(
                TimelineEntry(
                    timestamp=event.timestamp,
                    span_id=span.span_id,
                    span_name=span.name,
                    span_type=span.span_type.value,
                    event_type="event",
                    event_name=event.name,
                    details=event.attributes,
                )
            )

        # Span end entry
        entries.append(
            TimelineEntry(
                timestamp=span.end_time,
                span_id=span.span_id,
                span_name=span.name,
                span_type=span.span_type.value,
                event_type="span_end",
                duration_ms=span.duration_ms,
                status=span.status.value,
            )
        )

    # Sort chronologically
    def _ts_key(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    entries.sort(key=lambda e: _ts_key(e.timestamp))

    return TimelineResponse(trace_id=trace_id, entries=entries)


@router.post("/{trace_id}/analyze", response_model=AnalyzeTraceResponse)
def analyze_trace(trace_id: str) -> AnalyzeTraceResponse:
    """Run the failure taxonomy classifier on a stored trace (spec Section 29).

    Creates incidents with evidence for any detected failures.
    """
    db = get_database()
    service = IngestionService(db)
    incidents = service.classify_trace(trace_id)

    if incidents is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    incident_responses = [
        IncidentResponse(
            incident_id=inc.incident_id,
            trace_id=inc.trace_id,
            category=inc.category.value,
            severity=inc.severity.value,
            summary=inc.summary,
            first_suspicious_span_id=inc.first_suspicious_span_id,
            hypothesis_score=inc.hypothesis_score,
            evidence=[
                EvidenceResponse(
                    evidence_id=ev.evidence_id,
                    span_id=ev.span_id,
                    type=ev.type,
                    key=ev.key,
                    value_hash=ev.value_hash,
                    description=ev.description,
                    strength=ev.strength,
                )
                for ev in inc.evidence
            ],
            status=inc.status.value,
            created_at=inc.created_at,
            resolved_at=inc.resolved_at,
            metadata=inc.metadata,
        )
        for inc in incidents
    ]

    return AnalyzeTraceResponse(
        trace_id=trace_id,
        incidents_created=len(incidents),
        incidents=incident_responses,
    )
