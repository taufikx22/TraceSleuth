"""TraceSleuth API — Incident Routes.

Endpoints for listing, inspecting, updating incident status,
viewing incident timeline, similar incident matching, and evidence graphs.
Implements the Incident View requirements from spec Section 14, 28, and 29.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    EvidenceGraphResponse,
    EvidenceResponse,
    GraphEdge,
    GraphNode,
    IncidentListItem,
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusUpdate,
    SimilarIncidentItem,
    SimilarIncidentsResponse,
    TimelineEntry,
    TimelineResponse,
)
from src.storage.database import get_database
from src.storage.repository import TraceRepository

router = APIRouter(prefix="/incidents", tags=["incidents"])


def _get_repo() -> TraceRepository:
    db = get_database()
    return TraceRepository(db.get_session())


@router.get("", response_model=IncidentListResponse)
def list_incidents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    trace_id: Optional[str] = Query(default=None),
) -> IncidentListResponse:
    """List incidents with optional filters."""
    repo = _get_repo()
    incidents, total = repo.list_incidents(
        limit=limit,
        offset=offset,
        category=category,
        severity=severity,
        status=status,
        trace_id=trace_id,
    )

    items = [
        IncidentListItem(
            incident_id=inc.incident_id,
            trace_id=inc.trace_id,
            category=inc.category.value,
            severity=inc.severity.value,
            summary=inc.summary,
            hypothesis_score=inc.hypothesis_score,
            status=inc.status.value,
            created_at=inc.created_at,
        )
        for inc in incidents
    ]

    return IncidentListResponse(
        incidents=items, total=total, limit=limit, offset=offset
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str) -> IncidentResponse:
    """Get full incident detail with evidence and linked spans."""
    repo = _get_repo()
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(
            status_code=404, detail=f"Incident {incident_id} not found"
        )

    return IncidentResponse(
        incident_id=incident.incident_id,
        trace_id=incident.trace_id,
        category=incident.category.value,
        severity=incident.severity.value,
        summary=incident.summary,
        first_suspicious_span_id=incident.first_suspicious_span_id,
        hypothesis_score=incident.hypothesis_score,
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
            for ev in incident.evidence
        ],
        status=incident.status.value,
        created_at=incident.created_at,
        resolved_at=incident.resolved_at,
        metadata=incident.metadata,
    )


@router.patch("/{incident_id}", response_model=IncidentResponse)
def update_incident_status(
    incident_id: str, body: IncidentStatusUpdate
) -> IncidentResponse:
    """Update an incident's status (e.g., confirm, resolve, mark false positive)."""
    valid_statuses = {"open", "investigating", "confirmed", "resolved", "false_positive"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{body.status}'. Must be one of: {valid_statuses}",
        )

    repo = _get_repo()
    incident = repo.update_incident_status(incident_id, body.status)
    if not incident:
        raise HTTPException(
            status_code=404, detail=f"Incident {incident_id} not found"
        )

    return IncidentResponse(
        incident_id=incident.incident_id,
        trace_id=incident.trace_id,
        category=incident.category.value,
        severity=incident.severity.value,
        summary=incident.summary,
        first_suspicious_span_id=incident.first_suspicious_span_id,
        hypothesis_score=incident.hypothesis_score,
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
            for ev in incident.evidence
        ],
        status=incident.status.value,
        created_at=incident.created_at,
        resolved_at=incident.resolved_at,
        metadata=incident.metadata,
    )


@router.get("/{incident_id}/timeline", response_model=TimelineResponse)
def get_incident_timeline(incident_id: str) -> TimelineResponse:
    """Incident Timeline (spec Section 14).

    Returns a chronological sequence of span starts, span ends, and events
    for the trace, with the first suspicious event explicitly flagged.
    """
    repo = _get_repo()
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    trace = repo.get_trace(incident.trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Associated trace {incident.trace_id} not found")

    entries: List[TimelineEntry] = []
    suspicious_span_id = incident.first_suspicious_span_id

    for span in trace.spans:
        is_susp = (span.span_id == suspicious_span_id)
        # Span start entry
        entries.append(
            TimelineEntry(
                timestamp=span.start_time,
                span_id=span.span_id,
                span_name=span.name,
                span_type=span.span_type.value,
                event_type="span_start",
                details={"parent_span_id": span.parent_span_id},
                is_first_suspicious=is_susp,
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
                    is_first_suspicious=is_susp,
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
                is_first_suspicious=is_susp,
            )
        )

    def _ts_key(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    entries.sort(key=lambda e: _ts_key(e.timestamp))

    return TimelineResponse(
        trace_id=trace.trace_id,
        entries=entries,
        first_suspicious_span_id=suspicious_span_id,
    )


@router.get("/{incident_id}/similar", response_model=SimilarIncidentsResponse)
def get_similar_incidents(
    incident_id: str, limit: int = Query(default=5, ge=1, le=20)
) -> SimilarIncidentsResponse:
    """Similar Incidents Search (spec Section 11 & 31: 'Have we seen this failure before?')."""
    repo = _get_repo()
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    similar = repo.find_similar_incidents(incident_id, limit=limit)
    items = [
        SimilarIncidentItem(
            incident_id=other.incident_id,
            trace_id=other.trace_id,
            category=other.category.value,
            severity=other.severity.value,
            summary=other.summary,
            similarity_score=score,
            created_at=other.created_at,
        )
        for other, score in similar
    ]

    return SimilarIncidentsResponse(
        target_incident_id=incident.incident_id,
        target_category=incident.category.value,
        similar_incidents=items,
        total=len(items),
    )


@router.get("/{incident_id}/evidence-graph", response_model=EvidenceGraphResponse)
def get_incident_evidence_graph(incident_id: str) -> EvidenceGraphResponse:
    """Evidence Graph (spec Section 10.2).

    Constructs graph nodes and edges showing how evidence links to the
    suspicious span and the overall failure hypothesis.
    """
    repo = _get_repo()
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    # 1. Root hypothesis node
    hypo_id = f"hypo_{incident.category.value}"
    nodes.append(
        GraphNode(
            id=hypo_id,
            label=f"Hypothesis: {incident.category.value} ({incident.hypothesis_score:.2f})",
            type="hypothesis",
            attributes={"severity": incident.severity.value, "score": incident.hypothesis_score},
        )
    )

    # 2. Suspicious span node
    if incident.first_suspicious_span_id:
        nodes.append(
            GraphNode(
                id=incident.first_suspicious_span_id,
                label=f"Earliest Suspicious Span ({incident.first_suspicious_span_id[:8]})",
                type="span",
                is_suspicious=True,
            )
        )
        edges.append(
            GraphEdge(
                source=hypo_id,
                target=incident.first_suspicious_span_id,
                label="pinpointed_to",
                strength=incident.hypothesis_score,
            )
        )

    # 3. Evidence item nodes
    for ev in incident.evidence:
        ev_node_id = f"node_{ev.evidence_id}"
        nodes.append(
            GraphNode(
                id=ev_node_id,
                label=f"[{ev.type}] {ev.description}",
                type="evidence",
                attributes={"key": ev.key, "strength": ev.strength},
            )
        )
        # Link evidence to hypothesis
        edges.append(
            GraphEdge(
                source=hypo_id,
                target=ev_node_id,
                label="supported_by",
                strength=ev.strength,
            )
        )
        # Link evidence to span if specified
        if ev.span_id:
            edges.append(
                GraphEdge(
                    source=ev_node_id,
                    target=ev.span_id,
                    label="observed_on",
                    strength=ev.strength,
                )
            )

    return EvidenceGraphResponse(
        incident_id=incident.incident_id,
        trace_id=incident.trace_id,
        first_suspicious_span_id=incident.first_suspicious_span_id,
        nodes=nodes,
        edges=edges,
    )
