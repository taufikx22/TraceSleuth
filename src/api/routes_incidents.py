"""TraceSleuth API — Incident Routes.

Endpoints for listing, inspecting, and updating incident status.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    EvidenceResponse,
    IncidentListItem,
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusUpdate,
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
