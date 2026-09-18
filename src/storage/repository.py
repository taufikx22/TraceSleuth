"""TraceSleuth Storage — Repository Pattern.

CRUD operations for traces, spans, incidents, and evidence.
Converts between Pydantic domain models and SQLAlchemy ORM records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from src.domain.incident import (
    Evidence,
    FailureCategory,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from src.domain.trace import (
    SpanData,
    SpanStatus,
    SpanType,
    TraceData,
    TraceEvent,
    TraceOutcome,
)
from src.storage.models import EvidenceRecord, IncidentRecord, SpanRecord, TraceRecord


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware in UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TraceRepository:
    """Repository for persisting and querying traces, spans, incidents, and evidence."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── Trace CRUD ────────────────────────────────────────────

    def save_trace(self, trace: TraceData) -> None:
        """Upsert a full TraceData with all its spans."""
        existing = (
            self.session.query(TraceRecord)
            .filter(TraceRecord.trace_id == trace.trace_id)
            .first()
        )

        if existing:
            # Update existing trace
            existing.request_id = trace.request_id
            existing.session_id = trace.session_id
            existing.tenant_id = trace.tenant_id
            existing.application_version = trace.application_version
            existing.model_configuration_version = trace.model_configuration_version
            existing.prompt_version = trace.prompt_version
            existing.environment = trace.environment
            existing.start_time = trace.start_time
            existing.end_time = trace.end_time
            existing.duration_ms = trace.duration_ms
            existing.outcome = trace.outcome.value
            existing.status = trace.status
            existing.root_span_id = trace.root_span_id
            existing.metadata_json = json.dumps(trace.metadata)

            # Remove old spans and re-insert
            self.session.query(SpanRecord).filter(
                SpanRecord.trace_id == trace.trace_id
            ).delete()
        else:
            record = TraceRecord(
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
                metadata_json=json.dumps(trace.metadata),
            )
            self.session.add(record)

        # Insert spans
        for span in trace.spans:
            events_data = [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in span.events
            ]
            span_record = SpanRecord(
                span_id=span.span_id,
                trace_id=trace.trace_id,
                parent_span_id=span.parent_span_id,
                name=span.name,
                span_type=span.span_type.value,
                start_time=span.start_time,
                end_time=span.end_time,
                duration_ms=span.duration_ms,
                status=span.status.value,
                error_message=span.error_message,
                attributes_json=json.dumps(span.attributes),
                events_json=json.dumps(events_data),
            )
            self.session.add(span_record)

        self.session.commit()

    def get_trace(self, trace_id: str) -> Optional[TraceData]:
        """Load a trace and reconstruct the full Pydantic TraceData model."""
        record = (
            self.session.query(TraceRecord)
            .filter(TraceRecord.trace_id == trace_id)
            .first()
        )
        if not record:
            return None

        return self._record_to_trace_data(record)

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        outcome: Optional[str] = None,
        environment: Optional[str] = None,
        start_after: Optional[datetime] = None,
        start_before: Optional[datetime] = None,
    ) -> Tuple[List[TraceData], int]:
        """List traces with optional filters. Returns (traces, total_count)."""
        query = self.session.query(TraceRecord)

        if outcome:
            query = query.filter(TraceRecord.outcome == outcome)
        if environment:
            query = query.filter(TraceRecord.environment == environment)
        if start_after:
            query = query.filter(TraceRecord.start_time >= start_after)
        if start_before:
            query = query.filter(TraceRecord.start_time <= start_before)

        total = query.count()
        records = query.order_by(desc(TraceRecord.start_time)).offset(offset).limit(limit).all()

        traces = [self._record_to_trace_data(r) for r in records]
        return traces, total

    def search_traces(
        self,
        trace_id_prefix: Optional[str] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        failure_category: Optional[str] = None,
        limit: int = 50,
    ) -> List[TraceData]:
        """Search traces by various criteria."""
        query = self.session.query(TraceRecord)

        if trace_id_prefix:
            query = query.filter(TraceRecord.trace_id.startswith(trace_id_prefix))
        if prompt_version:
            query = query.filter(TraceRecord.prompt_version == prompt_version)
        if failure_category:
            # Join with incidents to filter by category
            query = (
                query.join(IncidentRecord, IncidentRecord.trace_id == TraceRecord.trace_id)
                .filter(IncidentRecord.category == failure_category)
            )

        records = query.order_by(desc(TraceRecord.start_time)).limit(limit).all()
        return [self._record_to_trace_data(r) for r in records]

    def _record_to_trace_data(self, record: TraceRecord) -> TraceData:
        """Convert a TraceRecord (with its SpanRecords) into a Pydantic TraceData."""
        span_records = (
            self.session.query(SpanRecord)
            .filter(SpanRecord.trace_id == record.trace_id)
            .all()
        )

        spans = []
        for sr in span_records:
            events_raw = json.loads(sr.events_json) if sr.events_json else []
            events = [
                TraceEvent(
                    name=e["name"],
                    timestamp=_ensure_utc(datetime.fromisoformat(e["timestamp"])),
                    attributes=e.get("attributes", {}),
                )
                for e in events_raw
            ]

            spans.append(
                SpanData(
                    span_id=sr.span_id,
                    trace_id=sr.trace_id,
                    parent_span_id=sr.parent_span_id,
                    name=sr.name,
                    span_type=SpanType(sr.span_type),
                    start_time=_ensure_utc(sr.start_time),
                    end_time=_ensure_utc(sr.end_time),
                    duration_ms=sr.duration_ms,
                    status=SpanStatus(sr.status),
                    error_message=sr.error_message,
                    attributes=json.loads(sr.attributes_json) if sr.attributes_json else {},
                    events=events,
                )
            )

        metadata = json.loads(record.metadata_json) if record.metadata_json else {}

        return TraceData(
            trace_id=record.trace_id,
            request_id=record.request_id,
            session_id=record.session_id,
            tenant_id=record.tenant_id,
            application_version=record.application_version,
            model_configuration_version=record.model_configuration_version,
            prompt_version=record.prompt_version,
            environment=record.environment,
            start_time=_ensure_utc(record.start_time),
            end_time=_ensure_utc(record.end_time),
            duration_ms=record.duration_ms,
            outcome=TraceOutcome(record.outcome),
            status=record.status,
            root_span_id=record.root_span_id,
            spans=spans,
            metadata=metadata,
        )

    # ── Incident CRUD ─────────────────────────────────────────

    def save_incident(self, incident: Incident) -> None:
        """Persist an incident and its evidence."""
        existing = (
            self.session.query(IncidentRecord)
            .filter(IncidentRecord.incident_id == incident.incident_id)
            .first()
        )

        if existing:
            existing.category = incident.category.value
            existing.severity = incident.severity.value
            existing.summary = incident.summary
            existing.first_suspicious_span_id = incident.first_suspicious_span_id
            existing.hypothesis_score = incident.hypothesis_score
            existing.status = incident.status.value
            existing.resolved_at = incident.resolved_at
            existing.metadata_json = json.dumps(incident.metadata)

            # Refresh evidence
            self.session.query(EvidenceRecord).filter(
                EvidenceRecord.incident_id == incident.incident_id
            ).delete()
        else:
            record = IncidentRecord(
                incident_id=incident.incident_id,
                trace_id=incident.trace_id,
                category=incident.category.value,
                severity=incident.severity.value,
                summary=incident.summary,
                first_suspicious_span_id=incident.first_suspicious_span_id,
                hypothesis_score=incident.hypothesis_score,
                status=incident.status.value,
                created_at=incident.created_at,
                resolved_at=incident.resolved_at,
                metadata_json=json.dumps(incident.metadata),
            )
            self.session.add(record)

        # Insert evidence
        for ev in incident.evidence:
            ev_record = EvidenceRecord(
                evidence_id=ev.evidence_id,
                incident_id=incident.incident_id,
                span_id=ev.span_id,
                type=ev.type,
                key=ev.key,
                value_hash=ev.value_hash,
                description=ev.description,
                strength=ev.strength,
            )
            self.session.add(ev_record)

        self.session.commit()

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Load an incident with its evidence."""
        record = (
            self.session.query(IncidentRecord)
            .filter(IncidentRecord.incident_id == incident_id)
            .first()
        )
        if not record:
            return None

        return self._record_to_incident(record)

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[List[Incident], int]:
        """List incidents with optional filters."""
        query = self.session.query(IncidentRecord)

        if category:
            query = query.filter(IncidentRecord.category == category)
        if severity:
            query = query.filter(IncidentRecord.severity == severity)
        if status:
            query = query.filter(IncidentRecord.status == status)
        if trace_id:
            query = query.filter(IncidentRecord.trace_id == trace_id)

        total = query.count()
        records = (
            query.order_by(desc(IncidentRecord.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

        incidents = [self._record_to_incident(r) for r in records]
        return incidents, total

    def update_incident_status(
        self, incident_id: str, new_status: str, resolved_at: Optional[datetime] = None
    ) -> Optional[Incident]:
        """Update the status of an incident."""
        record = (
            self.session.query(IncidentRecord)
            .filter(IncidentRecord.incident_id == incident_id)
            .first()
        )
        if not record:
            return None

        record.status = new_status
        if resolved_at:
            record.resolved_at = resolved_at
        elif new_status in ("resolved", "false_positive"):
            record.resolved_at = datetime.now(timezone.utc)

        self.session.commit()
        return self._record_to_incident(record)

    def _record_to_incident(self, record: IncidentRecord) -> Incident:
        """Convert an IncidentRecord into a Pydantic Incident."""
        evidence_records = (
            self.session.query(EvidenceRecord)
            .filter(EvidenceRecord.incident_id == record.incident_id)
            .all()
        )

        evidence_list = [
            Evidence(
                evidence_id=er.evidence_id,
                incident_id=er.incident_id,
                span_id=er.span_id,
                type=er.type,
                key=er.key,
                value_hash=er.value_hash,
                description=er.description,
                strength=er.strength,
            )
            for er in evidence_records
        ]

        return Incident(
            incident_id=record.incident_id,
            trace_id=record.trace_id,
            category=FailureCategory(record.category),
            severity=IncidentSeverity(record.severity),
            summary=record.summary,
            first_suspicious_span_id=record.first_suspicious_span_id,
            hypothesis_score=record.hypothesis_score,
            evidence=evidence_list,
            status=IncidentStatus(record.status),
            created_at=_ensure_utc(record.created_at),
            resolved_at=_ensure_utc(record.resolved_at),
            metadata=json.loads(record.metadata_json) if record.metadata_json else {},
        )

    # ── Evidence CRUD ─────────────────────────────────────────

    def get_evidence_for_incident(self, incident_id: str) -> List[Evidence]:
        """Retrieve all evidence for a specific incident."""
        records = (
            self.session.query(EvidenceRecord)
            .filter(EvidenceRecord.incident_id == incident_id)
            .all()
        )

        return [
            Evidence(
                evidence_id=er.evidence_id,
                incident_id=er.incident_id,
                span_id=er.span_id,
                type=er.type,
                key=er.key,
                value_hash=er.value_hash,
                description=er.description,
                strength=er.strength,
            )
            for er in records
        ]
