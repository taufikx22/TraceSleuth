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
from src.storage.models import (
    EvidenceRecord,
    IncidentRecord,
    RegressionCaseRecord,
    RegressionRunRecord,
    ReplayRecord,
    SpanRecord,
    TraceRecord,
)
from src.domain.regression import (
    AssertionOperator,
    AssertionResult,
    RegressionAssertion,
    RegressionCase,
    RegressionRunResult,
    RegressionStatus,
)
from src.replay.types import (
    ReplayConfig,
    ReplayMetricsDiff,
    ReplayRun,
    ReplayType,
    ReplayVerdict,
)
from src.analysis.similarity import FailureSignature, compute_incident_similarity


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
        sig = FailureSignature.generate(incident).signature_str

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
            existing.failure_signature = sig
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
                failure_signature=sig,
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

    # ── Similar Incidents Search (Sections 11 & 31) ───────────

    def find_similar_incidents(
        self, incident_id: str, limit: int = 5
    ) -> List[Tuple[Incident, float]]:
        """Find historical incidents similar to the target incident."""
        target = self.get_incident(incident_id)
        if not target:
            return []

        # Query other incidents with matching category
        candidates = (
            self.session.query(IncidentRecord)
            .filter(
                IncidentRecord.category == target.category.value,
                IncidentRecord.incident_id != incident_id,
            )
            .order_by(desc(IncidentRecord.created_at))
            .limit(50)
            .all()
        )

        results: List[Tuple[Incident, float]] = []
        for c_rec in candidates:
            other_inc = self._record_to_incident(c_rec)
            sim = compute_incident_similarity(target, other_inc)
            if sim > 0.0:
                results.append((other_inc, round(sim, 2)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ── Replay Runs CRUD (Sections 21, 25 & 32) ────────────────

    def save_replay(self, replay: ReplayRun) -> None:
        """Persist a replay execution record."""
        existing = (
            self.session.query(ReplayRecord)
            .filter(ReplayRecord.replay_id == replay.replay_id)
            .first()
        )

        cfg_json = json.dumps(replay.config.model_dump())
        diff_json = json.dumps(replay.metrics_diff.model_dump()) if replay.metrics_diff else "{}"
        meta_json = json.dumps(replay.metadata)

        if existing:
            existing.replayed_trace_id = replay.replayed_trace_id
            existing.replay_type = replay.replay_type.value
            existing.configuration_json = cfg_json
            existing.verdict = replay.verdict.value
            existing.completed_at = replay.completed_at
            existing.duration_ms = replay.duration_ms
            existing.metrics_diff_json = diff_json
            existing.summary = replay.summary
            existing.metadata_json = meta_json
        else:
            record = ReplayRecord(
                replay_id=replay.replay_id,
                source_trace_id=replay.source_trace_id,
                replayed_trace_id=replay.replayed_trace_id,
                replay_type=replay.replay_type.value,
                configuration_json=cfg_json,
                verdict=replay.verdict.value,
                started_at=replay.started_at,
                completed_at=replay.completed_at,
                duration_ms=replay.duration_ms,
                metrics_diff_json=diff_json,
                summary=replay.summary,
                metadata_json=meta_json,
            )
            self.session.add(record)

        self.session.commit()

    def get_replay(self, replay_id: str) -> Optional[ReplayRun]:
        """Load a replay record by ID."""
        record = (
            self.session.query(ReplayRecord)
            .filter(ReplayRecord.replay_id == replay_id)
            .first()
        )
        if not record:
            return None
        return self._record_to_replay_run(record)

    def list_replays(
        self, source_trace_id: Optional[str] = None, limit: int = 50
    ) -> List[ReplayRun]:
        """List replays with optional source_trace_id filter."""
        query = self.session.query(ReplayRecord)
        if source_trace_id:
            query = query.filter(ReplayRecord.source_trace_id == source_trace_id)
        records = query.order_by(desc(ReplayRecord.started_at)).limit(limit).all()
        return [self._record_to_replay_run(r) for r in records]

    def _record_to_replay_run(self, record: ReplayRecord) -> ReplayRun:
        """Convert a ReplayRecord into a Pydantic ReplayRun model."""
        cfg_dict = json.loads(record.configuration_json) if record.configuration_json else {}
        diff_dict = json.loads(record.metrics_diff_json) if record.metrics_diff_json else None
        meta_dict = json.loads(record.metadata_json) if record.metadata_json else {}

        metrics_diff = ReplayMetricsDiff(**diff_dict) if diff_dict else None

        return ReplayRun(
            replay_id=record.replay_id,
            source_trace_id=record.source_trace_id,
            replayed_trace_id=record.replayed_trace_id,
            replay_type=ReplayType(record.replay_type),
            config=ReplayConfig(**cfg_dict),
            verdict=ReplayVerdict(record.verdict),
            started_at=_ensure_utc(record.started_at),
            completed_at=_ensure_utc(record.completed_at),
            duration_ms=record.duration_ms,
            metrics_diff=metrics_diff,
            summary=record.summary,
            metadata=meta_dict,
        )

    # ── Regression Cases CRUD (Sections 12, 13, 25 & Milestone 8) ───

    def save_regression_case(self, case: RegressionCase) -> None:
        """Persist or update a regression test case."""
        existing = (
            self.session.query(RegressionCaseRecord)
            .filter(RegressionCaseRecord.case_id == case.id)
            .first()
        )

        ev_json = json.dumps(case.expected_evidence)
        asrt_json = json.dumps([a.model_dump() for a in case.assertions])
        meta_json = json.dumps(case.metadata)

        if existing:
            existing.status = case.status.value
            existing.fixed_version = case.fixed_version
            existing.assertions_json = asrt_json
            existing.expected_evidence_json = ev_json
            existing.metadata_json = meta_json
        else:
            rec = RegressionCaseRecord(
                case_id=case.id,
                incident_id=case.incident_id,
                source_trace_id=case.source_trace_id,
                failure_type=case.failure_type,
                query=case.query,
                input_hash=case.input_hash,
                expected_evidence_json=ev_json,
                assertions_json=asrt_json,
                owner=case.owner,
                introduced_version=case.introduced_version,
                fixed_version=case.fixed_version,
                status=case.status.value,
                created_at=case.created_at,
                metadata_json=meta_json,
            )
            self.session.add(rec)

        self.session.commit()

    def get_regression_case(self, case_id: str) -> Optional[RegressionCase]:
        """Load a regression test case by ID."""
        record = (
            self.session.query(RegressionCaseRecord)
            .filter(RegressionCaseRecord.case_id == case_id)
            .first()
        )
        if not record:
            return None
        return self._record_to_regression_case(record)

    def list_regression_cases(
        self,
        status: Optional[str] = None,
        failure_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[RegressionCase], int]:
        """List regression test cases with optional filters."""
        query = self.session.query(RegressionCaseRecord)
        if status:
            query = query.filter(RegressionCaseRecord.status == status)
        if failure_type:
            query = query.filter(RegressionCaseRecord.failure_type == failure_type)

        total = query.count()
        records = (
            query.order_by(desc(RegressionCaseRecord.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [self._record_to_regression_case(r) for r in records], total

    def save_regression_run(self, result: RegressionRunResult) -> None:
        """Persist the result of executing a regression test run."""
        import uuid
        run_id = f"regrun_{uuid.uuid4().hex[:12]}"
        asrt_json = json.dumps([a.model_dump() for a in result.assertion_results])
        rec = RegressionRunRecord(
            run_id=run_id,
            case_id=result.case_id,
            replayed_trace_id=result.replayed_trace_id,
            passed=result.passed,
            assertion_results_json=asrt_json,
            executed_at=result.executed_at,
            duration_ms=result.duration_ms,
            summary=result.summary,
        )
        self.session.add(rec)
        self.session.commit()

    def _record_to_regression_case(self, record: RegressionCaseRecord) -> RegressionCase:
        """Convert a RegressionCaseRecord to a Pydantic RegressionCase."""
        ev_list = json.loads(record.expected_evidence_json) if record.expected_evidence_json else []
        asrt_raw = json.loads(record.assertions_json) if record.assertions_json else []
        meta_dict = json.loads(record.metadata_json) if record.metadata_json else {}

        assertions = [
            RegressionAssertion(
                assertion_id=a["assertion_id"],
                name=a["name"],
                target_metric=a["target_metric"],
                operator=AssertionOperator(a.get("operator", "eq")),
                expected_value=a["expected_value"],
                description=a.get("description", ""),
            )
            for a in asrt_raw
        ]

        return RegressionCase(
            id=record.case_id,
            incident_id=record.incident_id or "",
            source_trace_id=record.source_trace_id,
            failure_type=record.failure_type,
            query=record.query,
            input_hash=record.input_hash,
            expected_evidence=ev_list,
            assertions=assertions,
            owner=record.owner,
            introduced_version=record.introduced_version,
            fixed_version=record.fixed_version,
            status=RegressionStatus(record.status),
            created_at=_ensure_utc(record.created_at),
            metadata=meta_dict,
        )

