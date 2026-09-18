"""TraceSleuth Storage — SQLAlchemy ORM Models.

Defines the relational schema for traces, spans, incidents, and evidence
aligned with the data model in spec Section 25.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all TraceSleuth models."""

    pass


class TraceRecord(Base):
    """Persistent representation of an end-to-end request trace."""

    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(64), unique=True, nullable=False, index=True)
    request_id = Column(String(64), nullable=False, index=True)
    session_id = Column(String(64), nullable=True)
    tenant_id = Column(String(64), nullable=True)
    application_version = Column(String(32), default="0.1.0")
    model_configuration_version = Column(String(32), default="v1.0.0")
    prompt_version = Column(String(32), default="v1.0.0")
    environment = Column(String(32), default="development", index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_ms = Column(Float, default=0.0)
    outcome = Column(String(16), default="success", index=True)
    status = Column(String(32), default="COMPLETED")
    root_span_id = Column(String(32), nullable=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    spans = relationship(
        "SpanRecord", back_populates="trace", cascade="all, delete-orphan"
    )
    incidents = relationship(
        "IncidentRecord", back_populates="trace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_traces_start_time", "start_time"),
        Index("ix_traces_outcome_env", "outcome", "environment"),
    )


class SpanRecord(Base):
    """Persistent representation of a single span within a trace."""

    __tablename__ = "spans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    span_id = Column(String(32), nullable=False, index=True)
    trace_id = Column(
        String(64), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False
    )
    parent_span_id = Column(String(32), nullable=True)
    name = Column(String(256), nullable=False)
    span_type = Column(String(32), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_ms = Column(Float, default=0.0)
    status = Column(String(16), default="OK")
    error_message = Column(Text, nullable=True)
    attributes_json = Column(Text, default="{}")
    events_json = Column(Text, default="[]")

    # Relationships
    trace = relationship("TraceRecord", back_populates="spans")

    __table_args__ = (
        Index("ix_spans_trace_id", "trace_id"),
        Index("ix_spans_type_status", "span_type", "status"),
    )


class IncidentRecord(Base):
    """Persistent representation of a classified failure incident."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String(64), unique=True, nullable=False, index=True)
    trace_id = Column(
        String(64), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False
    )
    category = Column(String(64), nullable=False, index=True)
    severity = Column(String(4), default="P2", index=True)
    summary = Column(Text, default="")
    first_suspicious_span_id = Column(String(32), nullable=True)
    hypothesis_score = Column(Float, default=0.0)
    status = Column(String(32), default="open", index=True)
    failure_signature = Column(String(128), nullable=True, index=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    resolved_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, default="{}")

    # Relationships
    trace = relationship("TraceRecord", back_populates="incidents")
    evidence_items = relationship(
        "EvidenceRecord", back_populates="incident", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_incidents_trace_id", "trace_id"),
        Index("ix_incidents_category_severity", "category", "severity"),
        Index("ix_incidents_failure_sig", "failure_signature"),
    )


class EvidenceRecord(Base):
    """Persistent representation of evidence supporting an incident hypothesis."""

    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(64), unique=True, nullable=False, index=True)
    incident_id = Column(
        String(64),
        ForeignKey("incidents.incident_id", ondelete="CASCADE"),
        nullable=False,
    )
    span_id = Column(String(32), nullable=True)
    type = Column(String(64), nullable=False)
    key = Column(String(256), nullable=False)
    value_hash = Column(String(128), default="")
    description = Column(Text, default="")
    strength = Column(Float, default=0.0)

    # Relationships
    incident = relationship("IncidentRecord", back_populates="evidence_items")

    __table_args__ = (Index("ix_evidence_incident_id", "incident_id"),)


class ReplayRecord(Base):
    """Persistent representation of an exact or controlled replay execution (spec Section 25)."""

    __tablename__ = "replay_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    replay_id = Column(String(64), unique=True, nullable=False, index=True)
    source_trace_id = Column(
        String(64), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False, index=True
    )
    replayed_trace_id = Column(String(64), nullable=True, index=True)
    replay_type = Column(String(32), default="controlled", nullable=False)
    configuration_json = Column(Text, default="{}")
    verdict = Column(String(32), default="inconclusive", nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, default=0.0)
    metrics_diff_json = Column(Text, default="{}")
    summary = Column(Text, default="")
    metadata_json = Column(Text, default="{}")

    # Relationships
    source_trace = relationship("TraceRecord", foreign_keys=[source_trace_id])

    __table_args__ = (
        Index("ix_replays_source_trace", "source_trace_id"),
        Index("ix_replays_verdict", "verdict"),
    )


class RegressionCaseRecord(Base):
    """Persistent representation of a regression test case (spec Section 13 & 25)."""

    __tablename__ = "regression_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), unique=True, nullable=False, index=True)
    incident_id = Column(
        String(64), ForeignKey("incidents.incident_id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_trace_id = Column(
        String(64), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False, index=True
    )
    failure_type = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False)
    input_hash = Column(String(128), default="")
    expected_evidence_json = Column(Text, default="[]")
    assertions_json = Column(Text, default="[]")
    owner = Column(String(64), default="ai-platform")
    introduced_version = Column(String(32), default="1.0.0")
    fixed_version = Column(String(32), nullable=True)
    status = Column(String(32), default="active", nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    metadata_json = Column(Text, default="{}")

    # Relationships
    source_trace = relationship("TraceRecord", foreign_keys=[source_trace_id])
    incident = relationship("IncidentRecord", foreign_keys=[incident_id])

    __table_args__ = (
        Index("ix_regression_status", "status"),
        Index("ix_regression_failure_type", "failure_type"),
    )


class RegressionRunRecord(Base):
    """Persistent record of executing a regression case test."""

    __tablename__ = "regression_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    case_id = Column(
        String(64), ForeignKey("regression_cases.case_id", ondelete="CASCADE"), nullable=False, index=True
    )
    replayed_trace_id = Column(String(64), nullable=True)
    passed = Column(Boolean, default=False, nullable=False)
    assertion_results_json = Column(Text, default="[]")
    executed_at = Column(DateTime, nullable=False)
    duration_ms = Column(Float, default=0.0)
    summary = Column(Text, default="")
