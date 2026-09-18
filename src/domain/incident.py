"""TraceSleuth Domain Models — Incident and Failure Taxonomy.

Defines the 13-category failure taxonomy, incident severity levels,
evidence structures, and the Incident model for forensic analysis.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """13-category failure taxonomy for AI pipeline forensics (spec Section 4)."""

    RETRIEVAL_FAILURE = "retrieval_failure"
    RANKING_FAILURE = "ranking_failure"
    CONTEXT_FAILURE = "context_failure"
    GENERATION_FAILURE = "generation_failure"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    TOOL_ARGUMENT_FAILURE = "tool_argument_failure"
    POLICY_FAILURE = "policy_failure"
    STATE_FAILURE = "state_failure"
    ROUTING_FAILURE = "routing_failure"
    RETRY_FAILURE = "retry_failure"
    TIMEOUT_FAILURE = "timeout_failure"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    EVALUATION_FAILURE = "evaluation_failure"


class IncidentSeverity(str, Enum):
    """Operational severity levels (spec Section 26)."""

    P0 = "P0"  # Critical unsafe or business-impacting failure
    P1 = "P1"  # Significant reliability or quality failure
    P2 = "P2"  # Degraded experience / limited scope
    P3 = "P3"  # Low-impact anomaly


class IncidentStatus(str, Enum):
    """Lifecycle status of an incident."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class Evidence(BaseModel):
    """A piece of evidence supporting an incident hypothesis."""

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    incident_id: Optional[str] = None
    span_id: Optional[str] = None
    type: str  # e.g. "missing_source", "unsupported_claim", "policy_bypass"
    key: str  # attribute or event key examined
    value_hash: str = ""  # hash of the actual value for privacy
    description: str = ""  # human-readable description of the evidence
    strength: float = 0.0  # 0.0 to 1.0 — how strongly this supports the hypothesis


class Incident(BaseModel):
    """A classified failure incident linked to a trace."""

    incident_id: str = Field(default_factory=lambda: f"inc_{uuid.uuid4().hex[:12]}")
    trace_id: str
    category: FailureCategory
    severity: IncidentSeverity = IncidentSeverity.P2
    summary: str = ""
    first_suspicious_span_id: Optional[str] = None
    hypothesis_score: float = 0.0  # Ranked hypothesis confidence (0.0 - 1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
