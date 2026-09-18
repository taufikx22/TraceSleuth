"""Alerting Engine and Evaluation (Milestone 9).

Monitors system thresholds and baseline metrics for critical policy violations,
quality-pass-rate drops, latency breaches, retry storms, and cost spikes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.domain.incident import Incident, IncidentSeverity


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class AlertStatus(str, Enum):
    """Lifecycle status of an alert."""

    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertItem:
    """An alert notification triggered by an SLO breach or critical incident."""

    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


class AlertEngine:
    """Evaluates telemetry, metrics, and incident streams against alert rules."""

    def __init__(
        self,
        quality_pass_rate_threshold: float = 0.95,
        p99_latency_threshold_ms: float = 350.0,
        cost_per_request_threshold_usd: float = 0.05,
    ) -> None:
        self.quality_pass_rate_threshold = quality_pass_rate_threshold
        self.p99_latency_threshold_ms = p99_latency_threshold_ms
        self.cost_per_request_threshold_usd = cost_per_request_threshold_usd

    def evaluate(
        self,
        sla_metrics: Dict[str, Any],
        recent_incidents: List[Incident],
    ) -> List[AlertItem]:
        """Evaluates current system metrics and recent incidents to produce active alerts."""
        alerts: List[AlertItem] = []
        now = datetime.now(timezone.utc)

        # 1. Critical Policy Bypass (P0) Alert (spec Section 27)
        policy_incidents = [
            inc for inc in recent_incidents
            if inc.severity == IncidentSeverity.P0 and inc.status.value in ("open", "investigating", "confirmed")
        ]
        if policy_incidents:
            alerts.append(
                AlertItem(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    name="CriticalPolicyBypass",
                    severity=AlertSeverity.P0,
                    status=AlertStatus.FIRING,
                    triggered_at=now,
                    summary=f"{len(policy_incidents)} critical P0 policy bypass incidents detected!",
                    details={
                        "incident_ids": [i.incident_id for i in policy_incidents],
                        "trace_ids": [i.trace_id for i in policy_incidents],
                    },
                )
            )

        # 2. Quality Pass Rate Decline Alert
        quality_rate = sla_metrics.get("quality_pass_rate", 1.0)
        total_traces = sla_metrics.get("total_traces", 0)
        if total_traces >= 5 and quality_rate < self.quality_pass_rate_threshold:
            alerts.append(
                AlertItem(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    name="QualityPassRateDrop",
                    severity=AlertSeverity.P1,
                    status=AlertStatus.FIRING,
                    triggered_at=now,
                    summary=f"Quality pass rate fell to {quality_rate * 100:.1f}% (SLO threshold: {self.quality_pass_rate_threshold * 100:.1f}%)",
                    details={
                        "quality_rate": quality_rate,
                        "threshold": self.quality_pass_rate_threshold,
                        "total_traces_evaluated": total_traces,
                    },
                )
            )

        # 3. p99 Latency Breach Alert
        p99_latency = sla_metrics.get("p99_latency_ms", 0.0)
        if total_traces >= 5 and p99_latency > self.p99_latency_threshold_ms:
            alerts.append(
                AlertItem(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    name="P99LatencyBreach",
                    severity=AlertSeverity.P2,
                    status=AlertStatus.FIRING,
                    triggered_at=now,
                    summary=f"p99 latency reached {p99_latency:.1f}ms exceeding SLA threshold ({self.p99_latency_threshold_ms:.1f}ms)",
                    details={
                        "p99_latency_ms": p99_latency,
                        "threshold_ms": self.p99_latency_threshold_ms,
                    },
                )
            )

        # 4. Retry Storm Alert
        retry_incidents = [
            inc for inc in recent_incidents
            if inc.category.value in ("retry_failure", "tool_selection_failure")
            and inc.status.value in ("open", "investigating", "confirmed")
        ]
        if len(retry_incidents) >= 2:
            alerts.append(
                AlertItem(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    name="ToolRetryStorm",
                    severity=AlertSeverity.P1,
                    status=AlertStatus.FIRING,
                    triggered_at=now,
                    summary=f"{len(retry_incidents)} retry storm / failure incidents detected across recent runs.",
                    details={"incident_ids": [i.incident_id for i in retry_incidents]},
                )
            )

        # 5. Token Cost Spike Alert
        total_cost = sla_metrics.get("estimated_cost_usd", 0.0)
        avg_cost = (total_cost / total_traces) if total_traces > 0 else 0.0
        if avg_cost > self.cost_per_request_threshold_usd:
            alerts.append(
                AlertItem(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    name="CostPerRequestSpike",
                    severity=AlertSeverity.P2,
                    status=AlertStatus.FIRING,
                    triggered_at=now,
                    summary=f"Average cost per request (${avg_cost:.4f}) exceeded budget (${self.cost_per_request_threshold_usd:.4f})",
                    details={"avg_cost_usd": avg_cost, "threshold_usd": self.cost_per_request_threshold_usd},
                )
            )

        return alerts
