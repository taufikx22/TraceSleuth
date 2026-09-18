"""Metrics and Observability Service (Milestone 9).

Computes SLO/SLA metrics, latency decomposition across pipeline stages,
quality pass rate vs availability, and active alert evaluations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.domain.incident import IncidentSeverity
from src.domain.trace import SpanType, TraceData, TraceOutcome
from src.metrics.alerts import AlertEngine, AlertItem
from src.metrics.cost import CostForensicsCalculator, TraceCostEstimate
from src.storage.database import DatabaseManager, get_database
from src.storage.repository import TraceRepository


class MetricsService:
    """Service providing latency forensics, SLO/SLA metrics, and alerting."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        cost_calc: Optional[CostForensicsCalculator] = None,
        alert_engine: Optional[AlertEngine] = None,
    ) -> None:
        self.db = db or get_database()
        self.cost_calc = cost_calc or CostForensicsCalculator()
        self.alert_engine = alert_engine or AlertEngine()

    def get_latency_breakdown(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Decomposes the total latency of a trace into individual pipeline stages (spec Section 17)."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        trace = repo.get_trace(trace_id)
        session.close()

        if not trace:
            return None

        total_ms = trace.duration_ms
        stages = {
            "router_ms": 0.0,
            "retrieval_ms": 0.0,
            "reranker_ms": 0.0,
            "planner_ms": 0.0,
            "model_ms": 0.0,
            "tool_ms": 0.0,
            "citation_ms": 0.0,
            "other_ms": 0.0,
        }

        for span in trace.spans:
            dur = span.duration_ms
            if span.span_type == SpanType.CLASSIFIER:
                stages["router_ms"] += dur
            elif span.span_type == SpanType.RETRIEVAL:
                stages["retrieval_ms"] += dur
            elif span.span_type == SpanType.RERANKER:
                stages["reranker_ms"] += dur
            elif span.span_type == SpanType.PLANNER:
                stages["planner_ms"] += dur
            elif span.span_type == SpanType.MODEL:
                stages["model_ms"] += dur
            elif span.span_type == SpanType.TOOL:
                stages["tool_ms"] += dur
            elif span.span_type == SpanType.CITATION:
                stages["citation_ms"] += dur
            elif span.span_type != SpanType.AGENT_RUN:
                stages["other_ms"] += dur

        # Round values
        return {
            "trace_id": trace.trace_id,
            "total_latency_ms": round(total_ms, 2),
            **{k: round(v, 2) for k, v in stages.items()},
        }

    def get_sla_metrics(self, limit: int = 200) -> Dict[str, Any]:
        """Calculates p50/p95/p99 latency, quality pass rate, availability, and costs (spec Section 15)."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        traces, total_count = repo.list_traces(limit=limit)
        incidents, _ = repo.list_incidents(limit=limit * 2)
        session.close()

        if not traces:
            return {
                "total_traces": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "availability_rate": 1.0,
                "quality_pass_rate": 1.0,
                "error_rate": 0.0,
                "tool_failure_rate": 0.0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }

        # Latency percentiles
        durations = sorted([t.duration_ms for t in traces])
        n = len(durations)
        p50 = durations[int(0.50 * (n - 1))]
        p95 = durations[int(0.95 * (n - 1))]
        p99 = durations[int(0.99 * (n - 1))]

        # Availability: traces with outcome == SUCCESS
        successful_traces = sum(1 for t in traces if t.outcome == TraceOutcome.SUCCESS)
        availability = successful_traces / n

        # Quality pass rate: traces without high-severity (P0/P1) incidents
        severe_trace_ids = {
            inc.trace_id for inc in incidents
            if inc.severity in (IncidentSeverity.P0, IncidentSeverity.P1)
        }
        quality_passed_traces = sum(1 for t in traces if t.trace_id not in severe_trace_ids)
        quality_rate = quality_passed_traces / n

        # Tool failure rate
        total_tools = 0
        failed_tools = 0
        for t in traces:
            for s in t.spans:
                if s.span_type == SpanType.TOOL:
                    total_tools += 1
                    if s.status.value != "OK":
                        failed_tools += 1
        tool_failure_rate = (failed_tools / total_tools) if total_tools > 0 else 0.0

        # Token & cost accounting
        total_tokens = sum(t.total_tokens() for t in traces)
        total_cost = sum(self.cost_calc.calculate_trace_cost(t).total_cost_usd for t in traces)

        return {
            "total_traces": n,
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "availability_rate": round(availability, 4),
            "quality_pass_rate": round(quality_rate, 4),
            "error_rate": round(1.0 - availability, 4),
            "tool_failure_rate": round(tool_failure_rate, 4),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(total_cost, 6),
        }

    def get_cost_breakdown(self, limit: int = 100) -> Dict[str, Any]:
        """Calculates token costs and average cost per request (spec Section 16)."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        traces, _ = repo.list_traces(limit=limit)
        session.close()

        total_cost = 0.0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0

        for t in traces:
            est = self.cost_calc.calculate_trace_cost(t)
            total_cost += est.total_cost_usd
            total_tokens += est.total_tokens
            input_tokens += est.input_tokens
            output_tokens += est.output_tokens

        n = len(traces)
        avg_cost = (total_cost / n) if n > 0 else 0.0

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_per_request_usd": round(avg_cost, 6),
            "model_pricing_version": "2026-Q1",
        }

    def get_alerts(self, limit: int = 100) -> List[AlertItem]:
        """Evaluates live system alerts."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        incidents, _ = repo.list_incidents(limit=limit)
        session.close()

        sla = self.get_sla_metrics(limit=limit)
        return self.alert_engine.evaluate(sla, incidents)
