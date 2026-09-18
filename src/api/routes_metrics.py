"""TraceSleuth API — Metrics & Alerts Routes (Milestone 9).

Endpoints for SLO/SLA tracking, latency stage decomposition,
token cost forensics, and active alert evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import (
    AlertResponse,
    CostBreakdownResponse,
    LatencyBreakdownResponse,
    SLAMetricsResponse,
)
from src.metrics.service import MetricsService
from src.storage.database import DatabaseManager, get_database

router = APIRouter(prefix="/metrics", tags=["Metrics & Alerts (Milestone 9)"])


def get_metrics_service(db: DatabaseManager = Depends(get_database)) -> MetricsService:
    return MetricsService(db)


@router.get(
    "/summary",
    response_model=SLAMetricsResponse,
    summary="Get System SLO / SLA Metrics Summary",
)
def get_metrics_summary(
    limit: int = Query(200, ge=1, le=1000),
    svc: MetricsService = Depends(get_metrics_service),
) -> SLAMetricsResponse:
    """Returns high-level reliability metrics: p50/p95/p99 latency, quality pass rate, availability, and cost."""
    metrics = svc.get_sla_metrics(limit=limit)
    return SLAMetricsResponse(**metrics)


@router.get(
    "/latency",
    summary="Get Latency Percentiles",
)
def get_latency_metrics(
    limit: int = Query(200, ge=1, le=1000),
    svc: MetricsService = Depends(get_metrics_service),
) -> Dict[str, Any]:
    """Returns p50, p95, and p99 latency percentiles across historical traces."""
    metrics = svc.get_sla_metrics(limit=limit)
    return {
        "p50_latency_ms": metrics["p50_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "p99_latency_ms": metrics["p99_latency_ms"],
        "total_traces_sampled": metrics["total_traces"],
    }


@router.get(
    "/latency/{trace_id}",
    response_model=LatencyBreakdownResponse,
    summary="Get Latency Stage Decomposition for Trace",
)
def get_trace_latency_breakdown(
    trace_id: str,
    svc: MetricsService = Depends(get_metrics_service),
) -> LatencyBreakdownResponse:
    """Decomposes the total duration into individual pipeline stages (router, retrieval, model, tools, etc.)."""
    breakdown = svc.get_latency_breakdown(trace_id)
    if not breakdown:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return LatencyBreakdownResponse(**breakdown)


@router.get(
    "/quality",
    summary="Get Quality vs Availability Metrics",
)
def get_quality_metrics(
    limit: int = Query(200, ge=1, le=1000),
    svc: MetricsService = Depends(get_metrics_service),
) -> Dict[str, Any]:
    """Demonstrates that a service can have 99.9% HTTP availability while quality pass rate is lower (spec Section 15)."""
    metrics = svc.get_sla_metrics(limit=limit)
    return {
        "availability_rate": metrics["availability_rate"],
        "quality_pass_rate": metrics["quality_pass_rate"],
        "error_rate": metrics["error_rate"],
        "tool_failure_rate": metrics["tool_failure_rate"],
        "total_traces": metrics["total_traces"],
    }


@router.get(
    "/cost",
    response_model=CostBreakdownResponse,
    summary="Get Token Usage & Cost Forensics",
)
def get_cost_metrics(
    limit: int = Query(200, ge=1, le=1000),
    svc: MetricsService = Depends(get_metrics_service),
) -> CostBreakdownResponse:
    """Calculates financial token usage and estimated cost across requests."""
    cost_data = svc.get_cost_breakdown(limit=limit)
    return CostBreakdownResponse(**cost_data)


# Top-level /alerts route
alerts_router = APIRouter(tags=["Metrics & Alerts (Milestone 9)"])


@alerts_router.get(
    "/alerts",
    response_model=List[AlertResponse],
    summary="Get Active Alerts",
)
def get_active_alerts(
    limit: int = Query(100, ge=1, le=500),
    svc: MetricsService = Depends(get_metrics_service),
) -> List[AlertResponse]:
    """Evaluates active system alerts for critical policy violations, latency breaches, and quality drops."""
    alerts = svc.get_alerts(limit=limit)
    return [
        AlertResponse(
            alert_id=a.alert_id,
            name=a.name,
            severity=a.severity.value,
            status=a.status.value,
            triggered_at=a.triggered_at,
            summary=a.summary,
            details=a.details,
        )
        for a in alerts
    ]
