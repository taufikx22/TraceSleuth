"""TraceSleuth API — Replay Routes.

Endpoints for triggering exact and controlled replays of historical traces,
inspecting replay results, and retrieving metric differences (spec Section 29).
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    ReplayListResponse,
    ReplayMetricsDiffResponse,
    ReplayRequest,
    ReplayResponse,
)
from src.replay.service import ReplayService
from src.replay.types import ReplayConfig, ReplayType
from src.storage.database import get_database
from src.storage.repository import TraceRepository

router = APIRouter(tags=["replay"])


def _get_service() -> ReplayService:
    return ReplayService(get_database())


def _get_repo() -> TraceRepository:
    return TraceRepository(get_database().get_session())


@router.post("/replay", response_model=ReplayResponse)
def trigger_replay(payload: ReplayRequest) -> ReplayResponse:
    """Execute a replay of a historical trace with configuration overrides (spec Section 21 & 32)."""
    service = _get_service()

    try:
        replay_type = ReplayType(payload.replay_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid replay_type '{payload.replay_type}'. Must be one of: {[e.value for e in ReplayType]}",
        )

    config = ReplayConfig(
        replay_type=replay_type,
        retriever_mode=payload.retriever_mode,
        vector_threshold=payload.vector_threshold,
        model_name=payload.model_name,
        prompt_version=payload.prompt_version,
        bypass_policy=payload.bypass_policy,
        tool_failure_rate=payload.tool_failure_rate,
        custom_parameters=payload.custom_parameters,
    )

    try:
        run = service.execute_replay(payload.source_trace_id, config=config)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Replay execution failed: {str(e)}")

    diff_resp = None
    if run.metrics_diff:
        diff_resp = ReplayMetricsDiffResponse(
            original_duration_ms=run.metrics_diff.original_duration_ms,
            replayed_duration_ms=run.metrics_diff.replayed_duration_ms,
            duration_diff_ms=run.metrics_diff.duration_diff_ms,
            original_tokens=run.metrics_diff.original_tokens,
            replayed_tokens=run.metrics_diff.replayed_tokens,
            tokens_diff=run.metrics_diff.tokens_diff,
            original_span_count=run.metrics_diff.original_span_count,
            replayed_span_count=run.metrics_diff.replayed_span_count,
            original_outcome=run.metrics_diff.original_outcome,
            replayed_outcome=run.metrics_diff.replayed_outcome,
            resolved_incident_categories=run.metrics_diff.resolved_incident_categories,
            new_incident_categories=run.metrics_diff.new_incident_categories,
        )

    return ReplayResponse(
        replay_id=run.replay_id,
        source_trace_id=run.source_trace_id,
        replayed_trace_id=run.replayed_trace_id,
        replay_type=run.replay_type.value,
        verdict=run.verdict.value,
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        metrics_diff=diff_resp,
        summary=run.summary,
    )


@router.get("/replay/{replay_id}", response_model=ReplayResponse)
def get_replay(replay_id: str) -> ReplayResponse:
    """Retrieve details and metrics comparison for a specific replay run."""
    repo = _get_repo()
    run = repo.get_replay(replay_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Replay {replay_id} not found")

    diff_resp = None
    if run.metrics_diff:
        diff_resp = ReplayMetricsDiffResponse(
            original_duration_ms=run.metrics_diff.original_duration_ms,
            replayed_duration_ms=run.metrics_diff.replayed_duration_ms,
            duration_diff_ms=run.metrics_diff.duration_diff_ms,
            original_tokens=run.metrics_diff.original_tokens,
            replayed_tokens=run.metrics_diff.replayed_tokens,
            tokens_diff=run.metrics_diff.tokens_diff,
            original_span_count=run.metrics_diff.original_span_count,
            replayed_span_count=run.metrics_diff.replayed_span_count,
            original_outcome=run.metrics_diff.original_outcome,
            replayed_outcome=run.metrics_diff.replayed_outcome,
            resolved_incident_categories=run.metrics_diff.resolved_incident_categories,
            new_incident_categories=run.metrics_diff.new_incident_categories,
        )

    return ReplayResponse(
        replay_id=run.replay_id,
        source_trace_id=run.source_trace_id,
        replayed_trace_id=run.replayed_trace_id,
        replay_type=run.replay_type.value,
        verdict=run.verdict.value,
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        metrics_diff=diff_resp,
        summary=run.summary,
    )


@router.get("/traces/{trace_id}/replays", response_model=ReplayListResponse)
def list_trace_replays(
    trace_id: str, limit: int = Query(default=50, ge=1, le=100)
) -> ReplayListResponse:
    """List all historical replays conducted against a specific trace."""
    repo = _get_repo()
    runs = repo.list_replays(source_trace_id=trace_id, limit=limit)

    items = []
    for run in runs:
        diff_resp = None
        if run.metrics_diff:
            diff_resp = ReplayMetricsDiffResponse(
                original_duration_ms=run.metrics_diff.original_duration_ms,
                replayed_duration_ms=run.metrics_diff.replayed_duration_ms,
                duration_diff_ms=run.metrics_diff.duration_diff_ms,
                original_tokens=run.metrics_diff.original_tokens,
                replayed_tokens=run.metrics_diff.replayed_tokens,
                tokens_diff=run.metrics_diff.tokens_diff,
                original_span_count=run.metrics_diff.original_span_count,
                replayed_span_count=run.metrics_diff.replayed_span_count,
                original_outcome=run.metrics_diff.original_outcome,
                replayed_outcome=run.metrics_diff.replayed_outcome,
                resolved_incident_categories=run.metrics_diff.resolved_incident_categories,
                new_incident_categories=run.metrics_diff.new_incident_categories,
            )
        items.append(
            ReplayResponse(
                replay_id=run.replay_id,
                source_trace_id=run.source_trace_id,
                replayed_trace_id=run.replayed_trace_id,
                replay_type=run.replay_type.value,
                verdict=run.verdict.value,
                started_at=run.started_at,
                completed_at=run.completed_at,
                duration_ms=run.duration_ms,
                metrics_diff=diff_resp,
                summary=run.summary,
            )
        )

    return ReplayListResponse(replays=items, total=len(items))
