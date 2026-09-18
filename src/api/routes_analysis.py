"""TraceSleuth API — Model-Assisted Forensic Analysis Routes (Milestone 10).

Endpoints for producing grounded, model-assisted root-cause explanations
with accountable citations back to concrete span telemetry.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from src.analysis.model_assisted import ModelAssistedForensicAnalyst, ModelHypothesisResult
from src.api.schemas import EvidenceCitationResponse, ModelHypothesisResponse
from src.storage.database import DatabaseManager, get_database
from src.storage.repository import TraceRepository

router = APIRouter(tags=["Model-Assisted Forensics (Milestone 10)"])


def get_repo(db: DatabaseManager = Depends(get_database)) -> TraceRepository:
    session = db.get_session()
    try:
        yield TraceRepository(session)
    finally:
        session.close()


def _hypothesis_to_response(res: ModelHypothesisResult) -> ModelHypothesisResponse:
    return ModelHypothesisResponse(
        trace_id=res.trace_id,
        incident_id=res.incident_id,
        primary_root_cause=res.primary_root_cause.value,
        confidence=res.confidence,
        explanation=res.explanation,
        evidence_citations=[
            EvidenceCitationResponse(
                span_id=c.span_id,
                evidence_key=c.evidence_key,
                description=c.description,
                strength=c.strength,
            )
            for c in res.evidence_citations
        ],
        recommended_remediation=res.recommended_remediation,
        engine=res.engine,
        generated_at=res.generated_at,
    )


@router.post(
    "/incidents/{incident_id}/model-hypothesis",
    response_model=ModelHypothesisResponse,
    summary="Generate Model-Assisted Explanation for Incident",
)
def generate_incident_model_hypothesis(
    incident_id: str,
    repo: TraceRepository = Depends(get_repo),
) -> ModelHypothesisResponse:
    """Uses model-assisted forensic reasoning to explain an incident while keeping hypotheses strictly grounded in telemetry."""
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")

    trace = repo.get_trace(incident.trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{incident.trace_id}' not found.")

    analyst = ModelAssistedForensicAnalyst()
    result = analyst.analyze(trace, incident=incident)
    return _hypothesis_to_response(result)


@router.post(
    "/traces/{trace_id}/model-hypothesis",
    response_model=ModelHypothesisResponse,
    summary="Generate Model-Assisted Explanation for Trace",
)
def generate_trace_model_hypothesis(
    trace_id: str,
    repo: TraceRepository = Depends(get_repo),
) -> ModelHypothesisResponse:
    """Analyzes a trace directly and generates a grounded hypothesis with verified evidence citations."""
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")

    analyst = ModelAssistedForensicAnalyst()
    result = analyst.analyze(trace)
    return _hypothesis_to_response(result)
