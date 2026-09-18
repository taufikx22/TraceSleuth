"""TraceSleuth API — Regression Conversion Routes (Milestone 8).

Endpoints for converting confirmed failure incidents into regression test cases,
running regression suites, and exporting CI/CD evaluation datasets.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from src.api.schemas import (
    CreateRegressionCaseRequest,
    RegressionAssertionResponse,
    RegressionCaseResponse,
    RegressionRunResponse,
)
from src.domain.regression import RegressionCase, RegressionRunResult
from src.regression.service import RegressionService
from src.storage.database import DatabaseManager, get_database
from src.storage.repository import TraceRepository

router = APIRouter(tags=["Regression Cases (Milestone 8)"])


def get_repo(db: DatabaseManager = Depends(get_database)) -> TraceRepository:
    session = db.get_session()
    try:
        yield TraceRepository(session)
    finally:
        session.close()


def get_regression_service(db: DatabaseManager = Depends(get_database)) -> RegressionService:
    return RegressionService(db)


def _case_to_response(c: RegressionCase) -> RegressionCaseResponse:
    return RegressionCaseResponse(
        id=c.id,
        incident_id=c.incident_id,
        source_trace_id=c.source_trace_id,
        failure_type=c.failure_type,
        query=c.query,
        input_hash=c.input_hash,
        expected_evidence=c.expected_evidence,
        assertions=[
            RegressionAssertionResponse(
                assertion_id=a.assertion_id,
                name=a.name,
                target_metric=a.target_metric,
                operator=a.operator.value,
                expected_value=a.expected_value,
                description=a.description,
            )
            for a in c.assertions
        ],
        owner=c.owner,
        introduced_version=c.introduced_version,
        fixed_version=c.fixed_version,
        status=c.status.value,
        created_at=c.created_at,
    )


@router.post(
    "/incidents/{incident_id}/regression",
    response_model=RegressionCaseResponse,
    status_code=201,
    summary="Convert Confirmed Incident to Regression Case",
)
def create_regression_case(
    incident_id: str,
    req: CreateRegressionCaseRequest = CreateRegressionCaseRequest(),
    svc: RegressionService = Depends(get_regression_service),
) -> RegressionCaseResponse:
    """Converts an investigated failure incident into a permanent regression test case."""
    case = svc.create_case_from_incident(
        incident_id=incident_id,
        owner=req.owner,
        introduced_version=req.introduced_version,
    )
    if not case:
        raise HTTPException(
            status_code=404,
            detail=f"Incident '{incident_id}' not found or has no associated source trace.",
        )
    return _case_to_response(case)


@router.get(
    "/regression/cases",
    response_model=List[RegressionCaseResponse],
    summary="List Regression Test Cases",
)
def list_regression_cases(
    status: Optional[str] = Query(None, description="Filter by status: active, passed, failed, retired"),
    failure_type: Optional[str] = Query(None, description="Filter by failure category"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo: TraceRepository = Depends(get_repo),
) -> List[RegressionCaseResponse]:
    """Lists regression test cases with optional filtering."""
    cases, _ = repo.list_regression_cases(
        status=status, failure_type=failure_type, limit=limit, offset=offset
    )
    return [_case_to_response(c) for c in cases]


@router.get(
    "/regression/cases/{case_id}",
    response_model=RegressionCaseResponse,
    summary="Get Regression Test Case Details",
)
def get_regression_case(
    case_id: str,
    repo: TraceRepository = Depends(get_repo),
) -> RegressionCaseResponse:
    """Fetches a specific regression test case by ID."""
    case = repo.get_regression_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Regression case '{case_id}' not found.")
    return _case_to_response(case)


@router.post(
    "/regression/cases/{case_id}/run",
    response_model=RegressionRunResponse,
    summary="Run Regression Test Case",
)
def run_regression_case(
    case_id: str,
    svc: RegressionService = Depends(get_regression_service),
) -> RegressionRunResponse:
    """Executes the reference agent against the regression case query and verifies all assertions."""
    result = svc.run_case(case_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Regression case '{case_id}' not found.")
    return RegressionRunResponse(
        case_id=result.case_id,
        passed=result.passed,
        replayed_trace_id=result.replayed_trace_id,
        assertion_results=[
            {
                "assertion_id": r.assertion_id,
                "name": r.name,
                "passed": r.passed,
                "expected": r.expected,
                "actual": r.actual,
                "message": r.message,
            }
            for r in result.assertion_results
        ],
        duration_ms=round(result.duration_ms, 2),
        summary=result.summary,
    )


@router.get(
    "/regression/export",
    summary="Export Regression Cases for CI/CD",
)
def export_regression_cases(
    format: str = Query("yaml", pattern="^(yaml|json)$"),
    svc: RegressionService = Depends(get_regression_service),
) -> Response:
    """Exports active regression test cases as YAML or JSON CI dataset (spec Section 13)."""
    content = svc.export_ci_dataset(format=format)
    media_type = "application/x-yaml" if format == "yaml" else "application/json"
    return Response(content=content, media_type=media_type)
