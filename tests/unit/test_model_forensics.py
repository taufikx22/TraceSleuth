"""Tests for Model-Assisted Hypothesis Generation (Milestone 10).

Verifies that model-assisted reasoning generates grounded root-cause hypotheses
with accountable citations back to concrete span telemetry (spec Sections 9.3 & 10).
"""

from __future__ import annotations

import pytest

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.analysis.model_assisted import ModelAssistedForensicAnalyst
from src.domain.incident import FailureCategory
from src.ingestion.service import IngestionService
from src.storage.database import DatabaseManager, reset_database
from src.storage.repository import TraceRepository


@pytest.fixture(autouse=True)
def test_db():
    reset_database()
    import src.storage.database as db_module
    db_module._default_db = DatabaseManager(database_url="sqlite:///:memory:")
    db_module._default_db.init_db()
    yield db_module._default_db
    reset_database()


class TestModelAssistedForensics:
    """Unit tests verifying evidence-traceable model-assisted analysis."""

    def test_model_assisted_hypothesis_grounded_in_evidence(self, test_db):
        """Milestone 10 exit criterion: Model hypotheses remain traceable to evidence."""
        agent = InstrumentedResearchAgent()
        req = AgentRequest(
            query="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
            failure_mode=FailureMode.RETRIEVAL_MISS,
        )
        _, trace = agent.run(req)

        ingestion = IngestionService(test_db)
        incidents = ingestion.ingest_trace(trace, auto_classify=True)
        assert len(incidents) > 0

        incident = incidents[0]
        analyst = ModelAssistedForensicAnalyst()
        result = analyst.analyze(trace, incident=incident)

        assert result is not None
        assert result.trace_id == trace.trace_id
        assert result.incident_id == incident.incident_id
        assert result.primary_root_cause == FailureCategory.RETRIEVAL_FAILURE
        assert result.confidence >= 0.8
        assert "retrieval" in result.explanation.lower()
        assert len(result.recommended_remediation) > 0

        # CRITICAL TEST: Evidence citations must be traceable to real spans in the trace
        assert len(result.evidence_citations) > 0
        for citation in result.evidence_citations:
            span = trace.get_span(citation.span_id)
            assert span is not None, f"Cited span_id {citation.span_id} not found in trace!"
            assert len(citation.evidence_key) > 0
            assert citation.strength > 0.0

    def test_model_assisted_hypothesis_from_raw_trace(self):
        """Model analyst works directly on raw trace data when no incident has been saved yet."""
        agent = InstrumentedResearchAgent()
        req = AgentRequest(
            query="What are the security policies?",
            failure_mode=FailureMode.POLICY_BYPASS,
        )
        _, trace = agent.run(req)

        analyst = ModelAssistedForensicAnalyst()
        result = analyst.analyze(trace)

        assert result is not None
        assert result.primary_root_cause == FailureCategory.POLICY_FAILURE
        assert result.confidence >= 0.9
        assert "guardrail" in result.explanation.lower() or "policy" in result.explanation.lower()
        assert len(result.evidence_citations) > 0

        # Citations must link to a valid span
        span = trace.get_span(result.evidence_citations[0].span_id)
        assert span is not None
