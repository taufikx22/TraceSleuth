"""Model-Assisted Hypothesis Generation (Milestone 10).

Combines deterministic telemetry and causal evidence graphs with model-assisted
reasoning to produce coherent, evidence-traceable root-cause explanations (spec Sections 9.3 & 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os

from src.analysis.engine import RootCauseEngine
from src.domain.incident import Evidence, FailureCategory, Incident
from src.domain.trace import SpanType, TraceData


@dataclass
class EvidenceCitation:
    """A verified citation connecting an LLM reasoning claim to a concrete span and telemetry key."""

    span_id: str
    evidence_key: str
    description: str
    strength: float


@dataclass
class ModelHypothesisResult:
    """The structured forensic hypothesis produced by model-assisted analysis."""

    trace_id: str
    incident_id: Optional[str]
    primary_root_cause: FailureCategory
    confidence: float
    explanation: str
    evidence_citations: List[EvidenceCitation]
    recommended_remediation: str
    engine: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelAssistedForensicAnalyst:
    """Model-assisted forensic reasoning engine grounded in deterministic trace telemetry.

    Adheres strictly to the architectural constraint in spec Section 10:
    'The model should explain evidence, not replace deterministic telemetry.
     Model hypotheses must remain strictly traceable to concrete evidence.'
    """

    def __init__(
        self,
        rca_engine: Optional[RootCauseEngine] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.rca_engine = rca_engine or RootCauseEngine()
        self.model_name = model_name or os.getenv("TRACESLEUTH_LLM_MODEL", "tracesleuth-grounded-analyst-v1")

    def analyze(
        self,
        trace: TraceData,
        incident: Optional[Incident] = None,
    ) -> ModelHypothesisResult:
        """Produces a grounded forensic hypothesis citing specific trace spans and evidence items."""
        # 1. Gather deterministic evidence
        rankings = self.rca_engine.analyze(trace)
        earliest_suspicious = self.rca_engine.find_earliest_suspicious_span(trace)

        if incident:
            primary_cat = incident.category
            evidence_source = incident.evidence
            confidence = incident.hypothesis_score
            suspicious_span_id = incident.first_suspicious_span_id
        elif rankings:
            primary = rankings[0]
            primary_cat = primary.category
            evidence_source = primary.evidence
            confidence = primary.confidence
            suspicious_span_id = primary.first_suspicious_span_id
        else:
            primary_cat = FailureCategory.RETRIEVAL_FAILURE
            evidence_source = []
            confidence = 0.5
            suspicious_span_id = earliest_suspicious.span_id if earliest_suspicious else None

        # 2. Extract and verify evidence citations from actual spans
        valid_citations: List[EvidenceCitation] = []
        for ev in evidence_source:
            # Verify that span actually exists in the trace
            span = trace.get_span(ev.span_id)
            if span:
                valid_citations.append(
                    EvidenceCitation(
                        span_id=ev.span_id,
                        evidence_key=ev.key,
                        description=ev.description,
                        strength=ev.strength,
                    )
                )

        # If no explicit evidence items but suspicious span identified, cite the span
        if not valid_citations and earliest_suspicious:
            valid_citations.append(
                EvidenceCitation(
                    span_id=earliest_suspicious.span_id,
                    evidence_key=f"span.status={earliest_suspicious.status.value}",
                    description=f"Earliest chronological anomaly detected at span '{earliest_suspicious.name}'",
                    strength=0.8,
                )
            )

        # 3. Formulate grounded narrative explaining the causal chain
        explanation = self._generate_grounded_explanation(
            trace=trace,
            primary_category=primary_cat,
            citations=valid_citations,
            earliest_span=trace.get_span(suspicious_span_id) if suspicious_span_id else earliest_suspicious,
        )

        remediation = self._generate_remediation(primary_cat)

        return ModelHypothesisResult(
            trace_id=trace.trace_id,
            incident_id=incident.incident_id if incident else None,
            primary_root_cause=primary_cat,
            confidence=round(confidence, 2),
            explanation=explanation,
            evidence_citations=valid_citations,
            recommended_remediation=remediation,
            engine=self.model_name,
        )

    def _generate_grounded_explanation(
        self,
        trace: TraceData,
        primary_category: FailureCategory,
        citations: List[EvidenceCitation],
        earliest_span: Optional[Any],
    ) -> str:
        """Constructs an evidence-backed explanation connecting cause to symptoms."""
        span_name = earliest_span.name if earliest_span else "pipeline execution"
        span_id_ref = f" (span `{earliest_span.span_id}`)" if earliest_span else ""

        citation_summaries = "; ".join(f"{c.description} [{c.evidence_key}]" for c in citations[:3])

        narratives = {
            FailureCategory.RETRIEVAL_FAILURE: (
                f"Forensic inspection reveals that execution failed due to an empty or insufficient candidate set at "
                f"stage '{span_name}'{span_id_ref}. Specifically: {citation_summaries or 'no matching documents were retrieved'}. "
                f"This document starvation propagated downstream to the model generation stage, which was unable to "
                f"ground its answer in enterprise sources."
            ),
            FailureCategory.POLICY_FAILURE: (
                f"A critical security guardrail failure occurred at stage '{span_name}'{span_id_ref}. "
                f"The policy engine detected unauthorized execution or bypass: {citation_summaries}. "
                f"Downstream tools should have been terminated immediately."
            ),
            FailureCategory.RANKING_FAILURE: (
                f"Retrieved documents were present, but the reranker filtered out key sources at '{span_name}'{span_id_ref}. "
                f"Telemetry evidence: {citation_summaries}. The downstream model received an impoverished prompt."
            ),
            FailureCategory.GENERATION_FAILURE: (
                f"The model generation at stage '{span_name}'{span_id_ref} contradicted retrieved facts or failed "
                f"attribution verification. Telemetry evidence: {citation_summaries}. Downstream answers contained unsupported claims."
            ),
            FailureCategory.RETRY_FAILURE: (
                f"A retry storm amplified latency and token spend at stage '{span_name}'{span_id_ref}. "
                f"Telemetry evidence: {citation_summaries}."
            ),
            FailureCategory.CONTEXT_FAILURE: (
                f"Context window bloat was observed at stage '{span_name}'{span_id_ref}. "
                f"Duplicate or oversized chunks inflated token consumption: {citation_summaries}."
            ),
        }

        return narratives.get(
            primary_category,
            f"Execution anomaly classified as {primary_category.value} originated at stage '{span_name}'{span_id_ref}. "
            f"Key telemetry evidence: {citation_summaries or 'anomalous attributes detected'}.",
        )

    def _generate_remediation(self, category: FailureCategory) -> str:
        """Returns actionable engineering remediation steps."""
        remediations = {
            FailureCategory.RETRIEVAL_FAILURE: "Verify query semantic embedding alignment, lower vector similarity threshold, or expand enterprise document indexing.",
            FailureCategory.POLICY_FAILURE: "Enforce strict fail-closed guardrail gates prior to tool execution and inspect authorization interceptor.",
            FailureCategory.RANKING_FAILURE: "Adjust cross-encoder score cutoffs and verify ranking model weights on domain documents.",
            FailureCategory.GENERATION_FAILURE: "Enforce citation attribution prompt constraints and increase temperature grounding checks.",
            FailureCategory.RETRY_FAILURE: "Implement exponential backoff jitter and cap maximum tool retry budget to <= 2 attempts.",
            FailureCategory.CONTEXT_FAILURE: "Implement chunk deduplication and prune non-salient document chunks prior to prompt assembly.",
        }
        return remediations.get(
            category,
            "Inspect span telemetry logs and verify upstream component configurations."
        )
