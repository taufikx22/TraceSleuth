"""Tests for Incident Fingerprinting and Similarity Matching (spec Section 11 & 31)."""

import pytest

from src.analysis.similarity import FailureSignature, compute_incident_similarity
from src.domain.incident import Evidence, FailureCategory, Incident, IncidentSeverity, IncidentStatus


def _make_incident(
    inc_id: str,
    category: FailureCategory,
    evidence_types: list[str],
    severity: IncidentSeverity = IncidentSeverity.P1,
) -> Incident:
    evidence = [
        Evidence(
            evidence_id=f"ev_{i}",
            incident_id=inc_id,
            type=et,
            key=f"key_{i}",
            value_hash=f"hash_{i}",
            strength=0.9,
        )
        for i, et in enumerate(evidence_types)
    ]
    return Incident(
        incident_id=inc_id,
        trace_id=f"tr_{inc_id}",
        category=category,
        severity=severity,
        summary=f"Incident {inc_id}",
        evidence=evidence,
        status=IncidentStatus.OPEN,
    )


class TestIncidentSimilarity:

    def test_signature_generation(self):
        inc = _make_incident("inc_1", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty", "missing_doc"])
        sig = FailureSignature.generate(inc)

        assert "retrieval_failure" in sig.signature_str
        assert "retrieval_empty" in sig.signature_str
        assert "missing_doc" in sig.signature_str
        assert len(sig.signature_hash) == 16

    def test_identical_signature_high_similarity(self):
        inc1 = _make_incident("inc_1", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty"])
        inc2 = _make_incident("inc_2", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty"])

        sim = compute_incident_similarity(inc1, inc2)
        assert sim >= 0.95

    def test_different_categories_zero_similarity(self):
        inc1 = _make_incident("inc_1", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty"])
        inc2 = _make_incident("inc_2", FailureCategory.POLICY_FAILURE, ["policy_bypass"])

        sim = compute_incident_similarity(inc1, inc2)
        assert sim == 0.0

    def test_partial_evidence_overlap(self):
        inc1 = _make_incident(
            "inc_1", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty", "low_confidence", "timeout"]
        )
        inc2 = _make_incident(
            "inc_2", FailureCategory.RETRIEVAL_FAILURE, ["retrieval_empty", "low_confidence", "unrelated"]
        )

        sim = compute_incident_similarity(inc1, inc2)
        assert 0.4 <= sim <= 0.85
