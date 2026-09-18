"""TraceSleuth Analysis — Similar Incident Search & Fingerprinting.

Implements failure signature generation and similarity search described
in spec Sections 11 and 31 ("Have we seen this failure before?").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from src.domain.incident import Incident
from src.domain.trace import TraceData


@dataclass
class FailureSignature:
    """Deterministic fingerprint representing an incident's failure mode."""

    signature_str: str
    signature_hash: str
    category: str
    key_evidence_types: List[str]
    failed_operation_names: List[str]

    @classmethod
    def generate(cls, incident: Incident, trace: Optional[TraceData] = None) -> FailureSignature:
        """Generates a failure signature from an incident and its associated trace."""
        category_str = incident.category.value
        ev_types = sorted({ev.type for ev in incident.evidence})
        
        failed_ops = []
        if trace and incident.first_suspicious_span_id:
            suspicious = trace.get_span_by_id(incident.first_suspicious_span_id)
            if suspicious:
                failed_ops.append(suspicious.name)

        # Build human-readable signature string, e.g.:
        # "retrieval_failure::retrieval_empty::retrieval.vector_search"
        parts = [category_str]
        if ev_types:
            parts.append("+".join(ev_types))
        if failed_ops:
            parts.append("+".join(failed_ops))

        signature_str = "::".join(parts)
        signature_hash = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()[:16]

        return cls(
            signature_str=signature_str,
            signature_hash=signature_hash,
            category=category_str,
            key_evidence_types=ev_types,
            failed_operation_names=failed_ops,
        )


def compute_incident_similarity(inc1: Incident, inc2: Incident) -> float:
    """Computes a Jaccard/weighted similarity score [0.0, 1.0] between two incidents."""
    if inc1.category != inc2.category:
        return 0.0

    ev1_types = {e.type for e in inc1.evidence}
    ev2_types = {e.type for e in inc2.evidence}

    if not ev1_types and not ev2_types:
        return 1.0 if inc1.severity == inc2.severity else 0.8

    intersection = len(ev1_types.intersection(ev2_types))
    union = len(ev1_types.union(ev2_types))
    jaccard = intersection / union if union > 0 else 0.0

    # Boost if severity matches
    severity_boost = 0.2 if inc1.severity == inc2.severity else 0.0
    return min(1.0, 0.8 * jaccard + severity_boost)
