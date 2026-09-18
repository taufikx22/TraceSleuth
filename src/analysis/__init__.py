"""TraceSleuth Analysis Package.

Exports failure taxonomy classifier, root-cause engine, diagnosis rules,
and incident similarity fingerprinting.
"""

from src.analysis.engine import CausalNode, HypothesisRanking, RootCauseEngine
from src.analysis.model_assisted import (
    EvidenceCitation,
    ModelAssistedForensicAnalyst,
    ModelHypothesisResult,
)
from src.analysis.rules import DEFAULT_FORENSIC_RULES, ForensicRule, ForensicRuleResult
from src.analysis.similarity import FailureSignature, compute_incident_similarity
from src.analysis.taxonomy import FailureTaxonomyClassifier

__all__ = [
    "FailureTaxonomyClassifier",
    "RootCauseEngine",
    "CausalNode",
    "HypothesisRanking",
    "ForensicRule",
    "ForensicRuleResult",
    "DEFAULT_FORENSIC_RULES",
    "FailureSignature",
    "compute_incident_similarity",
    "EvidenceCitation",
    "ModelAssistedForensicAnalyst",
    "ModelHypothesisResult",
]
