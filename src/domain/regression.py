"""Domain models for Regression Cases and Evaluation (Milestone 8).

Defines the structure for turning confirmed production failure incidents
into permanent regression tests and CI/CD evaluation datasets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RegressionStatus(str, Enum):
    """Lifecycle status of a regression test case."""

    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    RETIRED = "retired"


class AssertionOperator(str, Enum):
    """Operator used to evaluate regression assertions."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class RegressionAssertion(BaseModel):
    """A test assertion verifying that a previous failure mode is prevented."""

    assertion_id: str
    name: str
    target_metric: str  # e.g., "unsupported_claim_count", "retrieval.candidate_count", "policy.denied"
    operator: AssertionOperator = AssertionOperator.EQUALS
    expected_value: Any
    description: str = ""

    def evaluate(self, actual_value: Any) -> bool:
        """Evaluates whether the assertion holds for the actual value."""
        if self.operator == AssertionOperator.EQUALS:
            return actual_value == self.expected_value
        elif self.operator == AssertionOperator.NOT_EQUALS:
            return actual_value != self.expected_value
        elif self.operator == AssertionOperator.GREATER_THAN_OR_EQUAL:
            return float(actual_value) >= float(self.expected_value)
        elif self.operator == AssertionOperator.LESS_THAN_OR_EQUAL:
            return float(actual_value) <= float(self.expected_value)
        elif self.operator == AssertionOperator.CONTAINS:
            return self.expected_value in actual_value
        elif self.operator == AssertionOperator.NOT_CONTAINS:
            return self.expected_value not in actual_value
        elif self.operator == AssertionOperator.IS_TRUE:
            return bool(actual_value) is True
        elif self.operator == AssertionOperator.IS_FALSE:
            return bool(actual_value) is False
        return False


class RegressionCase(BaseModel):
    """A persistent regression test case created from an investigated failure incident."""

    id: str  # e.g. "reg_0172"
    incident_id: str
    source_trace_id: str
    failure_type: str  # e.g. "retrieval_failure", "policy_failure"
    query: str
    input_hash: str
    expected_evidence: List[str] = Field(default_factory=list)  # e.g. ["source_42", "doc_fin_2025_q3"]
    assertions: List[RegressionAssertion] = Field(default_factory=list)
    owner: str = "ai-platform"
    introduced_version: str = "1.0.0"
    fixed_version: Optional[str] = None
    status: RegressionStatus = RegressionStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssertionResult(BaseModel):
    """Result of evaluating a single regression assertion."""

    assertion_id: str
    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""


class RegressionRunResult(BaseModel):
    """Overall outcome of executing a regression case against current agent."""

    case_id: str
    passed: bool
    replayed_trace_id: Optional[str] = None
    assertion_results: List[AssertionResult] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    summary: str = ""
