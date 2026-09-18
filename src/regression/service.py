"""Regression Test Service (Milestone 8).

Implements the failure-to-regression conversion loop:
  Confirmed Incident -> Regression Case -> CI Dataset -> Continuous Release Protection.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import yaml

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.domain.incident import FailureCategory, Incident, IncidentStatus
from src.domain.regression import (
    AssertionOperator,
    AssertionResult,
    RegressionAssertion,
    RegressionCase,
    RegressionRunResult,
    RegressionStatus,
)
from src.domain.trace import SpanType, TraceData, TraceOutcome
from src.storage.database import DatabaseManager, get_database
from src.storage.repository import TraceRepository
from src.telemetry.instrumentation import TelemetryManager
from src.telemetry.redaction import hash_content


class RegressionService:
    """Manages the creation, execution, and CI/CD export of regression cases."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or get_database()

    def create_case_from_incident(
        self,
        incident_id: str,
        owner: str = "ai-platform",
        introduced_version: Optional[str] = None,
    ) -> Optional[RegressionCase]:
        """Converts an investigated failure incident into a regression test case."""
        session = self.db.get_session()
        repo = TraceRepository(session)

        incident = repo.get_incident(incident_id)
        if not incident:
            session.close()
            return None

        source_trace = repo.get_trace(incident.trace_id)
        if not source_trace:
            session.close()
            return None

        # Extract user query (prioritize unredacted trace metadata)
        query = ""
        if source_trace.metadata and source_trace.metadata.get("query"):
            candidate = str(source_trace.metadata["query"])
            if not candidate.startswith("hash:"):
                query = candidate

        if not query:
            for span in source_trace.spans:
                for key in ("agent.query", "retrieval.query", "query"):
                    val = span.attributes.get(key)
                    if val and isinstance(val, str) and not val.startswith("hash:"):
                        query = val
                        break
                if query:
                    break

        if not query:
            query = "What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?"

        case_id = f"reg_{uuid.uuid4().hex[:8]}"
        app_version = introduced_version or source_trace.application_version or "1.0.0"

        # Formulate deterministic assertions tailored to the failure category
        assertions: List[RegressionAssertion] = []
        expected_evidence: List[str] = []

        if incident.category == FailureCategory.RETRIEVAL_FAILURE:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="retrieval_candidate_count_positive",
                    target_metric="retrieval.candidate_count",
                    operator=AssertionOperator.GREATER_THAN_OR_EQUAL,
                    expected_value=1,
                    description="Retriever must return at least 1 relevant candidate document.",
                )
            )
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="no_empty_retrieval_event",
                    target_metric="has_retrieval_empty_event",
                    operator=AssertionOperator.IS_FALSE,
                    expected_value=False,
                    description="Retrieval span must not emit retrieval_empty event.",
                )
            )
            # Find evidence references from corpus
            expected_evidence = ["doc_fin_2025_q3"] if "ARR" in query or "financial" in query.lower() else ["enterprise_knowledge"]

        elif incident.category == FailureCategory.POLICY_FAILURE:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="policy_bypass_prevented",
                    target_metric="policy.is_bypassed",
                    operator=AssertionOperator.IS_FALSE,
                    expected_value=False,
                    description="Policy guardrail must never be bypassed.",
                )
            )

        elif incident.category == FailureCategory.GENERATION_FAILURE:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="citation_present",
                    target_metric="citation.verified",
                    operator=AssertionOperator.IS_TRUE,
                    expected_value=True,
                    description="Answer generation must contain verified grounding citations.",
                )
            )

        elif incident.category == FailureCategory.RETRY_FAILURE:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="retry_count_bounded",
                    target_metric="total_retries",
                    operator=AssertionOperator.LESS_THAN_OR_EQUAL,
                    expected_value=1,
                    description="Tool retry loop must not exceed retry limit.",
                )
            )

        elif incident.category == FailureCategory.CONTEXT_FAILURE:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="context_not_bloated",
                    target_metric="has_context_bloated_event",
                    operator=AssertionOperator.IS_FALSE,
                    expected_value=False,
                    description="Context chunk count must remain within baseline window.",
                )
            )

        else:
            assertions.append(
                RegressionAssertion(
                    assertion_id=f"asrt_{uuid.uuid4().hex[:6]}",
                    name="trace_successful",
                    target_metric="trace_outcome",
                    operator=AssertionOperator.EQUALS,
                    expected_value="success",
                    description="Trace execution outcome must be successful.",
                )
            )

        case = RegressionCase(
            id=case_id,
            incident_id=incident_id,
            source_trace_id=source_trace.trace_id,
            failure_type=incident.category.value,
            query=query,
            input_hash=hash_content(query),
            expected_evidence=expected_evidence,
            assertions=assertions,
            owner=owner,
            introduced_version=app_version,
            status=RegressionStatus.ACTIVE,
            metadata={"incident_severity": incident.severity.value},
        )

        repo.save_regression_case(case)
        session.close()
        return case

    def run_case(
        self, case_id: str, agent: Optional[InstrumentedResearchAgent] = None
    ) -> Optional[RegressionRunResult]:
        """Executes current agent against a regression case and evaluates all assertions."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        case = repo.get_regression_case(case_id)
        if not case:
            session.close()
            return None

        # Execute agent in clean telemetry sandbox
        telemetry = TelemetryManager(
            service_name=f"regression-{case.id}",
            export_to_file=False,
            export_to_db=False,
        )
        test_agent = agent or InstrumentedResearchAgent(telemetry_manager=telemetry)

        t0 = time.perf_counter()
        req = AgentRequest(query=case.query, failure_mode=FailureMode.HEALTHY)
        resp, trace = test_agent.run(req)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        assertion_results: List[AssertionResult] = []
        for asrt in case.assertions:
            actual = self._extract_metric_from_trace(trace, asrt.target_metric)
            passed = asrt.evaluate(actual)
            assertion_results.append(
                AssertionResult(
                    assertion_id=asrt.assertion_id,
                    name=asrt.name,
                    passed=passed,
                    expected=asrt.expected_value,
                    actual=actual,
                    message=f"Expected {asrt.target_metric} {asrt.operator.value} {asrt.expected_value}, got {actual}",
                )
            )

        all_passed = len(assertion_results) > 0 and all(r.passed for r in assertion_results)
        result = RegressionRunResult(
            case_id=case.id,
            passed=all_passed,
            replayed_trace_id=trace.trace_id if trace else None,
            assertion_results=assertion_results,
            duration_ms=duration_ms,
            summary=f"Regression test {'PASSED' if all_passed else 'FAILED'} ({sum(1 for r in assertion_results if r.passed)}/{len(assertion_results)} assertions satisfied)",
        )

        # Update case status if passed
        if all_passed:
            case.status = RegressionStatus.PASSED
            case.fixed_version = test_agent.app_version
        else:
            case.status = RegressionStatus.FAILED
        repo.save_regression_case(case)
        repo.save_regression_run(result)

        session.close()
        return result

    def run_all_cases(self) -> List[RegressionRunResult]:
        """Runs all active regression cases."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        cases, _ = repo.list_regression_cases(limit=100)
        session.close()

        results = []
        for c in cases:
            res = self.run_case(c.id)
            if res:
                results.append(res)
        return results

    def export_ci_dataset(self, format: str = "yaml") -> str:
        """Exports all active regression test cases in standard CI/CD format (spec Section 13)."""
        session = self.db.get_session()
        repo = TraceRepository(session)
        cases, _ = repo.list_regression_cases(limit=500)
        session.close()

        dataset = []
        for c in cases:
            dataset.append(
                {
                    "id": c.id,
                    "source_trace": c.source_trace_id,
                    "failure_type": c.failure_type,
                    "query": c.query,
                    "expected_evidence": c.expected_evidence,
                    "assertions": [f"{a.name}: {a.target_metric} {a.operator.value} {a.expected_value}" for a in c.assertions],
                    "owner": c.owner,
                    "introduced_version": c.introduced_version,
                    "fixed_version": c.fixed_version,
                    "status": c.status.value,
                }
            )

        if format.lower() == "json":
            return json.dumps({"regression_cases": dataset}, indent=2)
        return yaml.dump(dataset, sort_keys=False)

    def _extract_metric_from_trace(self, trace: Optional[TraceData], metric_key: str) -> Any:
        """Pulls targeted operational metrics and attributes from the trace hierarchy."""
        if not trace:
            return None

        if metric_key == "trace_outcome":
            return trace.outcome.value
        elif metric_key == "total_tokens":
            return trace.total_tokens()
        elif metric_key == "duration_ms":
            return trace.duration_ms

        for span in trace.spans:
            if metric_key == "retrieval.candidate_count" and span.span_type == SpanType.RETRIEVAL:
                return span.attributes.get("retrieval.candidate_count", 0)
            elif metric_key == "has_retrieval_empty_event":
                if any(e.name == "retrieval_empty" for e in span.events):
                    return True
            elif metric_key == "has_context_bloated_event":
                if any(e.name == "context_bloated" for e in span.events):
                    return True
            elif metric_key == "policy.is_bypassed" and span.span_type == SpanType.POLICY:
                return span.attributes.get("policy.is_bypassed", False)
            elif metric_key == "citation.verified" and span.span_type == SpanType.CITATION:
                invalid_cnt = span.attributes.get("citation.invalid_count", 0)
                return invalid_cnt == 0
            elif metric_key == "total_retries":
                return span.attributes.get("tool.retry_count", 0)

        if metric_key in ("has_retrieval_empty_event", "has_context_bloated_event", "policy.is_bypassed"):
            return False

        return None
