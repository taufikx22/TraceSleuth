"""TraceSleuth Analysis — Failure Taxonomy Classifier.

Deterministic rule-based classification engine that examines trace spans
and events to identify and categorize failures into the 13-category taxonomy.
Each rule returns evidence, a hypothesis score, and severity assignment.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from src.domain.incident import (
    Evidence,
    FailureCategory,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData, TraceOutcome


class FailureTaxonomyClassifier:
    """Classifies trace failures using deterministic rules over span data.

    Examines span attributes, events, and status to produce ranked
    Incident objects with linked evidence. Each detection rule documents:
      - Observed pattern
      - Hypothesis
      - Supporting evidence
      - Confidence scoring rationale
    """

    def classify(self, trace: TraceData) -> List[Incident]:
        """Analyze a trace and return zero or more classified incidents."""
        incidents: List[Incident] = []

        # Run all 13 category detectors
        detectors = [
            self._detect_retrieval_failure,
            self._detect_ranking_failure,
            self._detect_context_failure,
            self._detect_generation_failure,
            self._detect_tool_selection_failure,
            self._detect_tool_argument_failure,
            self._detect_policy_failure,
            self._detect_state_failure,
            self._detect_routing_failure,
            self._detect_retry_failure,
            self._detect_timeout_failure,
            self._detect_data_quality_failure,
            self._detect_evaluation_failure,
        ]

        for detector in detectors:
            incident = detector(trace)
            if incident:
                incidents.append(incident)

        # Sort by hypothesis score descending (highest confidence first)
        incidents.sort(key=lambda i: i.hypothesis_score, reverse=True)
        return incidents

    # ── Individual Failure Detectors ─────────────────────────

    def _detect_retrieval_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect when relevant sources were not retrieved.

        Pattern: retrieval span returns 0 candidates OR retrieval_empty event.
        """
        retrieval_spans = trace.get_spans_by_type(SpanType.RETRIEVAL)
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        for span in retrieval_spans:
            candidate_count = span.attributes.get("retrieval.candidate_count", -1)
            has_empty_event = any(e.name == "retrieval_empty" for e in span.events)

            if candidate_count == 0 or has_empty_event:
                if not first_span_id:
                    first_span_id = span.span_id
                score = max(score, 0.9)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="retrieval_empty",
                        key="retrieval.candidate_count",
                        value_hash=str(candidate_count),
                        description=f"Retrieval returned {candidate_count} candidates",
                        strength=0.9,
                    )
                )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.RETRIEVAL_FAILURE,
            severity=IncidentSeverity.P1,
            summary="Retrieval returned no or very few relevant documents",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_ranking_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect when relevant sources were retrieved but ranked too low.

        Pattern: reranker reduces candidate count significantly or
        reranker output is much smaller than input.
        """
        reranker_spans = trace.get_spans_by_type(SpanType.RERANKER)
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        for span in reranker_spans:
            input_count = span.attributes.get("reranker.input_count", 0)
            output_count = span.attributes.get("reranker.output_count", 0)

            if isinstance(input_count, (int, float)) and isinstance(output_count, (int, float)):
                if input_count > 0 and output_count == 0:
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.85)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type="ranking_eliminated_all",
                            key="reranker.output_count",
                            value_hash=f"{input_count}->{output_count}",
                            description=f"Reranker eliminated all {input_count} candidates",
                            strength=0.85,
                        )
                    )
                elif input_count > 3 and output_count <= 1:
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.6)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type="ranking_aggressive_filter",
                            key="reranker.output_count",
                            value_hash=f"{input_count}->{output_count}",
                            description=f"Reranker aggressively filtered {input_count} to {output_count}",
                            strength=0.6,
                        )
                    )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.RANKING_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Reranker aggressively filtered or eliminated candidates",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_context_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect context window issues (bloat, truncation).

        Pattern: context_bloated event, high token counts, or context_truncated event.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        # Check for context-related events across all spans
        for span in trace.spans:
            for event in span.events:
                if event.name in ("context_bloated", "context_truncated"):
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.8)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type=event.name,
                            key=f"event.{event.name}",
                            value_hash=str(event.attributes),
                            description=f"Context anomaly: {event.name} detected",
                            strength=0.8,
                        )
                    )

        # Check for unusually high token usage
        total_tokens = trace.total_tokens()
        if total_tokens > 10000:
            model_spans = trace.get_spans_by_type(SpanType.MODEL)
            for span in model_spans:
                input_tokens = span.attributes.get("gen_ai.usage.input_tokens", 0)
                if isinstance(input_tokens, (int, float)) and input_tokens > 8000:
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.7)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type="token_anomaly",
                            key="gen_ai.usage.input_tokens",
                            value_hash=str(input_tokens),
                            description=f"Unusually high input tokens: {input_tokens}",
                            strength=0.7,
                        )
                    )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.CONTEXT_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Context anomaly detected (bloat, truncation, or excessive tokens)",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_generation_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect when the model contradicts available evidence.

        Pattern: citation_invalid/citation_missing events, or model span has errors.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        # Check citation spans for invalid/missing citations
        citation_spans = trace.get_spans_by_type(SpanType.CITATION)
        for span in citation_spans:
            invalid_count = span.attributes.get("citation.invalid_count", 0)
            if isinstance(invalid_count, (int, float)) and invalid_count > 0:
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.75)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="citation_invalid",
                        key="citation.invalid_count",
                        value_hash=str(invalid_count),
                        description=f"{invalid_count} invalid citation(s) detected",
                        strength=0.75,
                    )
                )

            for event in span.events:
                if event.name in ("citation_missing", "citation_invalid"):
                    score = max(score, 0.8)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type=event.name,
                            key=f"event.{event.name}",
                            value_hash=str(event.attributes),
                            description=f"Generation evidence issue: {event.name}",
                            strength=0.8,
                        )
                    )

        # Check model spans for errors
        model_spans = trace.get_spans_by_type(SpanType.MODEL)
        for span in model_spans:
            if span.status == SpanStatus.ERROR:
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.85)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="model_error",
                        key="span.status",
                        value_hash="ERROR",
                        description=f"Model span failed: {span.error_message or 'unknown'}",
                        strength=0.85,
                    )
                )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.GENERATION_FAILURE,
            severity=IncidentSeverity.P1,
            summary="Model generation contradicts evidence or produced invalid citations",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_tool_selection_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect wrong tool chosen for the task.

        Pattern: Tool span with status=error where the tool name doesn't match
        the planned step, or no tool calls when tools were required.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        planner_spans = trace.get_spans_by_type(SpanType.PLANNER)
        tool_spans = trace.get_spans_by_type(SpanType.TOOL)

        # Check if planner suggested tools but none were executed
        for span in planner_spans:
            suggested = span.attributes.get("planner.suggested_tools", "[]")
            step_count = span.attributes.get("planner.step_count", 0)
            if isinstance(step_count, (int, float)) and step_count > 0 and len(tool_spans) == 0:
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.7)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="tools_planned_not_executed",
                        key="planner.step_count",
                        value_hash=str(step_count),
                        description=f"Planner suggested {step_count} steps but no tools were executed",
                        strength=0.7,
                    )
                )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.TOOL_SELECTION_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Tool selection mismatch: planned tools were not executed",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_tool_argument_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect correct tool chosen but with bad arguments.

        Pattern: Tool span has status=error with error_message suggesting argument issues.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        tool_spans = trace.get_spans_by_type(SpanType.TOOL)
        for span in tool_spans:
            if span.status == SpanStatus.ERROR:
                error_msg = (span.error_message or "").lower()
                result_status = span.attributes.get("tool.result_status", "")
                # Distinguish argument errors from connection/timeout errors
                if any(kw in error_msg for kw in ("argument", "invalid", "parse", "type", "value")):
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.75)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type="tool_argument_error",
                            key="tool.error_message",
                            value_hash=error_msg[:64],
                            description=f"Tool argument error: {span.error_message}",
                            strength=0.75,
                        )
                    )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.TOOL_ARGUMENT_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Correct tool selected but called with invalid arguments",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_policy_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect policy bypass or violation.

        Pattern: policy decision=deny AND execution proceeded (policy_bypass event),
        OR policy violations exist.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        policy_spans = trace.get_spans_by_type(SpanType.POLICY)
        for span in policy_spans:
            decision = span.attributes.get("policy.decision", "allow")
            is_bypassed = span.attributes.get("policy.is_bypassed", False)
            violations = span.attributes.get("policy.violations", "[]")

            has_bypass_event = any(e.name == "policy_bypass" for e in span.events)

            if decision == "deny" and (is_bypassed or has_bypass_event):
                # Critical: execution continued despite policy denial
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.95)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="policy_bypass",
                        key="policy.is_bypassed",
                        value_hash=str(is_bypassed),
                        description="Execution proceeded despite policy denial — CRITICAL",
                        strength=0.95,
                    )
                )
            elif decision == "deny":
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.6)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="policy_denial",
                        key="policy.decision",
                        value_hash="deny",
                        description=f"Policy denied request: {violations}",
                        strength=0.6,
                    )
                )

        if not evidence_list:
            return None

        # Policy bypass is always P0 (critical)
        severity = IncidentSeverity.P0 if score >= 0.9 else IncidentSeverity.P1

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.POLICY_FAILURE,
            severity=severity,
            summary="Policy violation or bypass detected",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_state_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect incorrect memory/session state.

        Pattern: state_corruption or memory_error events, or agent_run span
        with state-related errors.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        for span in trace.spans:
            for event in span.events:
                if event.name in ("state_corruption", "memory_error", "session_invalid"):
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.75)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type=event.name,
                            key=f"event.{event.name}",
                            value_hash=str(event.attributes),
                            description=f"State anomaly detected: {event.name}",
                            strength=0.75,
                        )
                    )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.STATE_FAILURE,
            severity=IncidentSeverity.P1,
            summary="Session or memory state corruption detected",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_routing_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect wrong model/provider routing.

        Pattern: router span with errors, or model span attributes
        indicate unexpected model/provider.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        router_spans = trace.get_spans_by_type(SpanType.ROUTER)
        for span in router_spans:
            if span.status == SpanStatus.ERROR:
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.8)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="routing_error",
                        key="span.status",
                        value_hash="ERROR",
                        description=f"Router span failed: {span.error_message or 'unknown'}",
                        strength=0.8,
                    )
                )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.ROUTING_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Model/provider routing failure detected",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_retry_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect retry amplification storms.

        Pattern: Multiple retry events, high retry_count, or latency spikes from retries.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0
        total_retries = 0

        tool_spans = trace.get_spans_by_type(SpanType.TOOL)
        for span in tool_spans:
            retry_count = span.attributes.get("tool.retry_count", 0)
            if isinstance(retry_count, (int, float)) and retry_count > 0:
                total_retries += int(retry_count)

            retry_events = [e for e in span.events if e.name == "retry"]
            if retry_events:
                first_span_id = first_span_id or span.span_id

        # Count tool spans that are retries (retry_count > 0)
        retry_tool_spans = [
            s for s in tool_spans
            if isinstance(s.attributes.get("tool.retry_count", 0), (int, float))
            and s.attributes.get("tool.retry_count", 0) > 0
        ]

        # Multiple retry spans or high total retry count
        if len(retry_tool_spans) >= 2 or total_retries >= 3:
            score = max(score, 0.8)
            evidence_list.append(
                Evidence(
                    span_id=first_span_id or (tool_spans[0].span_id if tool_spans else None),
                    type="retry_storm",
                    key="tool.retry_count",
                    value_hash=str(total_retries),
                    description=f"Retry amplification: {total_retries} total retries across {len(retry_tool_spans)} spans",
                    strength=0.8,
                )
            )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.RETRY_FAILURE,
            severity=IncidentSeverity.P2,
            summary=f"Retry amplification detected: {total_retries} retries",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_timeout_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect timeout-related failures.

        Pattern: Span with timeout status, or extremely high latency (>5000ms).
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        for span in trace.spans:
            result_status = span.attributes.get("tool.result_status", "")
            if result_status == "timeout":
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.9)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="tool_timeout",
                        key="tool.result_status",
                        value_hash="timeout",
                        description=f"Tool timed out: {span.name} ({span.duration_ms}ms)",
                        strength=0.9,
                    )
                )
            elif span.duration_ms > 5000 and span.span_type != SpanType.AGENT_RUN:
                first_span_id = first_span_id or span.span_id
                score = max(score, 0.6)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="latency_anomaly",
                        key="span.duration_ms",
                        value_hash=str(span.duration_ms),
                        description=f"Excessive latency: {span.name} took {span.duration_ms}ms",
                        strength=0.6,
                    )
                )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.TIMEOUT_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Timeout or excessive latency detected",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_data_quality_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect upstream data quality issues.

        Pattern: data_quality_error events, or retrieval returning stale/malformed data.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        for span in trace.spans:
            for event in span.events:
                if event.name in ("data_quality_error", "stale_data", "malformed_record"):
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.7)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type=event.name,
                            key=f"event.{event.name}",
                            value_hash=str(event.attributes),
                            description=f"Data quality issue: {event.name}",
                            strength=0.7,
                        )
                    )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.DATA_QUALITY_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Upstream data quality issue detected",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )

    def _detect_evaluation_failure(self, trace: TraceData) -> Optional[Incident]:
        """Detect when output was incorrectly classified as acceptable.

        Pattern: Trace outcome is success, but citation validation shows issues,
        OR answer looks plausible but has missing/invalid sources.
        This catches what a superficial output check misses.
        """
        evidence_list: List[Evidence] = []
        first_span_id: Optional[str] = None
        score = 0.0

        # Check if trace was marked success but citations reveal problems
        if trace.outcome == TraceOutcome.SUCCESS:
            citation_spans = trace.get_spans_by_type(SpanType.CITATION)
            for span in citation_spans:
                invalid_count = span.attributes.get("citation.invalid_count", 0)
                total_count = span.attributes.get("citation.total_count", 0)

                if isinstance(invalid_count, (int, float)) and invalid_count > 0:
                    first_span_id = first_span_id or span.span_id
                    score = max(score, 0.7)
                    evidence_list.append(
                        Evidence(
                            span_id=span.span_id,
                            type="undetected_citation_failure",
                            key="citation.invalid_count",
                            value_hash=str(invalid_count),
                            description=f"Trace marked success but {invalid_count}/{total_count} citations are invalid",
                            strength=0.7,
                        )
                    )

                # No citations at all when documents were available
                if isinstance(total_count, (int, float)) and total_count == 0:
                    retrieval_spans = trace.get_spans_by_type(SpanType.RETRIEVAL)
                    has_docs = any(
                        isinstance(s.attributes.get("retrieval.candidate_count", 0), (int, float))
                        and s.attributes.get("retrieval.candidate_count", 0) > 0
                        for s in retrieval_spans
                    )
                    if has_docs:
                        first_span_id = first_span_id or span.span_id
                        score = max(score, 0.65)
                        evidence_list.append(
                            Evidence(
                                span_id=span.span_id,
                                type="missing_citations",
                                key="citation.total_count",
                                value_hash="0",
                                description="Documents retrieved but no citations produced",
                                strength=0.65,
                            )
                        )

        if not evidence_list:
            return None

        return Incident(
            trace_id=trace.trace_id,
            category=FailureCategory.EVALUATION_FAILURE,
            severity=IncidentSeverity.P2,
            summary="Output incorrectly classified as acceptable despite evidence issues",
            first_suspicious_span_id=first_span_id,
            hypothesis_score=round(score, 2),
            evidence=evidence_list,
        )
