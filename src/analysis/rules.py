"""TraceSleuth Analysis — Forensic Diagnosis Rules.

Implements 8+ deterministic root-cause rules operating over span telemetry,
GenAI attributes, and causal events.
Each rule evaluates evidence, calculates a confidence score, pinpoints
the first suspicious span, and suggests remediation actions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.incident import Evidence, FailureCategory, IncidentSeverity
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData


@dataclass
class ForensicRuleResult:
    """Result of evaluating a single forensic rule against a trace."""

    rule_name: str
    category: FailureCategory
    matched: bool
    confidence: float = 0.0
    severity: IncidentSeverity = IncidentSeverity.P2
    first_suspicious_span_id: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    explanation: str = ""
    remediation: str = ""


class ForensicRule:
    """Base class for all deterministic forensic diagnosis rules."""

    name: str = "base_rule"
    category: FailureCategory = FailureCategory.RETRIEVAL_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        raise NotImplementedError


class RetrievalMissRule(ForensicRule):
    """Rule 1: Detects when retrieval failed to return relevant sources.

    Pattern:
      - retrieval span candidate count is 0, or
      - retrieval_empty event emitted, or
      - candidate count is <= 1 with low relevance.
    """

    name = "retrieval_miss_rule"
    category = FailureCategory.RETRIEVAL_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        retrieval_spans = trace.get_spans_by_type(SpanType.RETRIEVAL)
        if not retrieval_spans:
            return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)

        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None
        max_score = 0.0

        for span in retrieval_spans:
            candidate_count = span.attributes.get("retrieval.candidate_count", -1)
            has_empty_event = any(e.name == "retrieval_empty" for e in span.events)

            if candidate_count == 0 or has_empty_event:
                suspicious_span_id = suspicious_span_id or span.span_id
                max_score = max(max_score, 0.95)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="retrieval_empty",
                        key="retrieval.candidate_count",
                        value_hash=str(candidate_count),
                        description=f"Retrieval span returned 0 candidates (query='{span.attributes.get('retrieval.query', '')}')",
                        strength=0.95,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=max_score,
                severity=IncidentSeverity.P1,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Retrieval produced insufficient or empty documents, starving downstream generation.",
                remediation="Expand query terms, lower vector similarity thresholds, or re-index the enterprise corpus.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class RankingAnomalyRule(ForensicRule):
    """Rule 2: Detects when the reranker aggressively pruned or dropped documents.

    Pattern:
      - reranker input > 0 but output count == 0, or
      - reranker eliminated high-scoring candidates.
    """

    name = "ranking_anomaly_rule"
    category = FailureCategory.RANKING_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        reranker_spans = trace.get_spans_by_type(SpanType.RERANKER)
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None

        for span in reranker_spans:
            in_count = span.attributes.get("reranker.input_count", 0)
            out_count = span.attributes.get("reranker.output_count", -1)

            if in_count > 0 and out_count == 0:
                suspicious_span_id = suspicious_span_id or span.span_id
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="reranker_zero_output",
                        key="reranker.output_count",
                        value_hash="0",
                        description=f"Reranker eliminated all {in_count} input chunks",
                        strength=0.9,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=0.90,
                severity=IncidentSeverity.P2,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="The reranker filtered out all retrieved candidates, leaving the prompt with no context.",
                remediation="Adjust minimum relevance threshold in the reranker component.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class GenerationContradictionRule(ForensicRule):
    """Rule 3: Detects model generation contradicting ground truth context.

    Pattern:
      - model span OK or response returned, but answer contains unsupported/contradictory claims,
      - citation invalid count > 0 or model error.
    """

    name = "generation_contradiction_rule"
    category = FailureCategory.GENERATION_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None
        score = 0.0

        citation_spans = trace.get_spans_by_type(SpanType.CITATION)
        for span in citation_spans:
            invalid_count = span.attributes.get("citation.invalid_count", 0)
            has_invalid_event = any(e.name == "citation_invalid" for e in span.events)
            if invalid_count > 0 or has_invalid_event:
                suspicious_span_id = suspicious_span_id or span.span_id
                score = max(score, 0.85)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="citation_hallucination",
                        key="citation.invalid_count",
                        value_hash=str(invalid_count),
                        description=f"Model output cited non-existent or unsupported documents ({invalid_count} invalid)",
                        strength=0.85,
                    )
                )

        model_spans = trace.get_spans_by_type(SpanType.MODEL)
        for span in model_spans:
            if span.status == SpanStatus.ERROR:
                suspicious_span_id = suspicious_span_id or span.span_id
                score = max(score, 0.90)
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="model_execution_error",
                        key="error_message",
                        value_hash=str(span.error_message),
                        description=f"Model generation error: {span.error_message}",
                        strength=0.90,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=score,
                severity=IncidentSeverity.P1,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Model generated statements that failed grounding or citation verification.",
                remediation="Strengthen citation constraints in the generation prompt or use a more grounded model.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class ToolRetryStormRule(ForensicRule):
    """Rule 4: Detects tool execution retry amplification.

    Pattern:
      - 2 or more retry events on tool spans, or
      - tool execution latency exploding due to retries.
    """

    name = "tool_retry_storm_rule"
    category = FailureCategory.RETRY_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        tool_spans = trace.get_spans_by_type(SpanType.TOOL)
        retry_events = []
        suspicious_span_id: Optional[str] = None

        for span in tool_spans:
            for event in span.events:
                if event.name == "retry":
                    suspicious_span_id = suspicious_span_id or span.span_id
                    retry_events.append((span, event))

        if len(retry_events) >= 2:
            confidence = min(0.95, 0.70 + 0.08 * len(retry_events))
            evidence_list = [
                Evidence(
                    span_id=span.span_id,
                    type="retry_amplification",
                    key="retry_count",
                    value_hash=str(len(retry_events)),
                    description=f"Tool '{span.name}' retried {len(retry_events)} times, accumulating latency",
                    strength=confidence,
                )
            ]
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=confidence,
                severity=IncidentSeverity.P2,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation=f"Excessive tool retries detected ({len(retry_events)} retries), causing latency degradation.",
                remediation="Apply circuit breakers, exponential backoff caps, or fix upstream tool instability.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class PolicyBypassRule(ForensicRule):
    """Rule 5: Detects critical policy violations and guardrail bypasses.

    Pattern:
      - policy span allowed == False, but request completed, or
      - policy_bypass event present.
    """

    name = "policy_bypass_rule"
    category = FailureCategory.POLICY_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        policy_spans = trace.get_spans_by_type(SpanType.POLICY)
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None
        is_bypass = False

        for span in policy_spans:
            is_bypassed = span.attributes.get("policy.is_bypassed", False)
            has_bypass_event = any(e.name == "policy_bypass" for e in span.events)
            has_block_event = any(e.name == "policy_block" for e in span.events)
            allowed = span.attributes.get("policy.allowed", True)

            if is_bypassed or has_bypass_event:
                suspicious_span_id = suspicious_span_id or span.span_id
                is_bypass = True
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="policy_bypass",
                        key="policy.is_bypassed",
                        value_hash="True",
                        description="Security guardrail denied execution but agent bypassed policy check",
                        strength=0.98,
                    )
                )
            elif not allowed and trace.outcome.value == "success":
                suspicious_span_id = suspicious_span_id or span.span_id
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="policy_ignored",
                        key="policy.allowed",
                        value_hash="False",
                        description="Policy checker denied access but request completed as success",
                        strength=0.90,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=0.98 if is_bypass else 0.90,
                severity=IncidentSeverity.P0 if is_bypass else IncidentSeverity.P1,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Security guardrail policy violation or unauthorized bypass detected during execution.",
                remediation="Immediately enforce strict halting on policy denial in pipeline orchestrator.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class ContextExplosionRule(ForensicRule):
    """Rule 6: Detects context bloat and token budget inflation.

    Pattern:
      - context_bloated event, or
      - input token usage > 2000 tokens for single step.
    """

    name = "context_explosion_rule"
    category = FailureCategory.CONTEXT_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None

        for span in trace.spans:
            has_bloat_event = any(e.name == "context_bloated" for e in span.events)
            input_tokens = span.attributes.get("gen_ai.usage.input_tokens", 0)

            if has_bloat_event or input_tokens > 2000:
                suspicious_span_id = suspicious_span_id or span.span_id
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="context_bloat",
                        key="gen_ai.usage.input_tokens",
                        value_hash=str(input_tokens),
                        description=f"Context explosion: {input_tokens} input tokens in span '{span.name}'",
                        strength=0.85,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=0.85,
                severity=IncidentSeverity.P2,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Prompt context grew excessively large, causing latency and cost bloat.",
                remediation="Prune redundant documents, deduplicate chunks, and apply sliding window summarization.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class EvaluationBlindSpotRule(ForensicRule):
    """Rule 7: Detects plausible answers lacking verifiable evidence.

    Pattern:
      - trace marked success, but citation_missing event or citation count == 0 when required.
    """

    name = "evaluation_blind_spot_rule"
    category = FailureCategory.EVALUATION_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        citation_spans = trace.get_spans_by_type(SpanType.CITATION)
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None

        for span in citation_spans:
            has_missing_event = any(e.name == "citation_missing" for e in span.events)
            total_citations = span.attributes.get("citation.total_count", -1)

            if (has_missing_event or total_citations == 0) and trace.outcome.value == "success":
                suspicious_span_id = suspicious_span_id or span.span_id
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="unverified_generation",
                        key="citation.total_count",
                        value_hash=str(total_citations),
                        description="Response marked successful but contains zero grounded citations",
                        strength=0.80,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=0.80,
                severity=IncidentSeverity.P2,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Superficial check succeeded but generation lacked grounding evidence (evaluation blind spot).",
                remediation="Incorporate deterministic attribution checks into production response filters.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


class TimeoutSLABreachRule(ForensicRule):
    """Rule 8: Detects SLA latency breaches and upstream timeouts.

    Pattern:
      - span error containing timeout, or span duration exceeding SLA threshold (e.g. > 5000ms).
    """

    name = "timeout_sla_breach_rule"
    category = FailureCategory.TIMEOUT_FAILURE

    def evaluate(self, trace: TraceData) -> ForensicRuleResult:
        evidence_list: List[Evidence] = []
        suspicious_span_id: Optional[str] = None

        for span in trace.spans:
            is_timeout_err = "timeout" in (span.error_message or "").lower()
            exceeds_sla = span.duration_ms > 5000.0

            if is_timeout_err or exceeds_sla:
                suspicious_span_id = suspicious_span_id or span.span_id
                evidence_list.append(
                    Evidence(
                        span_id=span.span_id,
                        type="latency_breach",
                        key="duration_ms",
                        value_hash=f"{span.duration_ms:.1f}",
                        description=f"Span '{span.name}' took {span.duration_ms:.1f}ms, exceeding latency budget",
                        strength=0.90 if is_timeout_err else 0.75,
                    )
                )

        if evidence_list:
            return ForensicRuleResult(
                rule_name=self.name,
                category=self.category,
                matched=True,
                confidence=0.85,
                severity=IncidentSeverity.P1,
                first_suspicious_span_id=suspicious_span_id,
                evidence=evidence_list,
                explanation="Operation exceeded maximum latency SLA budget or timed out.",
                remediation="Review upstream API timeouts, configure fallback models, or enable speculative execution.",
            )
        return ForensicRuleResult(rule_name=self.name, category=self.category, matched=False)


DEFAULT_FORENSIC_RULES: List[ForensicRule] = [
    PolicyBypassRule(),
    RetrievalMissRule(),
    RankingAnomalyRule(),
    GenerationContradictionRule(),
    ToolRetryStormRule(),
    ContextExplosionRule(),
    EvaluationBlindSpotRule(),
    TimeoutSLABreachRule(),
]
