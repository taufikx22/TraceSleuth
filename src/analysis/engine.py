"""TraceSleuth Analysis — Root-Cause Analysis (RCA) Engine.

Constructs the causal execution graph from trace spans, isolates the earliest
suspicious operation, evaluates deterministic forensic rules, and ranks
root-cause hypotheses with confidence scores and evidence chains.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from src.analysis.rules import DEFAULT_FORENSIC_RULES, ForensicRule, ForensicRuleResult
from src.domain.incident import (
    Evidence,
    FailureCategory,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData


@dataclass
class CausalNode:
    """A node in the causal execution DAG."""

    span: SpanData
    parent_span_id: Optional[str] = None
    children_span_ids: List[str] = field(default_factory=list)
    depth: int = 0
    is_suspicious: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)


@dataclass
class HypothesisRanking:
    """A ranked root-cause hypothesis with confidence and supporting evidence."""

    category: FailureCategory
    confidence: float
    is_primary_root_cause: bool
    summary: str
    first_suspicious_span_id: Optional[str]
    evidence: List[Evidence]
    remediation: str
    downstream_symptoms: List[str] = field(default_factory=list)


class RootCauseEngine:
    """Forensic Root-Cause Analysis Engine for AI pipelines.

    Reconstructs the execution tree, identifies anomalies chronologically,
    differentiates root causes from downstream symptoms, and scores hypotheses.
    """

    def __init__(self, rules: Optional[List[ForensicRule]] = None) -> None:
        self.rules = rules or DEFAULT_FORENSIC_RULES

    def build_causal_graph(self, trace: TraceData) -> Dict[str, CausalNode]:
        """Builds a causal DAG representing the execution hierarchy."""
        nodes: Dict[str, CausalNode] = {}
        for s in trace.spans:
            nodes[s.span_id] = CausalNode(
                span=s,
                parent_span_id=s.parent_span_id,
            )

        for span_id, node in nodes.items():
            if node.parent_span_id and node.parent_span_id in nodes:
                nodes[node.parent_span_id].children_span_ids.append(span_id)

        # Compute depth
        def _set_depth(s_id: str, d: int) -> None:
            nodes[s_id].depth = d
            for c_id in nodes[s_id].children_span_ids:
                _set_depth(c_id, d + 1)

        for s_id, node in nodes.items():
            if not node.parent_span_id:
                _set_depth(s_id, 0)

        return nodes

    def find_earliest_suspicious_span(self, trace: TraceData) -> Optional[SpanData]:
        """Finds the earliest chronological span with an anomaly or error."""
        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        for span in sorted_spans:
            # 1. Error status
            if span.status == SpanStatus.ERROR:
                return span

            # 2. Critical events
            for event in span.events:
                if event.name in (
                    "policy_bypass",
                    "retrieval_empty",
                    "retry",
                    "citation_missing",
                    "citation_invalid",
                    "context_bloated",
                    "router_error",
                    "state_corruption",
                ):
                    return span

            # 3. Attributes indicating failure
            if span.span_type == SpanType.RETRIEVAL:
                if span.attributes.get("retrieval.candidate_count") == 0:
                    return span
            elif span.span_type == SpanType.POLICY:
                if span.attributes.get("policy.is_bypassed") or span.attributes.get("policy.allowed") is False:
                    return span
            elif span.span_type == SpanType.RERANKER:
                in_cnt = span.attributes.get("reranker.input_count", 0)
                out_cnt = span.attributes.get("reranker.output_count", -1)
                if in_cnt > 0 and out_cnt == 0:
                    return span

        return None

    def analyze(self, trace: TraceData) -> List[HypothesisRanking]:
        """Runs forensic analysis and returns ranked root-cause hypotheses."""
        rule_results: List[ForensicRuleResult] = []
        for rule in self.rules:
            res = rule.evaluate(trace)
            if res.matched:
                rule_results.append(res)

        if not rule_results:
            return []

        earliest_suspicious = self.find_earliest_suspicious_span(trace)
        earliest_id = earliest_suspicious.span_id if earliest_suspicious else None

        # Build causal order precedence
        # Spans earlier in the pipeline (Policy -> Retrieval -> Reranker -> Planner -> Tools -> Model -> Citations)
        # take precedence as root causes over later spans.
        type_precedence = {
            SpanType.POLICY: 10,
            SpanType.RETRIEVAL: 20,
            SpanType.RERANKER: 30,
            SpanType.CLASSIFIER: 35,
            SpanType.PLANNER: 40,
            SpanType.TOOL: 50,
            SpanType.MODEL: 60,
            SpanType.CITATION: 70,
            SpanType.AGENT_RUN: 99,
        }

        def _sort_key(r: ForensicRuleResult) -> Tuple[float, int]:
            # Priority: higher confidence, lower pipeline stage order
            span = trace.get_span_by_id(r.first_suspicious_span_id) if r.first_suspicious_span_id else None
            precedence = type_precedence.get(span.span_type, 50) if span else 50
            return (r.confidence, -precedence)

        rule_results.sort(key=_sort_key, reverse=True)

        rankings: List[HypothesisRanking] = []
        for i, res in enumerate(rule_results):
            is_primary = (i == 0)
            downstream = []
            if is_primary and len(rule_results) > 1:
                downstream = [f"{other.category.value} ({other.explanation})" for other in rule_results[1:]]

            rankings.append(
                HypothesisRanking(
                    category=res.category,
                    confidence=res.confidence,
                    is_primary_root_cause=is_primary,
                    summary=res.explanation,
                    first_suspicious_span_id=res.first_suspicious_span_id or earliest_id,
                    evidence=res.evidence,
                    remediation=res.remediation,
                    downstream_symptoms=downstream,
                )
            )

        return rankings

    def diagnose_to_incidents(self, trace: TraceData) -> List[Incident]:
        """Diagnoses a trace and produces full Incident models with evidence chains."""
        rankings = self.analyze(trace)
        if not rankings:
            return []

        now = datetime.now(timezone.utc)
        incidents: List[Incident] = []

        for rank in rankings:
            inc_id = f"inc_{uuid.uuid4().hex[:12]}"
            # Determine severity based on category and primary status
            if rank.category == FailureCategory.POLICY_FAILURE:
                sev = IncidentSeverity.P0
            elif rank.is_primary_root_cause:
                sev = IncidentSeverity.P1
            else:
                sev = IncidentSeverity.P2

            # Re-assign incident_id to evidence items
            evidence_items = []
            for ev in rank.evidence:
                evidence_items.append(
                    Evidence(
                        evidence_id=ev.evidence_id or f"ev_{uuid.uuid4().hex[:8]}",
                        incident_id=inc_id,
                        span_id=ev.span_id,
                        type=ev.type,
                        key=ev.key,
                        value_hash=ev.value_hash,
                        description=ev.description,
                        strength=ev.strength,
                    )
                )

            incidents.append(
                Incident(
                    incident_id=inc_id,
                    trace_id=trace.trace_id,
                    category=rank.category,
                    severity=sev,
                    summary=rank.summary,
                    first_suspicious_span_id=rank.first_suspicious_span_id,
                    hypothesis_score=rank.confidence,
                    evidence=evidence_items,
                    status=IncidentStatus.OPEN,
                    created_at=now,
                    metadata={
                        "is_primary_root_cause": rank.is_primary_root_cause,
                        "remediation": rank.remediation,
                        "downstream_symptoms": rank.downstream_symptoms,
                    },
                )
            )

        return incidents
