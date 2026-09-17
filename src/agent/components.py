"""Pipeline components for the Enterprise Research Agent.

Implements QueryClassifier, VectorRetriever, Reranker, Planner,
PolicyChecker, AnswerGenerator, and CitationFormatter with failure injection.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from src.agent.corpus import EnterpriseCorpus
from src.agent.tools import CalculatorTool, DocumentLookupTool
from src.agent.types import (
    Citation,
    DocumentChunk,
    ExecutionPlan,
    FailureMode,
    PlannedStep,
    PolicyCheckResult,
    QueryClassification,
    ToolExecutionResult,
)


class QueryClassifier:
    """Classifies incoming user queries by domain, sensitivity, and required capabilities."""

    def classify(self, query: str) -> QueryClassification:
        q_lower = query.lower()

        def _has_word(words: list[str]) -> bool:
            return any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in words)

        # Domain classification
        if _has_word(["arr", "revenue", "margin", "capex", "expenses", "net income", "budget", "financial"]) or "$" in q_lower:
            domain = "financial"
        elif _has_word(["slo", "sla", "latency", "api", "mtls", "security", "rto", "rpo", "zero trust"]):
            domain = "technical"
        elif _has_word(["pto", "vacation", "parental leave", "sabbatical", "hr", "benefits", "employee"]):
            domain = "hr"
        elif _has_word(["tier", "enterprise", "soc2", "hipaa", "compliance", "sso", "pricing", "feature"]):
            domain = "product"
        else:
            domain = "general"

        # Sensitivity detection
        if _has_word(["password", "secret", "private key", "ssn", "confidential", "insider"]):
            sensitivity = "restricted"
        elif domain in ["financial", "technical"]:
            sensitivity = "internal"
        else:
            sensitivity = "public"

        # Determine if tools like calculator are likely needed
        requires_math = any(op in q_lower for op in ["calculate", "margin", "%", "difference", "total", "sum", "ratio", "+", "-", "*", "/"])
        suggested_tools = ["calculator"] if requires_math else ["document_lookup"]

        return QueryClassification(
            category="factual_inquiry",
            domain=domain,
            sensitivity=sensitivity,
            requires_tools=requires_math,
            suggested_tools=suggested_tools,
        )


class VectorRetriever:
    """Vector & semantic retriever searching the enterprise knowledge store."""

    def __init__(self, corpus: Optional[EnterpriseCorpus] = None) -> None:
        self.corpus = corpus or EnterpriseCorpus()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        failure_mode: FailureMode = FailureMode.HEALTHY,
    ) -> List[DocumentChunk]:
        # In FailureMode.RETRIEVAL_MISS: deliberately withhold relevant documents
        if failure_mode == FailureMode.RETRIEVAL_MISS:
            return []

        results = self.corpus.search(query=query, top_k=top_k, category=category)

        # In FailureMode.CONTEXT_EXPLOSION: duplicate retrieved documents to inflate tokens
        if failure_mode == FailureMode.CONTEXT_EXPLOSION and results:
            bloated: List[DocumentChunk] = []
            for i in range(5):
                for chunk in results:
                    clone = chunk.model_copy()
                    clone.id = f"{chunk.id}_dup_{i}"
                    clone.content = (chunk.content + " ") * 6  # Bloat context tokens
                    bloated.append(clone)
            return bloated

        return results


class Reranker:
    """Reranks candidate document chunks based on score and relevance heuristics."""

    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: int = 3,
        min_threshold: float = 0.05,
    ) -> List[DocumentChunk]:
        if not candidates:
            return []

        # Sort descending by score
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        filtered = [c for c in sorted_candidates if c.score >= min_threshold]
        return filtered[:top_k]


class Planner:
    """Decomposes user queries into sequential reasoning and tool execution steps."""

    def plan(
        self,
        query: str,
        classification: QueryClassification,
    ) -> ExecutionPlan:
        steps: List[PlannedStep] = []

        # Step 1: Retrieval
        steps.append(
            PlannedStep(
                step_id=1,
                description=f"Retrieve relevant enterprise documentation for domain: {classification.domain}",
                tool_name="document_lookup",
                tool_arguments={"domain": classification.domain},
            )
        )

        # Step 2: Tool execution if calculation is needed
        if classification.requires_tools:
            # Extract basic math expression if present, or generic arithmetic
            expression = "428 - 184" if "financial" in classification.domain else "25 - 5"
            steps.append(
                PlannedStep(
                    step_id=2,
                    description="Perform arithmetic computation on retrieved financial/operational metrics",
                    tool_name="calculator",
                    tool_arguments={"expression": expression},
                )
            )

        # Step 3: Synthesis
        steps.append(
            PlannedStep(
                step_id=len(steps) + 1,
                description="Synthesize final evidence-grounded answer with source citations",
            )
        )

        reasoning = (
            f"Query classified as {classification.domain} ({classification.sensitivity}). "
            f"Execution decomposed into {len(steps)} verified steps."
        )
        return ExecutionPlan(reasoning=reasoning, steps=steps)


class PolicyChecker:
    """Evaluates security, compliance, data access, and PII guardrails."""

    def evaluate(
        self,
        query: str,
        classification: QueryClassification,
        failure_mode: FailureMode = FailureMode.HEALTHY,
    ) -> PolicyCheckResult:
        rules_evaluated = [
            "rule_check_pii_request",
            "rule_check_restricted_data_access",
            "rule_check_rate_limit_compliance",
        ]
        violations: List[str] = []

        if classification.sensitivity == "restricted":
            violations.append("Query requests restricted enterprise credentials or sensitive keys")

        # In FailureMode.POLICY_BYPASS: force a policy denial violation
        if failure_mode == FailureMode.POLICY_BYPASS:
            violations.append("Unauthorized administrative query without Level-3 MFA")
            return PolicyCheckResult(
                decision="deny",
                rules_evaluated=rules_evaluated,
                violations=violations,
                is_bypassed=True,  # Execution bypasses this block to demonstrate the forensic failure
            )

        decision = "deny" if violations else "allow"
        return PolicyCheckResult(
            decision=decision,
            rules_evaluated=rules_evaluated,
            violations=violations,
            is_bypassed=False,
        )


class AnswerGenerator:
    """Synthesizes final answer from context, planner observations, and tool results."""

    def generate(
        self,
        query: str,
        context_docs: List[DocumentChunk],
        tool_results: List[ToolExecutionResult],
        failure_mode: FailureMode = FailureMode.HEALTHY,
    ) -> Tuple[str, Dict[str, int]]:
        """Generates answer and returns (answer_text, token_counts)."""

        # In FailureMode.GENERATION_CONTRADICTION: model deliberately contradicts available context
        if failure_mode == FailureMode.GENERATION_CONTRADICTION:
            answer = (
                "Based on the enterprise records, Enterprise Corp suffered a catastrophic revenue decline "
                "in Q3 2025 with total cloud ARR dropping by 85% to only $12M, and operating expenses surging "
                "out of control [doc_fin_2025_q3]."
            )
            return answer, {"input_tokens": 1450, "output_tokens": 68}

        # In FailureMode.EVALUATION_BLIND_SPOT: plausible-looking prose omitting valid citations
        if failure_mode == FailureMode.EVALUATION_BLIND_SPOT:
            answer = (
                "Enterprise Corp maintains strong financial momentum across all cloud divisions. "
                "Infrastructure investments are scaling efficiently in accordance with modern industry standards."
            )
            return answer, {"input_tokens": 820, "output_tokens": 42}

        # If retrieval miss occurred and no documents are present
        if not context_docs:
            answer = "I apologize, but I could not find any relevant enterprise documentation to answer your query."
            return answer, {"input_tokens": 310, "output_tokens": 28}

        # Healthy grounded answer generation
        doc_citations = " ".join([f"[{d.id}]" for d in context_docs])
        doc_summaries = "\n".join([f"- {d.title}: {d.content}" for d in context_docs])

        calc_summary = ""
        for tr in tool_results:
            if tr.tool_name == "calculator" and tr.status == "success":
                calc_summary = f" Computed calculation: {tr.arguments.get('expression')} = {tr.result}."

        answer = (
            f"Based on the retrieved enterprise records {doc_citations}:\n"
            f"{doc_summaries}\n"
            f"{calc_summary}\n"
            f"The data confirms that the target metrics satisfy enterprise operational guidelines."
        )

        # Estimate realistic token counts
        input_tokens = len(query.split()) + sum(len(d.content.split()) for d in context_docs) + 250
        output_tokens = len(answer.split())
        return answer, {"input_tokens": input_tokens, "output_tokens": output_tokens}


class CitationFormatter:
    """Verifies that facts in generated output cite retrieved document IDs."""

    def validate_citations(
        self,
        answer: str,
        retrieved_docs: List[DocumentChunk],
    ) -> List[Citation]:
        valid_doc_ids = {d.id for d in retrieved_docs}
        cited_ids = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", answer)

        citations: List[Citation] = []
        for cid in cited_ids:
            if cid in valid_doc_ids:
                citations.append(Citation(claim=f"Citation to {cid}", document_id=cid, is_valid=True))
            else:
                citations.append(
                    Citation(
                        claim=f"Citation to {cid}",
                        document_id=cid,
                        is_valid=False,
                        reason=f"Cited document '{cid}' was not among retrieved documents",
                    )
                )

        # If docs were retrieved but answer cites none, record missing citation
        if retrieved_docs and not cited_ids:
            citations.append(
                Citation(
                    claim="Grounding check",
                    document_id="NONE",
                    is_valid=False,
                    reason="Answer failed to cite any of the retrieved documents",
                )
            )

        return citations
