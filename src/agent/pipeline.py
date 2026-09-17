"""Enterprise Research Agent Pipeline.

Orchestrates all 8 agent operations:
1. Query Classification
2. Vector Retrieval
3. Reranking
4. Planning
5. Tool Invocations (Calculator, Document Lookup with retries)
6. Policy Guardrails Check
7. Answer Generation
8. Citation Verification
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional
from src.agent.components import (
    AnswerGenerator,
    CitationFormatter,
    Planner,
    PolicyChecker,
    QueryClassifier,
    Reranker,
    VectorRetriever,
)
from src.agent.corpus import EnterpriseCorpus
from src.agent.tools import CalculatorTool, DocumentLookupTool
from src.agent.types import (
    AgentRequest,
    AgentResponse,
    Citation,
    DocumentChunk,
    ExecutionPlan,
    FailureMode,
    PolicyCheckResult,
    QueryClassification,
    ToolExecutionResult,
)


class EnterpriseResearchAgent:
    """The Reference Agent for TraceSleuth failure forensics."""

    def __init__(self, corpus: Optional[EnterpriseCorpus] = None) -> None:
        self.corpus = corpus or EnterpriseCorpus()
        self.classifier = QueryClassifier()
        self.retriever = VectorRetriever(corpus=self.corpus)
        self.reranker = Reranker()
        self.planner = Planner()
        self.calculator = CalculatorTool()
        self.doc_lookup = DocumentLookupTool(corpus=self.corpus)
        self.policy_checker = PolicyChecker()
        self.generator = AnswerGenerator()
        self.citation_formatter = CitationFormatter()

    def run(self, request: AgentRequest) -> AgentResponse:
        """Executes the reference agent pipeline synchronously."""
        start_time = time.perf_counter()
        req_id = request.request_id or f"req_{uuid.uuid4().hex[:12]}"
        failure_mode = request.failure_mode

        # 1. Query Classification
        classification: QueryClassification = self.classifier.classify(request.query)

        # 2. Policy Verification
        policy_result: PolicyCheckResult = self.policy_checker.evaluate(
            query=request.query,
            classification=classification,
            failure_mode=failure_mode,
        )
        if policy_result.decision == "deny" and not policy_result.is_bypassed:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return AgentResponse(
                request_id=req_id,
                query=request.query,
                answer=f"Request denied by Enterprise Security Policy: {', '.join(policy_result.violations)}",
                classification=classification,
                policy_check=policy_result,
                duration_ms=round(elapsed, 2),
                status="denied",
                error="Policy violation",
            )

        # 3. Vector Retrieval
        candidate_docs: List[DocumentChunk] = self.retriever.retrieve(
            query=request.query,
            top_k=5,
            category=classification.domain if classification.domain != "general" else None,
            failure_mode=failure_mode,
        )

        # 4. Reranking
        reranked_docs: List[DocumentChunk] = self.reranker.rerank(
            query=request.query,
            candidates=candidate_docs,
            top_k=3,
        )

        # 5. Planning
        plan: ExecutionPlan = self.planner.plan(
            query=request.query,
            classification=classification,
        )

        # 6. Tool Execution with retry handling
        tool_results: List[ToolExecutionResult] = []
        for step in plan.steps:
            if step.tool_name == "calculator":
                expr = step.tool_arguments.get("expression", "100 + 50")
                retry = 0
                max_retries = request.max_retries
                while retry <= max_retries:
                    res = self.calculator.execute(
                        expression=expr,
                        failure_mode=failure_mode,
                        retry_count=retry,
                    )
                    tool_results.append(res)
                    if res.status == "success":
                        break
                    retry += 1

            elif step.tool_name == "document_lookup" and reranked_docs:
                first_doc_id = reranked_docs[0].id
                res = self.doc_lookup.execute(
                    document_id=first_doc_id,
                    failure_mode=failure_mode,
                    retry_count=0,
                )
                tool_results.append(res)

        # 7. Answer Generation
        answer, _token_stats = self.generator.generate(
            query=request.query,
            context_docs=reranked_docs,
            tool_results=tool_results,
            failure_mode=failure_mode,
        )

        # 8. Citation Verification
        citations: List[Citation] = self.citation_formatter.validate_citations(
            answer=answer,
            retrieved_docs=reranked_docs,
        )

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return AgentResponse(
            request_id=req_id,
            query=request.query,
            answer=answer,
            classification=classification,
            retrieved_documents=reranked_docs,
            plan=plan,
            tool_results=tool_results,
            policy_check=policy_result,
            citations=citations,
            duration_ms=round(elapsed, 2),
            status="success",
        )
