"""Instrumented Enterprise Research Agent.

Instruments all 8 pipeline operations with OpenTelemetry spans, standard GenAI attributes,
events (retry, policy_block, citation_missing), and correlates traces end-to-end.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional, Tuple
from opentelemetry.trace import StatusCode
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
from src.domain.trace import SpanType, TraceData
from src.telemetry.instrumentation import TelemetryManager, telemetry
from src.telemetry.redaction import default_redactor, hash_content


class InstrumentedResearchAgent:
    """Enterprise Research Agent with full OpenTelemetry causal tracing."""

    def __init__(
        self,
        corpus: Optional[EnterpriseCorpus] = None,
        telemetry_manager: Optional[TelemetryManager] = None,
        app_version: str = "0.1.0",
        model_config_version: str = "v1.0.0",
        prompt_version: str = "v1.2.0",
        environment: str = "development",
    ) -> None:
        self.corpus = corpus or EnterpriseCorpus()
        self.telemetry = telemetry_manager or telemetry
        self.app_version = app_version
        self.model_config_version = model_config_version
        self.prompt_version = prompt_version
        self.environment = environment

        # Pipeline components
        self.classifier = QueryClassifier()
        self.retriever = VectorRetriever(corpus=self.corpus)
        self.reranker = Reranker()
        self.planner = Planner()
        self.calculator = CalculatorTool()
        self.doc_lookup = DocumentLookupTool(corpus=self.corpus)
        self.policy_checker = PolicyChecker()
        self.generator = AnswerGenerator()
        self.citation_formatter = CitationFormatter()

    def run(self, request: AgentRequest) -> Tuple[AgentResponse, TraceData]:
        """Runs the agent with full end-to-end OpenTelemetry instrumentation."""
        start_wall = time.perf_counter()
        req_id = request.request_id or f"req_{uuid.uuid4().hex[:12]}"
        failure_mode = request.failure_mode

        root_attributes = {
            "tracesleuth.request_id": req_id,
            "tracesleuth.session_id": request.session_id or "sess_default",
            "tracesleuth.tenant_id": request.tenant_id,
            "tracesleuth.app_version": self.app_version,
            "tracesleuth.model_config_version": self.model_config_version,
            "tracesleuth.prompt_version": self.prompt_version,
            "tracesleuth.environment": self.environment,
            "agent.query": request.query,
            "agent.query_hash": hash_content(request.query),
            "agent.failure_mode": failure_mode.value,
        }

        with self.telemetry.start_span("agent.request", SpanType.AGENT_RUN, root_attributes) as root_span:
            trace_id_hex = format(root_span.get_span_context().trace_id, "032x")

            # 1. Query Classification
            with self.telemetry.start_span("classifier.query_classification", SpanType.CLASSIFIER) as span:
                classification = self.classifier.classify(request.query)
                span.set_attribute("classification.domain", classification.domain)
                span.set_attribute("classification.category", classification.category)
                span.set_attribute("classification.sensitivity", classification.sensitivity)
                span.set_attribute("classification.requires_tools", classification.requires_tools)

            # 2. Policy Verification
            with self.telemetry.start_span("policy.evaluate_guardrails", SpanType.POLICY) as span:
                policy_result = self.policy_checker.evaluate(
                    query=request.query,
                    classification=classification,
                    failure_mode=failure_mode,
                )
                span.set_attribute("policy.decision", policy_result.decision)
                span.set_attribute("policy.rules_evaluated", str(policy_result.rules_evaluated))
                span.set_attribute("policy.violations", str(policy_result.violations))
                span.set_attribute("policy.is_bypassed", policy_result.is_bypassed)

                if policy_result.decision == "deny":
                    if policy_result.is_bypassed:
                        span.add_event(
                            "policy_bypass",
                            {"reason": "Simulated bypass: Execution proceeded despite policy denial"},
                        )
                    else:
                        span.add_event(
                            "policy_block",
                            {"reason": f"Request blocked: {policy_result.violations}"},
                        )
                        span.set_status(StatusCode.ERROR, description="Policy violation")
                        elapsed = (time.perf_counter() - start_wall) * 1000.0
                        resp = AgentResponse(
                            request_id=req_id,
                            query=request.query,
                            answer=f"Request denied by Enterprise Security Policy: {', '.join(policy_result.violations)}",
                            classification=classification,
                            policy_check=policy_result,
                            duration_ms=round(elapsed, 2),
                            status="denied",
                            error="Policy violation",
                        )
                        # Root span marks completed
                        return resp, self.telemetry.get_trace(trace_id_hex)

            # 3. Vector Retrieval
            with self.telemetry.start_span("retrieval.vector_search", SpanType.RETRIEVAL) as span:
                candidate_docs = self.retriever.retrieve(
                    query=request.query,
                    top_k=5,
                    category=classification.domain if classification.domain != "general" else None,
                    failure_mode=failure_mode,
                )
                span.set_attribute("retrieval.retriever", "enterprise_vector_store")
                span.set_attribute("retrieval.candidate_count", len(candidate_docs))
                span.set_attribute("retrieval.returned_document_ids", str([d.id for d in candidate_docs]))
                span.set_attribute("retrieval.scores", str([d.score for d in candidate_docs]))

                if failure_mode == FailureMode.RETRIEVAL_MISS or len(candidate_docs) == 0:
                    span.add_event("retrieval_empty", {"query": request.query[:50]})
                elif failure_mode == FailureMode.CONTEXT_EXPLOSION:
                    span.add_event(
                        "context_bloated",
                        {"chunk_count": len(candidate_docs), "multiplier": 5},
                    )

            # 4. Reranking
            with self.telemetry.start_span("reranker.score_and_filter", SpanType.RERANKER) as span:
                reranked_docs = self.reranker.rerank(
                    query=request.query,
                    candidates=candidate_docs,
                    top_k=3,
                )
                span.set_attribute("reranker.input_count", len(candidate_docs))
                span.set_attribute("reranker.output_count", len(reranked_docs))
                span.set_attribute("reranker.returned_ids", str([d.id for d in reranked_docs]))

            # 5. Planning
            with self.telemetry.start_span("planner.decompose_steps", SpanType.PLANNER) as span:
                plan = self.planner.plan(
                    query=request.query,
                    classification=classification,
                )
                span.set_attribute("planner.step_count", len(plan.steps))
                span.set_attribute("planner.suggested_tools", str(classification.suggested_tools))

            # 6. Tool Execution
            tool_results: List[ToolExecutionResult] = []
            for step in plan.steps:
                if step.tool_name == "calculator":
                    expr = step.tool_arguments.get("expression", "100 + 50")
                    retry = 0
                    max_retries = request.max_retries
                    while retry <= max_retries:
                        with self.telemetry.start_span("tool.calculator", SpanType.TOOL) as tool_span:
                            tool_span.set_attribute("tool.name", "calculator")
                            tool_span.set_attribute("tool.arguments_hash", hash_content(expr))
                            tool_span.set_attribute("tool.retry_count", retry)

                            res = self.calculator.execute(
                                expression=expr,
                                failure_mode=failure_mode,
                                retry_count=retry,
                            )
                            tool_results.append(res)
                            tool_span.set_attribute("tool.result_status", res.status)
                            tool_span.set_attribute("tool.latency_ms", res.latency_ms)

                            if res.status == "error":
                                tool_span.set_status(StatusCode.ERROR, description=res.error_message)
                                tool_span.add_event(
                                    "retry",
                                    {"retry_count": retry, "error": str(res.error_message)},
                                )
                            else:
                                tool_span.set_status(StatusCode.OK)
                                break
                        retry += 1

                elif step.tool_name == "document_lookup" and reranked_docs:
                    first_doc_id = reranked_docs[0].id
                    with self.telemetry.start_span("tool.document_lookup", SpanType.TOOL) as tool_span:
                        tool_span.set_attribute("tool.name", "document_lookup")
                        tool_span.set_attribute("tool.arguments_hash", hash_content(first_doc_id))
                        tool_span.set_attribute("tool.retry_count", 0)

                        res = self.doc_lookup.execute(
                            document_id=first_doc_id,
                            failure_mode=failure_mode,
                            retry_count=0,
                        )
                        tool_results.append(res)
                        tool_span.set_attribute("tool.result_status", res.status)
                        tool_span.set_attribute("tool.latency_ms", res.latency_ms)

            # 7. Model Answer Generation (GenAI Semantic Conventions)
            with self.telemetry.start_span("model.answer_generation", SpanType.MODEL) as span:
                answer, token_stats = self.generator.generate(
                    query=request.query,
                    context_docs=reranked_docs,
                    tool_results=tool_results,
                    failure_mode=failure_mode,
                )
                span.set_attribute("gen_ai.system", "synthetic_llm")
                span.set_attribute("gen_ai.request.model", "enterprise-research-v1")
                span.set_attribute("gen_ai.usage.input_tokens", token_stats["input_tokens"])
                span.set_attribute("gen_ai.usage.output_tokens", token_stats["output_tokens"])
                span.set_attribute("gen_ai.response.finish_reasons", "['stop']")
                span.set_attribute("gen_ai.prompt_version", self.prompt_version)
                span.set_attribute("model.answer_hash", hash_content(answer))

            # 8. Citation Verification
            with self.telemetry.start_span("citation.verify_attribution", SpanType.CITATION) as span:
                citations = self.citation_formatter.validate_citations(
                    answer=answer,
                    retrieved_docs=reranked_docs,
                )
                valid_count = sum(1 for c in citations if c.is_valid)
                invalid_count = len(citations) - valid_count
                span.set_attribute("citation.total_count", len(citations))
                span.set_attribute("citation.valid_count", valid_count)
                span.set_attribute("citation.invalid_count", invalid_count)

                for c in citations:
                    if not c.is_valid:
                        span.add_event(
                            "citation_missing" if c.document_id == "NONE" else "citation_invalid",
                            {"claim": c.claim, "reason": c.reason or ""},
                        )

            elapsed = (time.perf_counter() - start_wall) * 1000.0
            resp = AgentResponse(
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

        # Retrieve assembled trace from OpenTelemetry collector
        trace_data = self.telemetry.get_trace(trace_id_hex)
        if trace_data:
            trace_data.metadata["query"] = request.query
            trace_data.metadata["failure_mode"] = failure_mode.value
        return resp, trace_data
