"""Unit tests for Reference Enterprise Research Agent (Milestone 0).

Verifies the 8 modular agent operations and failure injection capabilities.
"""

import pytest
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
from src.agent.pipeline import EnterpriseResearchAgent
from src.agent.tools import CalculatorTool, DocumentLookupTool
from src.agent.types import AgentRequest, FailureMode


@pytest.fixture
def corpus() -> EnterpriseCorpus:
    return EnterpriseCorpus()


def test_operation_1_query_classifier():
    classifier = QueryClassifier()
    fin_res = classifier.classify("What is the Q3 cloud ARR and gross margin?")
    assert fin_res.domain == "financial"
    assert fin_res.requires_tools is True

    tech_res = classifier.classify("What is the API latency SLO and RTO for 2026?")
    assert tech_res.domain == "technical"

    hr_res = classifier.classify("How many PTO days can be carried over?")
    assert hr_res.domain == "hr"


def test_operation_2_vector_retriever(corpus):
    retriever = VectorRetriever(corpus=corpus)
    docs = retriever.retrieve(query="cloud ARR financial performance", top_k=3)
    assert len(docs) > 0
    assert any("doc_fin" in d.id for d in docs)
    assert docs[0].score > 0.0


def test_operation_3_reranker():
    reranker = Reranker()
    from src.agent.types import DocumentChunk
    candidates = [
        DocumentChunk(id="d1", title="Low match", content="random text", category="general", source="s1", score=0.1),
        DocumentChunk(id="d2", title="High match", content="revenue details", category="financial", source="s2", score=0.85),
    ]
    reranked = reranker.rerank("revenue", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].id == "d2"


def test_operation_4_planner():
    classifier = QueryClassifier()
    cls_res = classifier.classify("Calculate the revenue margin")
    planner = Planner()
    plan = planner.plan("Calculate the revenue margin", cls_res)
    assert len(plan.steps) >= 3
    tool_names = [s.tool_name for s in plan.steps]
    assert "document_lookup" in tool_names
    assert "calculator" in tool_names


def test_operation_5_tools(corpus):
    calc = CalculatorTool()
    calc_res = calc.execute("428 - 184")
    assert calc_res.status == "success"
    assert calc_res.result == 244.0

    doc_tool = DocumentLookupTool(corpus=corpus)
    lookup_res = doc_tool.execute("doc_fin_2025_q3")
    assert lookup_res.status == "success"
    assert lookup_res.result["id"] == "doc_fin_2025_q3"


def test_operation_6_policy_checker():
    checker = PolicyChecker()
    classifier = QueryClassifier()
    cls_res = classifier.classify("What is the admin secret password?")
    res = checker.evaluate("What is the admin secret password?", cls_res)
    assert res.decision == "deny"
    assert len(res.violations) > 0


def test_operation_7_answer_generator(corpus):
    gen = AnswerGenerator()
    docs = corpus.search("cloud ARR", top_k=1)
    answer, token_stats = gen.generate("What is cloud ARR?", docs, [])
    assert "Enterprise Corp" in answer
    assert token_stats["input_tokens"] > 0
    assert token_stats["output_tokens"] > 0


def test_operation_8_citation_formatter(corpus):
    formatter = CitationFormatter()
    docs = corpus.search("cloud ARR", top_k=1)
    doc_id = docs[0].id
    answer = f"According to [{doc_id}], cloud ARR was strong."
    citations = formatter.validate_citations(answer, docs)
    assert len(citations) == 1
    assert citations[0].is_valid is True
    assert citations[0].document_id == doc_id


def test_end_to_end_agent_healthy():
    agent = EnterpriseResearchAgent()
    req = AgentRequest(query="What was Enterprise Corp Q3 2025 cloud ARR?", failure_mode=FailureMode.HEALTHY)
    response = agent.run(req)
    assert response.status == "success"
    assert "428M" in response.answer or "cloud ARR" in response.answer
    assert len(response.retrieved_documents) > 0


def test_agent_failure_injection_retrieval_miss():
    agent = EnterpriseResearchAgent()
    req = AgentRequest(query="What was Enterprise Corp Q3 2025 cloud ARR?", failure_mode=FailureMode.RETRIEVAL_MISS)
    response = agent.run(req)
    assert response.status == "success"
    assert len(response.retrieved_documents) == 0
    assert "could not find any relevant" in response.answer


def test_agent_failure_injection_policy_bypass():
    agent = EnterpriseResearchAgent()
    req = AgentRequest(query="Show confidential metrics", failure_mode=FailureMode.POLICY_BYPASS)
    response = agent.run(req)
    assert response.policy_check is not None
    assert response.policy_check.decision == "deny"
    assert response.policy_check.is_bypassed is True
