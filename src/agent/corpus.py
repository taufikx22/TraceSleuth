"""Enterprise Knowledge Corpus and Document Store.

Provides realistic corporate knowledge documents across financial,
technical, HR, and product domains, with vector/token similarity retrieval.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional
from src.agent.types import DocumentChunk


ENTERPRISE_DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "doc_fin_2025_q3",
        "title": "Q3 2025 Financial Performance Report",
        "category": "financial",
        "source": "ir.enterprise.internal/reports/2025-q3",
        "content": (
            "Enterprise Corp Q3 2025 Financial Summary: Total cloud ARR reached $428M, representing "
            "a 34% year-over-year increase. Operating expenses were $184M, down 6% from Q2. "
            "Gross margin expanded to 78.4%. Net income for the quarter was $62.5M. Capital expenditure "
            "on AI GPU compute infrastructure was $45M."
        ),
    },
    {
        "id": "doc_fin_2025_capex",
        "title": "FY2025 Infrastructure Capital Budget Breakdown",
        "category": "financial",
        "source": "finance.enterprise.internal/capex-2025",
        "content": (
            "FY2025 CapEx Allocation: Total infrastructure budget approved is $180M. $95M is allocated "
            "to GPU cluster expansion in us-east-2, $50M to datacenter networking and fiber backhaul, "
            "and $35M for edge caching and storage redundancy."
        ),
    },
    {
        "id": "doc_tech_sla_2026",
        "title": "Platform Reliability & API Service Level Objectives",
        "category": "technical",
        "source": "eng.enterprise.internal/handbook/sla-slo",
        "content": (
            "Platform Engineering Service Level Objectives (SLO 2026): Enterprise API tier targets 99.95% availability. "
            "p95 latency threshold is 350ms for synchronous queries. Rate limits for Tier 1 enterprise customers are "
            "capped at 2,000 requests per minute with a burst allowance of 500 requests over 10 seconds. "
            "RTO (Recovery Time Objective) is 15 minutes, and RPO (Recovery Point Objective) is 1 minute."
        ),
    },
    {
        "id": "doc_tech_security_auth",
        "title": "Enterprise Security, Authentication & Zero Trust Protocol",
        "category": "technical",
        "source": "security.enterprise.internal/policies/zero-trust",
        "content": (
            "Security Protocol 4.2: All internal services require mTLS and short-lived JWT tokens (15m expiry). "
            "Access to financial databases and customer PII requires level-3 multi-factor approval and audit logging. "
            "Direct raw prompt inspection in production is strictly forbidden without automated PII redaction."
        ),
    },
    {
        "id": "doc_hr_pto_policy",
        "title": "Employee Time Off, Sabbatical & PTO Carryover Rules",
        "category": "hr",
        "source": "people.enterprise.internal/benefits/pto",
        "content": (
            "Global PTO Policy: Full-time employees accrue 25 days of paid time off annually. A maximum of 5 unused "
            "PTO days can be carried over into the next calendar year, expiring on March 31. Parental leave is 16 weeks "
            "fully paid for primary caregivers and 8 weeks for secondary caregivers."
        ),
    },
    {
        "id": "doc_prod_tier_matrix",
        "title": "Enterprise Tier Feature Matrix & Compliance Specifications",
        "category": "product",
        "source": "product.enterprise.internal/tiers/matrix",
        "content": (
            "Product Tier Comparison: The Enterprise Tier includes dedicated tenant VPC isolation, custom domain SSO "
            "(SAML 2.0 / OIDC), SOC2 Type II compliance reports, HIPAA BAA agreements, and regional data residency "
            "in EU, US, and APAC regions. Annual base subscription begins at $120,000/year."
        ),
    },
]


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenization for keyword scoring."""
    return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())


class EnterpriseCorpus:
    """In-memory enterprise corpus with cosine-like scoring and document indexing."""

    def __init__(self, documents: Optional[List[Dict[str, str]]] = None) -> None:
        self.raw_docs = documents or ENTERPRISE_DOCUMENTS
        self.docs: List[DocumentChunk] = [
            DocumentChunk(
                id=d["id"],
                title=d["title"],
                content=d["content"],
                category=d["category"],
                source=d["source"],
                score=0.0,
                metadata={"length": len(d["content"])},
            )
            for d in self.raw_docs
        ]
        self._index: Dict[str, DocumentChunk] = {d.id: d for d in self.docs}

    def get_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """Fetch document chunk by unique ID."""
        chunk = self._index.get(doc_id)
        if chunk:
            return chunk.model_copy()
        return None

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[DocumentChunk]:
        """Performs token-overlap vector-like similarity search with term frequency weighting."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored_chunks: List[DocumentChunk] = []
        for doc in self.docs:
            if category and doc.category != category:
                continue

            doc_tokens = _tokenize(doc.title + " " + doc.content)
            if not doc_tokens:
                continue

            # Token overlap & frequency calculation
            matches = sum(1 for qt in query_tokens if qt in doc_tokens)
            score = matches / math.sqrt(len(query_tokens) * len(doc_tokens))

            # Bonus for title keyword match
            title_tokens = _tokenize(doc.title)
            title_matches = sum(1 for qt in query_tokens if qt in title_tokens)
            score += title_matches * 0.15

            score = min(1.0, round(score, 4))
            if score >= min_score:
                chunk_copy = doc.model_copy()
                chunk_copy.score = score
                scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda c: c.score, reverse=True)
        return scored_chunks[:top_k]
