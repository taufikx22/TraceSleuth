"""TraceSleuth Replay — Domain Models and Types.

Defines schemas and enums for exact, controlled, and counterfactual replay
as specified in spec Section 21, 25, and 32.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReplayType(str, Enum):
    """Classification of replay execution (spec Section 32)."""

    EXACT = "exact"  # Same inputs, same configuration
    CONTROLLED = "controlled"  # Same input, single variable modified
    COUNTERFACTUAL = "counterfactual"  # Testing a specific fix hypothesis


class ReplayVerdict(str, Enum):
    """Outcome assessment of a replay run compared to the source trace."""

    FIXED = "fixed"  # Failure mode eliminated; request succeeded
    STILL_FAILING = "still_failing"  # Same failure mode persisted
    REGRESSED = "regressed"  # Introduced a new or worse failure
    IDENTICAL = "identical"  # Output and behavior reproduced exactly
    INCONCLUSIVE = "inconclusive"


class ReplayConfig(BaseModel):
    """Configuration overrides to apply during a controlled or counterfactual replay."""

    # Component overrides
    retriever_mode: Optional[str] = None  # e.g. "fixed", "default", "dense_expanded"
    vector_threshold: Optional[float] = None  # custom similarity threshold
    model_name: Optional[str] = None  # alternative LLM model
    prompt_version: Optional[str] = None  # updated prompt version
    bypass_policy: Optional[bool] = None  # override policy bypass
    tool_failure_rate: Optional[float] = None  # override tool failure rate
    
    # Execution options
    replay_type: ReplayType = ReplayType.CONTROLLED
    dry_run: bool = False
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class ReplayMetricsDiff(BaseModel):
    """Quantitative comparison between source trace and replayed trace."""

    original_duration_ms: float
    replayed_duration_ms: float
    duration_diff_ms: float
    
    original_tokens: int
    replayed_tokens: int
    tokens_diff: int
    
    original_span_count: int
    replayed_span_count: int
    
    original_outcome: str
    replayed_outcome: str
    
    resolved_incident_categories: List[str] = Field(default_factory=list)
    new_incident_categories: List[str] = Field(default_factory=list)


class ReplayRun(BaseModel):
    """A persistent record of a historical trace replay."""

    replay_id: str
    source_trace_id: str
    replayed_trace_id: Optional[str] = None
    replay_type: ReplayType = ReplayType.CONTROLLED
    config: ReplayConfig = Field(default_factory=ReplayConfig)
    verdict: ReplayVerdict = ReplayVerdict.INCONCLUSIVE
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    metrics_diff: Optional[ReplayMetricsDiff] = None
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
