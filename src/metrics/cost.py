"""Token and Model Cost Forensics (Milestone 9).

Calculates financial costs per trace and span based on GenAI token usage,
identifies cost amplification anomalies, and tracks pricing versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.domain.trace import SpanType, TraceData


@dataclass
class ModelPricing:
    """Pricing rates per million tokens and per tool invocation (in USD)."""

    input_cost_per_1m: float = 2.00
    output_cost_per_1m: float = 8.00
    tool_cost_per_1k_calls: float = 0.50
    pricing_version: str = "2026-Q1"


DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00, pricing_version="2026-Q1"),
    "claude-3-5-sonnet": ModelPricing(input_cost_per_1m=3.00, output_cost_per_1m=15.00, pricing_version="2026-Q1"),
    "gemini-1.5-pro": ModelPricing(input_cost_per_1m=1.25, output_cost_per_1m=5.00, pricing_version="2026-Q1"),
    "default": ModelPricing(input_cost_per_1m=2.00, output_cost_per_1m=8.00, pricing_version="2026-Q1"),
}


@dataclass
class TraceCostEstimate:
    """Detailed cost breakdown for a single trace execution."""

    trace_id: str
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_call_count: int
    pricing_version: str
    is_cost_anomaly: bool = False
    anomaly_reason: str = ""


class CostForensicsCalculator:
    """Calculates financial costs and identifies cost amplification anomalies."""

    def __init__(self, pricing_table: Optional[Dict[str, ModelPricing]] = None) -> None:
        self.pricing_table = pricing_table or DEFAULT_MODEL_PRICING

    def calculate_trace_cost(self, trace: TraceData) -> TraceCostEstimate:
        """Calculates token costs and tool execution costs across all spans in a trace."""
        input_tokens = 0
        output_tokens = 0
        tool_calls = 0

        model_name = "default"
        for span in trace.spans:
            if span.span_type == SpanType.MODEL:
                model_name = span.attributes.get("gen_ai.request.model", "default")
                in_tok = int(span.attributes.get("gen_ai.usage.input_tokens", 0))
                out_tok = int(span.attributes.get("gen_ai.usage.output_tokens", 0))
                input_tokens += in_tok
                output_tokens += out_tok
            elif span.span_type == SpanType.TOOL:
                tool_calls += 1

        pricing = self.pricing_table.get(model_name, self.pricing_table["default"])

        input_cost = (input_tokens / 1_000_000.0) * pricing.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000.0) * pricing.output_cost_per_1m
        tool_cost = (tool_calls / 1_000.0) * pricing.tool_cost_per_1k_calls
        total_cost = round(input_cost + output_cost + tool_cost, 6)

        total_tokens = input_tokens + output_tokens

        # Check for cost amplification anomalies (spec Section 16)
        is_anomaly = False
        reason = ""
        if total_tokens > 10_000:
            is_anomaly = True
            reason = f"High token consumption ({total_tokens} tokens)"
        elif tool_calls > 5:
            is_anomaly = True
            reason = f"Tool call loop/storm detected ({tool_calls} tool calls)"

        return TraceCostEstimate(
            trace_id=trace.trace_id,
            total_cost_usd=total_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tool_call_count=tool_calls,
            pricing_version=pricing.pricing_version,
            is_cost_anomaly=is_anomaly,
            anomaly_reason=reason,
        )
