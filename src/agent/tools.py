"""Agent tools: CalculatorTool and DocumentLookupTool.

Implements arithmetic evaluation, document lookup, retry mechanics,
and error handling with configurable simulated latency.
"""

from __future__ import annotations

import ast
import operator as op
import time
from typing import Any, Dict, List, Optional
from src.agent.corpus import EnterpriseCorpus
from src.agent.types import FailureMode, ToolExecutionResult
from src.telemetry.redaction import hash_content


# Safe mathematical operator map to prevent arbitrary code execution
SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def safe_eval_math(expr: str) -> float:
    """Evaluates mathematical expressions safely using Python's AST parser."""
    def _eval(node: Any) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported math operator: {op_type}")
            return SAFE_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Invalid math expression element: {type(node)}")

    # Clean expression string
    clean_expr = expr.replace("$", "").replace("%", "").replace(",", "").strip()
    tree = ast.parse(clean_expr, mode="eval")
    return _eval(tree.body)


class CalculatorTool:
    """Performs financial, arithmetic, and growth rate computations."""

    name: str = "calculator"

    def execute(
        self,
        expression: str,
        failure_mode: FailureMode = FailureMode.HEALTHY,
        retry_count: int = 0,
    ) -> ToolExecutionResult:
        start = time.perf_counter()
        args = {"expression": expression}
        args_hash = hash_content(args)

        # In failure mode TOOL_RETRY_STORM: simulate repeated failures until max retries
        if failure_mode == FailureMode.TOOL_RETRY_STORM:
            time.sleep(0.04)  # Inject simulated network/retry latency
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="error",
                error_message="Simulated upstream calculator timeout / service failure (retry storm)",
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )

        try:
            result_val = safe_eval_math(expression)
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="success",
                result=round(result_val, 4),
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="error",
                error_message=str(exc),
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )


class DocumentLookupTool:
    """Looks up documents from the enterprise knowledge corpus."""

    name: str = "document_lookup"

    def __init__(self, corpus: Optional[EnterpriseCorpus] = None) -> None:
        self.corpus = corpus or EnterpriseCorpus()

    def execute(
        self,
        document_id: str,
        failure_mode: FailureMode = FailureMode.HEALTHY,
        retry_count: int = 0,
    ) -> ToolExecutionResult:
        start = time.perf_counter()
        args = {"document_id": document_id}
        args_hash = hash_content(args)

        if failure_mode == FailureMode.TOOL_RETRY_STORM:
            time.sleep(0.03)
            elapsed = (time.perf_counter() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="error",
                error_message="Simulated document lookup service unreachable",
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )

        doc = self.corpus.get_by_id(document_id)
        elapsed = (time.perf_counter() - start) * 1000.0
        if doc:
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="success",
                result={"id": doc.id, "title": doc.title, "content": doc.content},
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )
        else:
            return ToolExecutionResult(
                tool_name=self.name,
                arguments=args,
                arguments_hash=args_hash,
                status="error",
                error_message=f"Document '{document_id}' not found in corpus",
                retry_count=retry_count,
                latency_ms=round(elapsed, 2),
            )
