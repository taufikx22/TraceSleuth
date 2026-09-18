"""TraceSleuth Telemetry Exporters.

Provides in-memory and JSON file storage for OpenTelemetry traces,
converting OpenTelemetry spans into TraceSleuth domain models (TraceData, SpanData).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from src.domain.trace import SpanData, SpanStatus, SpanType, TraceData, TraceEvent, TraceOutcome


def _format_timestamp(ns_timestamp: int) -> datetime:
    """Converts nanosecond epoch integer from OpenTelemetry into UTC datetime."""
    return datetime.fromtimestamp(ns_timestamp / 1e9, tz=timezone.utc)


def _convert_otel_span(span: ReadableSpan) -> SpanData:
    """Transforms an OpenTelemetry ReadableSpan into TraceSleuth SpanData."""
    trace_id_hex = format(span.context.trace_id, "032x")
    span_id_hex = format(span.context.span_id, "016x")
    parent_id_hex = (
        format(span.parent.span_id, "016x")
        if span.parent and span.parent.span_id
        else None
    )

    # Determine span type from attributes or name
    raw_type = span.attributes.get("tracesleuth.span_type") if span.attributes else None
    if raw_type:
        try:
            span_type = SpanType(raw_type)
        except ValueError:
            span_type = SpanType.INTERNAL
    else:
        name_lower = span.name.lower()
        if "model" in name_lower or "gen_ai" in name_lower or "generate" in name_lower:
            span_type = SpanType.MODEL
        elif "retriev" in name_lower or "vector" in name_lower:
            span_type = SpanType.RETRIEVAL
        elif "tool" in name_lower:
            span_type = SpanType.TOOL
        elif "policy" in name_lower:
            span_type = SpanType.POLICY
        elif "plan" in name_lower:
            span_type = SpanType.PLANNER
        elif "classif" in name_lower:
            span_type = SpanType.CLASSIFIER
        elif "citation" in name_lower:
            span_type = SpanType.CITATION
        elif "rerank" in name_lower:
            span_type = SpanType.RERANKER
        else:
            span_type = SpanType.INTERNAL

    # Format status
    status = SpanStatus.OK if span.status.is_ok else SpanStatus.ERROR
    error_msg = span.status.description if not span.status.is_ok else None

    # Parse span events
    events: List[TraceEvent] = []
    for evt in span.events:
        events.append(
            TraceEvent(
                name=evt.name,
                timestamp=_format_timestamp(evt.timestamp),
                attributes=dict(evt.attributes) if evt.attributes else {},
            )
        )

    # Clean attributes dictionary
    clean_attrs: Dict[str, Any] = {}
    if span.attributes:
        for k, v in span.attributes.items():
            # Convert non-serializable objects to string
            if isinstance(v, (str, int, float, bool)):
                clean_attrs[k] = v
            else:
                clean_attrs[k] = str(v)

    start_dt = _format_timestamp(span.start_time)
    end_dt = _format_timestamp(span.end_time) if span.end_time else start_dt
    duration_ms = max(0.0, (span.end_time - span.start_time) / 1e6) if span.end_time else 0.0

    return SpanData(
        span_id=span_id_hex,
        trace_id=trace_id_hex,
        parent_span_id=parent_id_hex,
        name=span.name,
        span_type=span_type,
        start_time=start_dt,
        end_time=end_dt,
        duration_ms=round(duration_ms, 3),
        status=status,
        error_message=error_msg,
        attributes=clean_attrs,
        events=events,
    )


class InMemoryTraceCollector(SpanExporter):
    """Collects spans in memory and groups them by trace_id into TraceData objects."""

    def __init__(self) -> None:
        self.raw_spans: List[ReadableSpan] = []
        self._traces: Dict[str, List[SpanData]] = {}

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            self.raw_spans.append(s)
            span_data = _convert_otel_span(s)
            t_id = span_data.trace_id
            if t_id not in self._traces:
                self._traces[t_id] = []
            if not any(existing.span_id == span_data.span_id for existing in self._traces[t_id]):
                self._traces[t_id].append(span_data)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.raw_spans.clear()
        self._traces.clear()

    def get_all_trace_ids(self) -> List[str]:
        return list(self._traces.keys())

    def get_trace(self, trace_id: str) -> Optional[TraceData]:
        """Assembles a full TraceData object for a specific trace_id."""
        spans = self._traces.get(trace_id)
        if not spans:
            return None

        # Find root span (span with no parent or span_type == AGENT_RUN)
        root_span = None
        for s in spans:
            if s.parent_span_id is None:
                root_span = s
                break
        if not root_span and spans:
            root_span = spans[0]

        start_time = min(s.start_time for s in spans)
        end_time = max(s.end_time for s in spans)
        total_duration = (end_time - start_time).total_seconds() * 1000.0

        # Determine outcome from spans and root attributes
        has_errors = any(s.status == SpanStatus.ERROR for s in spans)
        outcome = TraceOutcome.FAILURE if has_errors else TraceOutcome.SUCCESS

        req_id = root_span.attributes.get("tracesleuth.request_id", f"req_{trace_id[:8]}") if root_span else f"req_{trace_id[:8]}"
        session_id = root_span.attributes.get("tracesleuth.session_id") if root_span else None
        tenant_id = root_span.attributes.get("tracesleuth.tenant_id", "default_tenant") if root_span else "default_tenant"
        app_version = root_span.attributes.get("tracesleuth.app_version", "0.1.0") if root_span else "0.1.0"
        model_ver = root_span.attributes.get("tracesleuth.model_config_version", "v1.0.0") if root_span else "v1.0.0"
        prompt_ver = root_span.attributes.get("tracesleuth.prompt_version", "v1.0.0") if root_span else "v1.0.0"
        env = root_span.attributes.get("tracesleuth.environment", "development") if root_span else "development"

        return TraceData(
            trace_id=trace_id,
            request_id=req_id,
            session_id=session_id,
            tenant_id=tenant_id,
            application_version=app_version,
            model_configuration_version=model_ver,
            prompt_version=prompt_ver,
            environment=env,
            start_time=start_time,
            end_time=end_time,
            duration_ms=round(max(0.0, total_duration), 3),
            outcome=outcome,
            status="COMPLETED",
            root_span_id=root_span.span_id if root_span else None,
            spans=spans,
            metadata={"span_count": len(spans)},
        )


class JsonFileTraceExporter(SpanExporter):
    """Exports structured TraceData directly to a local JSON file fixture directory."""

    def __init__(self, output_dir: str = "./fixtures/traces", memory_collector: Optional[InMemoryTraceCollector] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = memory_collector

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        if self.collector:
            for trace_id in self.collector.get_all_trace_ids():
                trace = self.collector.get_trace(trace_id)
                if trace:
                    filepath = self.output_dir / f"{trace_id}.json"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(trace.model_dump_json(indent=2))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class DatabaseTraceExporter(SpanExporter):
    """Exports assembled TraceData to the SQLAlchemy database via IngestionService.

    Requires a memory_collector to assemble complete traces from individual spans,
    and a DatabaseManager for persistence.
    """

    def __init__(
        self,
        memory_collector: InMemoryTraceCollector,
        db_manager: Any = None,
        auto_classify: bool = True,
    ) -> None:
        self.collector = memory_collector
        self.db_manager = db_manager
        self.auto_classify = auto_classify
        self._exported_traces: set = set()

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        if not self.db_manager:
            return SpanExportResult.SUCCESS

        # Lazy import to avoid circular dependencies
        from src.ingestion.service import IngestionService

        service = IngestionService(self.db_manager)

        for trace_id in self.collector.get_all_trace_ids():
            if trace_id not in self._exported_traces:
                trace = self.collector.get_trace(trace_id)
                if trace and len(trace.spans) > 0:
                    try:
                        service.ingest_trace(trace, auto_classify=self.auto_classify)
                        self._exported_traces.add(trace_id)
                    except Exception:
                        # Don't crash the application if DB export fails
                        pass

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

