"""TraceSleuth OpenTelemetry Instrumentation Setup.

Initializes OpenTelemetry TracerProvider, hooks up TraceSleuth exporters,
and provides context managers and span recording helpers.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Generator, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Span, StatusCode
from src.domain.trace import SpanType, TraceData
from src.telemetry.exporters import (
    DatabaseTraceExporter,
    InMemoryTraceCollector,
    JsonFileTraceExporter,
)
from src.telemetry.redaction import default_redactor


class TelemetryManager:
    """Manages the OpenTelemetry TracerProvider and TraceSleuth exporters."""

    def __init__(
        self,
        service_name: str = "tracesleuth-agent",
        export_to_file: bool = True,
        output_dir: str = "./fixtures/traces",
        export_to_db: bool = False,
        db_manager: Any = None,
    ) -> None:
        self.service_name = service_name
        self.memory_collector = InMemoryTraceCollector()
        self.provider = TracerProvider()

        # SimpleSpanProcessor exports spans synchronously as soon as they finish
        self.provider.add_span_processor(SimpleSpanProcessor(self.memory_collector))

        if export_to_file:
            self.file_exporter = JsonFileTraceExporter(
                output_dir=output_dir,
                memory_collector=self.memory_collector,
            )
            self.provider.add_span_processor(SimpleSpanProcessor(self.file_exporter))

        if export_to_db:
            from src.storage.database import get_database

            self.db_manager = db_manager or get_database()
            self.db_exporter = DatabaseTraceExporter(
                memory_collector=self.memory_collector,
                db_manager=self.db_manager,
            )
            self.provider.add_span_processor(SimpleSpanProcessor(self.db_exporter))

        self.tracer = self.provider.get_tracer(service_name)

    def get_trace(self, trace_id: str) -> Optional[TraceData]:
        """Fetch the assembled TraceData for a given trace_id."""
        return self.memory_collector.get_trace(trace_id)

    def clear(self) -> None:
        self.memory_collector.clear()

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        span_type: SpanType,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager to start an OpenTelemetry span with sanitized attributes."""
        sanitized_attrs: Dict[str, Any] = {
            "tracesleuth.span_type": span_type.value,
        }
        if attributes:
            for k, v in attributes.items():
                sanitized_attrs[k] = default_redactor.sanitize_payload(v)

        with self.tracer.start_as_current_span(name, attributes=sanitized_attrs) as span:
            try:
                yield span
                if span.is_recording():
                    span.set_status(StatusCode.OK)
            except Exception as exc:
                if span.is_recording():
                    span.set_status(StatusCode.ERROR, description=str(exc))
                    span.record_exception(exc)
                raise


# Default singleton telemetry instance
telemetry = TelemetryManager()
