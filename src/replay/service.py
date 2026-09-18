"""TraceSleuth Replay — Execution and Comparison Engine.

Executes exact, controlled, and counterfactual replays against historical traces
and evaluates outcome changes, latency diffs, token deltas, and incident resolution.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from src.agent.corpus import EnterpriseCorpus
from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.analysis.engine import RootCauseEngine
from src.analysis.taxonomy import FailureTaxonomyClassifier
from src.domain.trace import TraceData, TraceOutcome
from src.replay.types import (
    ReplayConfig,
    ReplayMetricsDiff,
    ReplayRun,
    ReplayType,
    ReplayVerdict,
)
from src.storage.database import DatabaseManager, get_database
from src.telemetry.instrumentation import TelemetryManager


class ReplayService:
    """Orchestrates trace replay runs and before/after forensics comparisons."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or get_database()
        from src.ingestion.service import IngestionService
        self.ingestion = IngestionService(self.db)
        self.rca_engine = RootCauseEngine()

    def execute_replay(
        self,
        source_trace_id: str,
        config: Optional[ReplayConfig] = None,
    ) -> ReplayRun:
        """Executes a replay of a historical trace with specified configuration overrides."""
        config = config or ReplayConfig()
        from src.storage.repository import TraceRepository
        session = self.db.get_session()
        repo = TraceRepository(session)

        source_trace = repo.get_trace(source_trace_id)
        if not source_trace:
            session.close()
            raise ValueError(f"Source trace {source_trace_id} not found in database")

        source_incidents, _ = repo.list_incidents(trace_id=source_trace_id)
        session.close()

        # Extract original query from root span or spans
        query = "What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?"
        for span in source_trace.spans:
            if "retrieval.query" in span.attributes:
                query = span.attributes["retrieval.query"]
                break
            if "query" in span.attributes:
                query = span.attributes["query"]
                break

        started_at = datetime.now(timezone.utc)
        replay_id = f"rpl_{uuid.uuid4().hex[:12]}"

        # Configure agent with overrides
        # Determine failure mode for agent:
        # If controlled replay specifies a fix for retriever:
        failure_mode = FailureMode.HEALTHY
        if config.replay_type == ReplayType.EXACT:
            # Reconstruct original failure mode from metadata or root span attributes
            orig_mode = source_trace.metadata.get("failure_mode")
            if not orig_mode:
                for span in source_trace.spans:
                    if "agent.failure_mode" in span.attributes:
                        orig_mode = span.attributes["agent.failure_mode"]
                        break
            if orig_mode:
                try:
                    failure_mode = FailureMode(orig_mode)
                except ValueError:
                    failure_mode = FailureMode.HEALTHY
        else:
            # Controlled / counterfactual replay
            if config.retriever_mode == "miss":
                failure_mode = FailureMode.RETRIEVAL_MISS
            elif config.bypass_policy is True:
                failure_mode = FailureMode.POLICY_BYPASS
            elif config.tool_failure_rate and config.tool_failure_rate > 0.5:
                failure_mode = FailureMode.TOOL_RETRY_STORM

        # Create isolated telemetry collector for the replay
        replayed_service_name = f"tracesleuth-replay-{replay_id[:8]}"
        replayed_telemetry = TelemetryManager(
            service_name=replayed_service_name,
            export_to_file=False,
            export_to_db=False,
        )

        agent = InstrumentedResearchAgent(
            telemetry_manager=replayed_telemetry,
            app_version=source_trace.application_version,
            model_config_version=config.model_name or source_trace.model_configuration_version,
            prompt_version=config.prompt_version or source_trace.prompt_version,
            environment=f"replay-{source_trace.environment}",
        )

        req = AgentRequest(
            query=query,
            session_id=source_trace.session_id,
            tenant_id=source_trace.tenant_id,
            failure_mode=failure_mode,
        )

        # Run agent
        t0 = time.perf_counter()
        agent_response, replayed_trace = agent.run(req)
        wall_duration_ms = (time.perf_counter() - t0) * 1000.0

        completed_at = datetime.now(timezone.utc)

        # Ingest and classify replayed trace
        replayed_incidents = []
        if replayed_trace:
            replayed_trace.metadata["replay_of"] = source_trace_id
            replayed_trace.metadata["replay_id"] = replay_id
            replayed_incidents = self.ingestion.ingest_trace(replayed_trace, auto_classify=True)

        # Compare metrics and determine verdict
        metrics_diff = self.compare_traces(source_trace, replayed_trace, source_incidents, replayed_incidents)
        verdict = self._determine_verdict(source_trace, replayed_trace, source_incidents, replayed_incidents, config)

        summary_parts = [f"Replay verdict: {verdict.value.upper()}."]
        if metrics_diff.resolved_incident_categories:
            summary_parts.append(f"Resolved categories: {', '.join(metrics_diff.resolved_incident_categories)}.")
        if metrics_diff.new_incident_categories:
            summary_parts.append(f"New categories: {', '.join(metrics_diff.new_incident_categories)}.")
        summary_parts.append(f"Latency delta: {metrics_diff.duration_diff_ms:+.1f}ms, Token delta: {metrics_diff.tokens_diff:+d}.")

        replay_run = ReplayRun(
            replay_id=replay_id,
            source_trace_id=source_trace_id,
            replayed_trace_id=replayed_trace.trace_id if replayed_trace else None,
            replay_type=config.replay_type,
            config=config,
            verdict=verdict,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=wall_duration_ms,
            metrics_diff=metrics_diff,
            summary=" ".join(summary_parts),
            metadata={
                "agent_status": agent_response.status,
                "agent_answer": agent_response.answer[:200] if agent_response else "",
            },
        )

        # Persist replay record
        save_session = self.db.get_session()
        save_repo = TraceRepository(save_session)
        save_repo.save_replay(replay_run)
        save_session.close()

        return replay_run

    def compare_traces(
        self,
        original: TraceData,
        replayed: Optional[TraceData],
        original_incidents: list,
        replayed_incidents: list,
    ) -> ReplayMetricsDiff:
        """Computes differences in latency, token usage, spans, and incident status."""
        replayed_dur = replayed.duration_ms if replayed else 0.0
        replayed_tok = replayed.total_tokens() if replayed else 0
        replayed_spans = len(replayed.spans) if replayed else 0
        replayed_out = replayed.outcome.value if replayed else "failed"

        orig_dur = original.duration_ms
        orig_tok = original.total_tokens()
        orig_spans = len(original.spans)
        orig_out = original.outcome.value

        orig_cats = {inc.category.value for inc in original_incidents}
        replayed_cats = {inc.category.value for inc in replayed_incidents}

        resolved = sorted(orig_cats - replayed_cats)
        new_cats = sorted(replayed_cats - orig_cats)

        return ReplayMetricsDiff(
            original_duration_ms=orig_dur,
            replayed_duration_ms=replayed_dur,
            duration_diff_ms=round(replayed_dur - orig_dur, 2),
            original_tokens=orig_tok,
            replayed_tokens=replayed_tok,
            tokens_diff=replayed_tok - orig_tok,
            original_span_count=orig_spans,
            replayed_span_count=replayed_spans,
            original_outcome=orig_out,
            replayed_outcome=replayed_out,
            resolved_incident_categories=resolved,
            new_incident_categories=new_cats,
        )

    def _determine_verdict(
        self,
        original: TraceData,
        replayed: Optional[TraceData],
        original_incidents: list,
        replayed_incidents: list,
        config: ReplayConfig,
    ) -> ReplayVerdict:
        """Evaluates whether the replay fixed the problem, regressed, or reproduced identically."""
        if not replayed:
            return ReplayVerdict.STILL_FAILING

        orig_had_failures = (original.outcome != TraceOutcome.SUCCESS) or len(original_incidents) > 0
        replayed_has_failures = (replayed.outcome != TraceOutcome.SUCCESS) or len(replayed_incidents) > 0

        if config.replay_type == ReplayType.EXACT:
            if orig_had_failures == replayed_has_failures:
                return ReplayVerdict.IDENTICAL
            return ReplayVerdict.INCONCLUSIVE

        # Controlled / counterfactual replay
        if orig_had_failures and not replayed_has_failures:
            return ReplayVerdict.FIXED

        orig_cats = {inc.category.value for inc in original_incidents}
        replayed_cats = {inc.category.value for inc in replayed_incidents}

        if replayed_cats - orig_cats:
            return ReplayVerdict.REGRESSED

        if replayed_cats and (replayed_cats == orig_cats or bool(replayed_cats.intersection(orig_cats))):
            return ReplayVerdict.STILL_FAILING

        return ReplayVerdict.IDENTICAL if orig_had_failures == replayed_has_failures else ReplayVerdict.INCONCLUSIVE
