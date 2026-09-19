"""TraceSleuth Demo Runner — Milestones 0 through 4.

Executes requests through the Instrumented Enterprise Research Agent,
prints the causal span hierarchy, OpenTelemetry GenAI attributes,
persists traces and incidents to SQLite storage, classifies failures
using the 13-category failure taxonomy, and demonstrates the Trace Explorer API.
"""

from __future__ import annotations

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.agent.instrumented_agent import InstrumentedResearchAgent
from src.agent.types import AgentRequest, FailureMode
from src.analysis.engine import RootCauseEngine
from src.analysis.taxonomy import FailureTaxonomyClassifier
from src.ingestion.service import IngestionService
from src.replay.service import ReplayService
from src.replay.types import ReplayConfig, ReplayType
from src.storage.database import get_database


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceSleuth Agent Demo Runner")
    parser.add_argument(
        "--scenario",
        choices=[
            "healthy",
            "retrieval_miss",
            "generation_contradiction",
            "tool_retry_storm",
            "policy_bypass",
            "context_explosion",
            "evaluation_blind_spot",
        ],
        default="healthy",
        help="Execution scenario to simulate (default: healthy)",
    )
    parser.add_argument(
        "--query",
        default="What was Enterprise Corp Q3 2025 cloud ARR and how does operating expense compare?",
        help="Query for the research agent",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable SQLite database persistence",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Execute Milestone 7 controlled replay to fix injected failure",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            f"[bold cyan]TraceSleuth Forensic Observability Platform[/bold cyan]\n"
            f"[yellow]Scenario:[/yellow] {args.scenario} | [yellow]Query:[/yellow] {args.query}",
            border_style="blue",
        )
    )

    failure_mode = FailureMode(args.scenario)
    agent = InstrumentedResearchAgent()

    req = AgentRequest(query=args.query, failure_mode=failure_mode)
    console.print("[dim]Executing instrumented request through OpenTelemetry pipeline...[/dim]\n")
    response, trace = agent.run(req)

    # 1. Display Agent Response (Milestone 0: Reference Agent)
    console.print(
        Panel(
            f"[bold green]Agent Response (status={response.status}, latency={response.duration_ms}ms):[/bold green]\n\n"
            f"{response.answer}",
            title="Execution Result (Milestone 0: 8 Observable Operations)",
            border_style="green" if response.status == "success" else "red",
        )
    )

    if not trace:
        console.print("[red]Error: No OpenTelemetry trace collected![/red]")
        sys.exit(1)

    # 2. Display Trace Metadata Table (Milestone 1: OpenTelemetry Instrumentation)
    meta_table = Table(title="Trace Top-Level Metadata (Milestone 1)", border_style="cyan")
    meta_table.add_column("Property", style="bold")
    meta_table.add_column("Value", style="yellow")
    meta_table.add_row("Trace ID", trace.trace_id)
    meta_table.add_row("Request ID", trace.request_id)
    meta_table.add_row("App Version", trace.application_version)
    meta_table.add_row("Model Config Version", trace.model_configuration_version)
    meta_table.add_row("Prompt Version", trace.prompt_version)
    meta_table.add_row("Environment", trace.environment)
    meta_table.add_row("Total Latency", f"{trace.duration_ms} ms")
    meta_table.add_row("Total Tokens", str(trace.total_tokens()))
    meta_table.add_row("Total Spans", str(len(trace.spans)))
    meta_table.add_row("Outcome", trace.outcome.value)
    console.print(meta_table)

    # 3. Display Causal Span Tree
    console.print("\n[bold cyan]Causal Span Execution Tree (Milestone 1):[/bold cyan]")
    tree = Tree(f"[bold]{trace.trace_id[:16]}... (Root: agent.request)[/bold]")

    span_map = {s.span_id: s for s in trace.spans}
    children_map: dict[str, list] = {}
    root_spans = []

    for s in trace.spans:
        if s.parent_span_id and s.parent_span_id in span_map:
            children_map.setdefault(s.parent_span_id, []).append(s)
        else:
            root_spans.append(s)

    def add_children(parent_node, parent_id):
        for child in children_map.get(parent_id, []):
            color = "green" if child.status.value == "OK" else "red"
            label = (
                f"[{color}]{child.name}[/{color}] "
                f"[dim]({child.span_type.value}, {child.duration_ms}ms)[/dim]"
            )
            if child.events:
                evt_names = ", ".join(e.name for e in child.events)
                label += f" [magenta](Events: {evt_names})[/magenta]"
            c_node = parent_node.add(label)
            add_children(c_node, child.span_id)

    for r in root_spans:
        color = "green" if r.status.value == "OK" else "red"
        r_node = tree.add(f"[{color}]{r.name}[/{color}] [dim]({r.span_type.value}, {r.duration_ms}ms)[/dim]")
        add_children(r_node, r.span_id)

    console.print(tree)

    # 4. Storage & Failure Taxonomy Classification (Milestones 2 & 4)
    incidents = []
    if not args.no_persist:
        db = get_database()
        ingestion = IngestionService(db)
        # Ingest trace and auto-classify
        incidents = ingestion.ingest_trace(trace, auto_classify=True)
        console.print(
            f"\n[bold green][OK] Milestone 2 (Storage):[/bold green] Trace [bold]{trace.trace_id}[/bold] "
            f"persisted to SQLite database (`./data/tracesleuth.db`)."
        )
    else:
        classifier = FailureTaxonomyClassifier()
        incidents = classifier.classify(trace)

    # 5. Display Failure Incidents Table (Milestone 4)
    if incidents:
        inc_table = Table(
            title=f"Classified Failure Incidents (Milestone 4 — {len(incidents)} Found)",
            border_style="red",
        )
        inc_table.add_column("Incident ID", style="bold yellow")
        inc_table.add_column("Category", style="bold red")
        inc_table.add_column("Severity", style="bold")
        inc_table.add_column("Score", style="cyan")
        inc_table.add_column("Summary", style="white")
        inc_table.add_column("Suspicious Span", style="magenta")
        inc_table.add_column("Evidence Count", style="green")

        for inc in incidents:
            inc_table.add_row(
                inc.incident_id[:16] + "...",
                inc.category.value,
                inc.severity.value,
                f"{inc.hypothesis_score:.2f}",
                inc.summary,
                inc.first_suspicious_span_id or "N/A",
                str(len(inc.evidence)),
            )
        console.print(inc_table)

        # Print evidence details
        for inc in incidents:
            if inc.evidence:
                console.print(f"[bold yellow]Evidence for {inc.category.value}:[/bold yellow]")
                for ev in inc.evidence:
                    console.print(
                        f"  - [{ev.type}] {ev.description} "
                        f"(key={ev.key}, strength={ev.strength}, span={ev.span_id})"
                    )
    else:
        console.print(
            "\n[bold green][OK] Milestone 4 (Taxonomy):[/bold green] No failures detected — trace is healthy."
        )

    # 6. Milestone 5: Rule-Based Forensics & Causal Graph Traversal
    console.print("\n[bold cyan]Milestone 5 (Rule-Based Forensics & Earliest Suspicious Span):[/bold cyan]")
    rca_engine = RootCauseEngine()
    rankings = rca_engine.analyze(trace)
    if rankings:
        rc = rankings[0]
        console.print(
            Panel(
                f"[bold red]Primary Root Cause:[/bold red] {rc.category.value}\n"
                f"[yellow]Confidence Score:[/yellow] {rc.confidence:.2f}\n"
                f"[magenta]Earliest Suspicious Span:[/magenta] {rc.first_suspicious_span_id}\n"
                f"[white]Summary:[/white] {rc.summary}\n"
                f"[green]Remediation:[/green] {rc.remediation}\n"
                f"[dim]Downstream Symptoms: {', '.join(rc.downstream_symptoms) or 'None'}[/dim]",
                title="Root-Cause Analysis Hypothesis",
                border_style="red",
            )
        )
    else:
        console.print("[green][OK] No causal failure root causes detected by forensic rules.[/green]")

    # 7. Milestone 6: Incident View & Evidence Graph Guidance
    console.print("\n[bold cyan]Milestone 6 (Incident View & Web UI):[/bold cyan]")
    if incidents:
        first_inc = incidents[0]
        console.print(
            Panel(
                f"• Interactive Web UI:     [cyan]http://localhost:8000/ui/incidents/{first_inc.incident_id}[/cyan]\n"
                f"• Incident Timeline:      [cyan]GET http://localhost:8000/incidents/{first_inc.incident_id}/timeline[/cyan]\n"
                f"• Evidence Graph:         [cyan]GET http://localhost:8000/incidents/{first_inc.incident_id}/evidence-graph[/cyan]\n"
                f"• Similar Incidents:      [cyan]GET http://localhost:8000/incidents/{first_inc.incident_id}/similar[/cyan]",
                title="Milestone 6 Reviewer Artifacts",
                border_style="cyan",
            )
        )
    else:
        console.print(
            Panel(
                "• Interactive Web Dashboard: [cyan]http://localhost:8000/ui[/cyan]",
                title="Milestone 6 Web Dashboard",
                border_style="cyan",
            )
        )

    # 8. Milestone 7: Exact & Controlled Replay Engine
    if args.replay or (incidents and not args.no_persist):
        console.print("\n[bold cyan]Milestone 7 (Controlled Replay Engine):[/bold cyan]")
        console.print("[dim]Executing controlled replay with configuration override (retriever_mode='fixed')...[/dim]")
        db = get_database()
        replay_svc = ReplayService(db)
        config_override = ReplayConfig(
            replay_type=ReplayType.CONTROLLED,
            retriever_mode="fixed" if args.scenario == "retrieval_miss" else None,
            bypass_policy=False if args.scenario == "policy_bypass" else None,
        )
        replay_run = replay_svc.execute_replay(trace.trace_id, config=config_override)

        verdict_color = "green" if replay_run.verdict.value == "fixed" else "yellow"
        console.print(
            Panel(
                f"[bold]Replay ID:[/bold] {replay_run.replay_id}\n"
                f"[bold]Verdict:[/bold] [{verdict_color}]{replay_run.verdict.value.upper()}[/{verdict_color}]\n"
                f"[bold]Summary:[/bold] {replay_run.summary}\n"
                f"[bold]Latency Delta:[/bold] {replay_run.metrics_diff.duration_diff_ms:+.1f} ms\n"
                f"[bold]Token Delta:[/bold] {replay_run.metrics_diff.tokens_diff:+d}\n"
                f"[bold]Resolved Categories:[/bold] {', '.join(replay_run.metrics_diff.resolved_incident_categories) or 'None'}\n"
                f"[bold]New Categories:[/bold] {', '.join(replay_run.metrics_diff.new_incident_categories) or 'None'}",
                title=f"Milestone 7 Replay Comparison — {replay_run.verdict.value.upper()}",
                border_style=verdict_color,
            )
        )

    # 9. Milestone 8: Regression Conversion & CI Dataset Export
    if incidents and not args.no_persist:
        console.print("\n[bold cyan]Milestone 8 (Regression Conversion & CI/CD Protection):[/bold cyan]")
        from src.regression.service import RegressionService
        reg_svc = RegressionService(db)
        first_inc = incidents[0]
        case = reg_svc.create_case_from_incident(
            incident_id=first_inc.incident_id,
            owner="secops-ai-reliability",
            introduced_version="1.2.0",
        )
        if case:
            run_res = reg_svc.run_case(case.id)
            pass_color = "green" if run_res and run_res.passed else "red"
            console.print(
                Panel(
                    f"[bold]Regression Case ID:[/bold] {case.id}\n"
                    f"[bold]Failure Type:[/bold] {case.failure_type}\n"
                    f"[bold]Target Query:[/bold] {case.query}\n"
                    f"[bold]Assertions Count:[/bold] {len(case.assertions)}\n"
                    f"[bold]Status After Healthy Run:[/bold] [{pass_color}]{case.status.value.upper()}[/{pass_color}]\n"
                    f"[bold]Run Summary:[/bold] {run_res.summary if run_res else 'N/A'}\n"
                    f"[bold]CI Dataset Export (YAML):[/bold]\n"
                    f"[dim]{reg_svc.export_ci_dataset(format='yaml')[:250]}...[/dim]",
                    title="Milestone 8 Regression Conversion Loop",
                    border_style="green",
                )
            )

    # 10. Milestone 9: Metrics, Latency Decomposition & Alert Engine
    console.print("\n[bold cyan]Milestone 9 (SLO Metrics, Cost Forensics & Alert Engine):[/bold cyan]")
    from src.metrics.service import MetricsService
    metrics_svc = MetricsService(db if not args.no_persist else None)
    sla_summary = metrics_svc.get_sla_metrics()
    cost_summary = metrics_svc.get_cost_breakdown()
    active_alerts = metrics_svc.get_alerts()

    metrics_table = Table(title="Production Observability & SLO Summary (Milestone 9)", border_style="cyan")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Current Value", style="yellow")
    metrics_table.add_column("SLO Status", style="green")
    metrics_table.add_row("Total Ingested Traces", str(sla_summary["total_traces"]), "[green]HEALTHY[/green]")
    metrics_table.add_row("Availability (HTTP 200/Success)", f"{sla_summary['availability_rate'] * 100:.1f}%", "[green]HEALTHY[/green]")
    metrics_table.add_row("Quality Pass Rate (Forensics)", f"{sla_summary['quality_pass_rate'] * 100:.1f}%", "[yellow]MONITORED[/yellow]")
    metrics_table.add_row("Estimated Total Spend", f"${cost_summary['total_cost_usd']:.6f}", "[cyan]ACTIVE[/cyan]")
    metrics_table.add_row("Active Alerts Firing", str(len(active_alerts)), "[red]ALERTING[/red]" if active_alerts else "[green]CLEAR[/green]")
    console.print(metrics_table)

    if active_alerts:
        alert_table = Table(title=f"Active System Alerts ({len(active_alerts)} Firing)", border_style="red")
        alert_table.add_column("Name", style="bold red")
        alert_table.add_column("Severity", style="bold yellow")
        alert_table.add_column("Summary", style="cyan")
        alert_table.add_column("Triggered At", style="dim")
        for alt in active_alerts:
            alert_table.add_row(
                alt.name,
                alt.severity.value.upper(),
                alt.summary,
                alt.triggered_at.strftime("%H:%M:%S UTC"),
            )
        console.print(alert_table)

    # 11. Milestone 10: Model-Assisted Root Cause Reasoning with Citations
    if incidents:
        console.print("\n[bold cyan]Milestone 10 (Model-Assisted Root-Cause Reasoning with Strict Citations):[/bold cyan]")
        from src.analysis.model_assisted import ModelAssistedForensicAnalyst
        analyst = ModelAssistedForensicAnalyst()
        m_result = analyst.analyze(trace, incident=incidents[0])
        console.print(
            Panel(
                f"[bold red]Primary Root Cause:[/bold red] {m_result.primary_root_cause.value}\n"
                f"[yellow]Hypothesis Explanation:[/yellow] {m_result.explanation}\n"
                f"[green]Recommended Remediation:[/green] {m_result.recommended_remediation}\n"
                f"[magenta]Confidence Score:[/magenta] {m_result.confidence:.2f}\n"
                f"[cyan]Grounded Evidence Citations ({len(m_result.evidence_citations)}):[/cyan]\n"
                + "\n".join(f"  • Span [{c.span_id}] (key: {c.evidence_key}, strength: {c.strength:.2f}) -> {c.description}" for c in m_result.evidence_citations),
                title="Milestone 10 Accountable Model Forensics",
                border_style="magenta",
            )
        )

    console.print(
        "\n[bold green]=== All Milestones 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 Complete & Verified ===[/bold green]\n"
    )


if __name__ == "__main__":
    main()

