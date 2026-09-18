"""TraceSleuth UI — Incident Review & Forensic Dashboard.

Delivers an interactive, web-based incident review experience allowing reviewers
to inspect timelines, examine evidence chains, explore similar incidents,
and trigger controlled replays (spec Section 14, 28, and Milestone 6 exit criteria).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from src.storage.database import get_database
from src.storage.repository import TraceRepository

router = APIRouter(prefix="/ui", tags=["ui"])


def _get_repo() -> TraceRepository:
    return TraceRepository(get_database().get_session())


_CSS_STYLES = """
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-card: #21262d;
  --text-primary: #f0f6fc;
  --text-secondary: #8b949e;
  --accent-blue: #58a6ff;
  --accent-green: #3fb950;
  --accent-red: #f85149;
  --accent-amber: #d29922;
  --accent-purple: #bc8cff;
  --border-color: #30363d;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  padding: 24px;
}
.container { max-width: 1200px; margin: 0 auto; }
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
h1 { font-size: 24px; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.logo-badge { background: #1f6feb; color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 14px; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}
.badge-p0 { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
.badge-p1 { background: rgba(210, 153, 34, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
.badge-p2 { background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
.badge-p3 { background: rgba(139, 148, 158, 0.2); color: var(--text-secondary); border: 1px solid var(--border-color); }
.badge-category { background: rgba(188, 140, 255, 0.15); color: var(--accent-purple); border: 1px solid var(--accent-purple); }

.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}
.card-header {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.alert-suspicious {
  background: rgba(210, 153, 34, 0.1);
  border-left: 4px solid var(--accent-amber);
  padding: 14px 18px;
  border-radius: 4px;
  margin-bottom: 20px;
}

/* Timeline */
.timeline { position: relative; padding-left: 28px; }
.timeline::before {
  content: '';
  position: absolute;
  left: 8px;
  top: 6px;
  bottom: 6px;
  width: 2px;
  background: var(--border-color);
}
.timeline-item {
  position: relative;
  margin-bottom: 16px;
}
.timeline-item::before {
  content: '';
  position: absolute;
  left: -24px;
  top: 5px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent-blue);
  border: 2px solid var(--bg-primary);
}
.timeline-item.suspicious::before {
  background: var(--accent-red);
  box-shadow: 0 0 8px var(--accent-red);
}
.timeline-item.suspicious {
  background: rgba(248, 81, 73, 0.08);
  padding: 10px 14px;
  border-radius: 6px;
  border: 1px solid rgba(248, 81, 73, 0.3);
}
.timeline-time { font-family: monospace; font-size: 12px; color: var(--text-secondary); }
.timeline-title { font-weight: 600; font-size: 14px; }
.timeline-details { font-size: 13px; color: var(--text-secondary); margin-top: 4px; font-family: monospace; }

/* Evidence Grid */
.evidence-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
.evidence-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 14px;
}
.evidence-type { font-size: 12px; color: var(--accent-blue); font-weight: 600; text-transform: uppercase; }
.evidence-desc { font-size: 14px; margin: 8px 0; color: var(--text-primary); }
.evidence-meta { font-size: 12px; color: var(--text-secondary); font-family: monospace; }

/* Similar incidents table */
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border-color); font-size: 13px; }
th { color: var(--text-secondary); font-weight: 600; }
a { color: var(--accent-blue); text-decoration: none; }
a:hover { text-decoration: underline; }

.btn {
  display: inline-block;
  background: #238636;
  color: #fff;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 13px;
  border: none;
  cursor: pointer;
}
.btn:hover { background: #2ea043; text-decoration: none; }
"""


@router.get("/incidents/{incident_id}", response_class=HTMLResponse)
def incident_view(incident_id: str) -> str:
    """Renders the comprehensive Incident Review page (Milestone 6 Exit Criterion)."""
    repo = _get_repo()
    incident = repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    trace = repo.get_trace(incident.trace_id)
    similar_incidents = repo.find_similar_incidents(incident_id, limit=5)

    # Timeline reconstruction
    timeline_entries = []
    if trace:
        suspicious_span_id = incident.first_suspicious_span_id
        for span in trace.spans:
            is_susp = (span.span_id == suspicious_span_id)
            timeline_entries.append({
                "time": span.start_time.strftime("%H:%M:%S.%f")[:-3],
                "title": f"Span Started: {span.name}",
                "type": span.span_type.value,
                "is_suspicious": is_susp,
                "details": f"span_id={span.span_id}, parent={span.parent_span_id or 'none'}",
            })
            for ev in span.events:
                timeline_entries.append({
                    "time": ev.timestamp.strftime("%H:%M:%S.%f")[:-3],
                    "title": f"Event: {ev.name}",
                    "type": "event",
                    "is_suspicious": is_susp or (ev.name in ("retrieval_empty", "policy_bypass", "retry")),
                    "details": str(ev.attributes),
                })
            timeline_entries.append({
                "time": span.end_time.strftime("%H:%M:%S.%f")[:-3],
                "title": f"Span Completed: {span.name}",
                "type": span.span_type.value,
                "is_suspicious": is_susp,
                "details": f"duration={span.duration_ms:.1f}ms, status={span.status.value}",
            })
        timeline_entries.sort(key=lambda x: x["time"])

    # Build HTML components
    sev_class = f"badge-{incident.severity.value.lower()}"
    remediation = incident.metadata.get("remediation", "Review pipeline stage parameters.")
    first_suspicious_text = (
        f"<strong>First Suspicious Span:</strong> <code>{incident.first_suspicious_span_id}</code>"
        if incident.first_suspicious_span_id
        else "No specific suspicious span isolated."
    )

    # Evidence items HTML
    evidence_html = ""
    for ev in incident.evidence:
        evidence_html += f"""
        <div class="evidence-card">
          <div class="evidence-type">{ev.type} (strength: {ev.strength:.2f})</div>
          <div class="evidence-desc">{ev.description}</div>
          <div class="evidence-meta">key: {ev.key} | span: {ev.span_id or 'root'}</div>
        </div>
        """

    # Timeline HTML
    timeline_html = ""
    for item in timeline_entries:
        susp_cls = "suspicious" if item["is_suspicious"] else ""
        badge = " <span class='badge badge-p0'>First Suspicious Event</span>" if item["is_suspicious"] else ""
        timeline_html += f"""
        <div class="timeline-item {susp_cls}">
          <div class="timeline-time">{item['time']}</div>
          <div class="timeline-title">{item['title']}{badge}</div>
          <div class="timeline-details">{item['details']}</div>
        </div>
        """

    # Similar incidents HTML
    similar_html = ""
    if similar_incidents:
        for other, sim_score in similar_incidents:
            similar_html += f"""
            <tr>
              <td><a href="/ui/incidents/{other.incident_id}">{other.incident_id[:16]}...</a></td>
              <td><span class="badge badge-category">{other.category.value}</span></td>
              <td><span class="badge badge-{other.severity.value.lower()}">{other.severity.value}</span></td>
              <td><strong>{int(sim_score * 100)}%</strong></td>
              <td>{other.summary[:80]}</td>
            </tr>
            """
    else:
        similar_html = "<tr><td colspan='5' style='color: var(--text-secondary); text-align: center;'>No other incidents match this failure signature yet.</td></tr>"

    # Milestone 10: Model-Assisted Forensic Analysis
    model_hypo_html = ""
    if trace:
        from src.analysis.model_assisted import ModelAssistedForensicAnalyst
        analyst = ModelAssistedForensicAnalyst()
        m_hypo = analyst.analyze(trace, incident=incident)
        citations_html = "".join(
            f"""<span class="badge" style="background: rgba(188,140,255,0.15); color: var(--accent-purple); border: 1px solid var(--accent-purple); margin-right: 6px; margin-bottom: 4px;" title="{c.description}">
               📌 <code>{c.span_id[:8]}...</code> ({c.evidence_key})
            </span>"""
            for c in m_hypo.evidence_citations
        ) or "<span style='color: var(--text-secondary); font-size: 12px;'>No explicit span citations required.</span>"

        model_hypo_html = f"""
        <div class="card" style="border-left: 4px solid var(--accent-purple); background: rgba(188,140,255,0.04);">
          <div class="card-header">
            <span style="display: flex; align-items: center; gap: 8px;">
              <span>🤖 Model-Assisted Forensic Analysis (Milestone 10)</span>
            </span>
            <span style="font-size: 12px; color: var(--accent-purple); font-weight: normal;">
              Engine: <code>{m_hypo.engine}</code> | Grounded Confidence: <strong>{m_hypo.confidence:.2f}</strong>
            </span>
          </div>
          <div style="font-size: 14px; margin-bottom: 12px; line-height: 1.6; color: var(--text-primary);">
            {m_hypo.explanation}
          </div>
          <div style="border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; text-transform: uppercase;">
              Accountable Evidence Citations (Spec Section 9.3 & 10):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
              {citations_html}
            </div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Incident Review — {incident.incident_id} | TraceSleuth</title>
  <style>{_CSS_STYLES}</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        <span class="logo-badge">TraceSleuth</span>
        Incident Review: {incident.incident_id[:16]}...
      </h1>
      <div>
        <span class="badge badge-category">{incident.category.value}</span>
        <span class="badge {sev_class}">{incident.severity.value}</span>
        <span class="badge" style="background: rgba(88,166,255,0.1); color: var(--accent-blue); border: 1px solid var(--accent-blue);">Status: {incident.status.value.upper()}</span>
      </div>
    </header>

    <div class="alert-suspicious">
      <div style="font-size: 15px; font-weight: 600; margin-bottom: 4px;">Root-Cause Forensics Diagnosis</div>
      <div>{incident.summary}</div>
      <div style="margin-top: 6px; font-size: 13px;">{first_suspicious_text} | Confidence Score: <strong>{incident.hypothesis_score:.2f}</strong></div>
      <div style="margin-top: 6px; font-size: 13px; color: var(--accent-amber);"><strong>Recommended Remediation:</strong> {remediation}</div>
    </div>

    {model_hypo_html}

    <div class="card">
      <div class="card-header">
        <span>Supporting Evidence Items ({len(incident.evidence)})</span>
        <span style="font-size: 13px; color: var(--text-secondary); font-weight: normal;">Trace ID: <a href="/traces/{incident.trace_id}">{incident.trace_id[:16]}...</a></span>
      </div>
      <div class="evidence-grid">
        {evidence_html}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span>Causal Execution Timeline (Earliest Suspicious Event Highlighted)</span>
        <span style="font-size: 13px; color: var(--text-secondary); font-weight: normal;">{len(timeline_entries)} span & event transitions</span>
      </div>
      <div class="timeline">
        {timeline_html}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span>Similar Historical Incidents (Failure Signature Matching)</span>
        <span style="font-size: 13px; color: var(--text-secondary); font-weight: normal;">Spec Section 11 & 31</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Incident ID</th>
            <th>Category</th>
            <th>Severity</th>
            <th>Similarity</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody>
          {similar_html}
        </tbody>
      </table>
    </div>

    <div class="card" style="text-align: center; padding: 24px;">
      <h3 style="margin-bottom: 8px;">Replay & Verification (Milestone 7)</h3>
      <p style="color: var(--text-secondary); font-size: 14px; margin-bottom: 16px;">
        Test whether patching the retriever or policy resolves this incident using controlled replay.
      </p>
      <form action="/replay" method="post" style="display: inline-block;">
        <button class="btn" type="button" onclick="triggerControlledReplay()">Trigger Controlled Replay Fix</button>
      </form>
      <script>
        function triggerControlledReplay() {{
          fetch('/replay', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
              source_trace_id: '{incident.trace_id}',
              replay_type: 'controlled',
              retriever_mode: 'fixed'
            }})
          }})
          .then(r => r.json())
          .then(data => {{
            alert('Replay completed with verdict: ' + data.verdict.toUpperCase() + '\\n' + data.summary);
            window.location.reload();
          }})
          .catch(e => alert('Replay error: ' + e));
        }}
      </script>
    </div>
  </div>
</body>
</html>
"""
    return html


@router.get("", response_class=HTMLResponse)
def dashboard_overview() -> str:
    """Renders the overview dashboard listing recent traces, incidents, SLO metrics, and alerts."""
    from src.metrics.service import MetricsService
    metrics_svc = MetricsService()
    sla = metrics_svc.get_sla_metrics(limit=50)
    alerts = metrics_svc.get_alerts(limit=50)

    repo = _get_repo()
    incidents, total_inc = repo.list_incidents(limit=20)
    traces, total_tr = repo.list_traces(limit=20)
    cases, total_cases = repo.list_regression_cases(limit=10)

    # Active alerts banner
    alerts_html = ""
    firing_alerts = [a for a in alerts if a.status.value == "firing"]
    if firing_alerts:
        alert_items = "".join(
            f"""<div style="margin-bottom: 6px;">
                 <span class="badge badge-p0">{a.severity.value}</span>
                 <strong>{a.name}</strong>: {a.summary}
               </div>"""
            for a in firing_alerts
        )
        alerts_html = f"""
        <div style="background: rgba(248,81,73,0.12); border: 1px solid var(--accent-red); border-radius: 8px; padding: 16px; margin-bottom: 20px;">
          <div style="font-weight: 600; font-size: 15px; color: var(--accent-red); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span>🚨 Active Alerts ({len(firing_alerts)} Firing)</span>
          </div>
          {alert_items}
        </div>
        """

    inc_rows = ""
    for inc in incidents:
        inc_rows += f"""
        <tr>
          <td><a href="/ui/incidents/{inc.incident_id}">{inc.incident_id[:16]}...</a></td>
          <td><span class="badge badge-category">{inc.category.value}</span></td>
          <td><span class="badge badge-{inc.severity.value.lower()}">{inc.severity.value}</span></td>
          <td>{inc.hypothesis_score:.2f}</td>
          <td>{inc.summary[:70]}</td>
          <td>{inc.status.value.upper()}</td>
        </tr>
        """

    tr_rows = ""
    for tr in traces:
        tr_rows += f"""
        <tr>
          <td><a href="/traces/{tr.trace_id}">{tr.trace_id[:16]}...</a></td>
          <td>{tr.duration_ms:.1f}ms</td>
          <td>{len(tr.spans)}</td>
          <td>{tr.total_tokens()}</td>
          <td><strong>{tr.outcome.value.upper()}</strong></td>
        </tr>
        """

    reg_rows = ""
    for c in cases:
        st_color = "var(--accent-green)" if c.status.value == "passed" else ("var(--accent-amber)" if c.status.value == "active" else "var(--accent-red)")
        reg_rows += f"""
        <tr>
          <td><code>{c.id}</code></td>
          <td><span class="badge badge-category">{c.failure_type}</span></td>
          <td><strong style="color: {st_color};">{c.status.value.upper()}</strong></td>
          <td>{len(c.assertions)} assertions</td>
          <td>{c.query[:60]}...</td>
          <td><a href="/regression/cases/{c.id}" style="font-size: 12px;">Inspect</a></td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TraceSleuth — Forensic Observability Dashboard</title>
  <style>
    {_CSS_STYLES}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-title {{ font-size: 13px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
    .metric-val {{ font-size: 28px; font-weight: 700; color: var(--text-primary); }}
    .metric-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><span class="logo-badge">TraceSleuth</span> Reliability & Forensics Dashboard</h1>
      <div>
        <span class="badge" style="background: rgba(63,185,80,0.15); color: var(--accent-green);">Total Traces: {total_tr}</span>
        <span class="badge" style="background: rgba(248,81,73,0.15); color: var(--accent-red);">Total Incidents: {total_inc}</span>
      </div>
    </header>

    {alerts_html}

    <!-- Milestone 9: Live SLO Metrics Cards -->
    <div class="metrics-grid">
      <div class="metric-card" style="border-top: 3px solid var(--accent-green);">
        <div class="metric-title">Quality Pass Rate</div>
        <div class="metric-val" style="color: var(--accent-green);">{sla['quality_pass_rate'] * 100:.1f}%</div>
        <div class="metric-sub">SLO Target: 95.0%</div>
      </div>
      <div class="metric-card" style="border-top: 3px solid var(--accent-blue);">
        <div class="metric-title">Availability Rate</div>
        <div class="metric-val">{sla['availability_rate'] * 100:.1f}%</div>
        <div class="metric-sub">Technical HTTP 200 pass rate</div>
      </div>
      <div class="metric-card" style="border-top: 3px solid var(--accent-amber);">
        <div class="metric-title">Latency (p50 / p95)</div>
        <div class="metric-val" style="font-size: 22px;">{sla['p50_latency_ms']:.0f}ms / {sla['p95_latency_ms']:.0f}ms</div>
        <div class="metric-sub">p99: {sla['p99_latency_ms']:.0f}ms (SLO: 350ms)</div>
      </div>
      <div class="metric-card" style="border-top: 3px solid var(--accent-purple);">
        <div class="metric-title">Total Est. Cost</div>
        <div class="metric-val" style="font-size: 22px;">${sla['estimated_cost_usd']:.4f}</div>
        <div class="metric-sub">{sla['total_tokens']:,} tokens consumed</div>
      </div>
    </div>

    <!-- Milestone 8: Regression Test Cases -->
    <div class="card">
      <div class="card-header">
        <span>CI/CD Regression Coverage (Milestone 8 — {total_cases} Cases)</span>
        <a href="/regression/export?format=yaml" target="_blank" style="font-size: 12px; color: var(--accent-blue);">Export YAML for CI</a>
      </div>
      <table>
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Failure Type</th>
            <th>Status</th>
            <th>Assertions</th>
            <th>Query</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {reg_rows or "<tr><td colspan='6' style='text-align: center; color: var(--text-secondary);'>No regression cases created yet. Convert confirmed incidents to regression tests.</td></tr>"}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-header">Recent Classified Failure Incidents</div>
      <table>
        <thead>
          <tr>
            <th>Incident ID</th>
            <th>Category</th>
            <th>Severity</th>
            <th>Score</th>
            <th>Summary</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {inc_rows or "<tr><td colspan='6' style='text-align: center; color: var(--text-secondary);'>No incidents recorded yet.</td></tr>"}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-header">Recent Traces</div>
      <table>
        <thead>
          <tr>
            <th>Trace ID</th>
            <th>Latency</th>
            <th>Spans</th>
            <th>Tokens</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {tr_rows or "<tr><td colspan='5' style='text-align: center; color: var(--text-secondary);'>No traces recorded yet.</td></tr>"}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
