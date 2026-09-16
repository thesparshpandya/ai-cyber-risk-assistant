"""
app.py — TawasolPay Executive Cyber Risk Dashboard.

A Streamlit front end over `backend.py`. This module renders only; it never
scores, ranks, or reorders anything. Every number shown here comes from the
deterministic engine, and every NIST control shown was retrieved from the
catalogue rather than generated.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

import backend
from backend import Risk, RiskReport, SchemaValidationError

try:  # Plotly gives nicer executive charts, but the app must not depend on it.
    import plotly.graph_objects as go

    PLOTLY = True
except Exception:  # pragma: no cover - optional dependency
    PLOTLY = False


# Streamlit renamed the container-width argument in 1.49 and deprecated the old
# spelling. Feature-detect once so the dashboard runs clean on either generation.
def _stretch_kwargs() -> Dict[str, Any]:
    """Return the correct full-width keyword for the installed Streamlit."""
    try:
        from packaging.version import Version

        if Version(str(st.__version__)) >= Version("1.49"):
            return {"width": "stretch"}
    except Exception:  # unparseable version string, or packaging absent
        pass
    return {"use_container_width": True}


STRETCH: Dict[str, Any] = _stretch_kwargs()


# ---------------------------------------------------------------------------
# Page setup and theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TawasolPay — Cyber Risk Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#0e1117"
PANEL = "#161b26"
PANEL_EDGE = "#243044"
INK = "#e6edf7"
INK_MUTED = "#8b98ad"

SEVERITY_COLORS: Dict[str, str] = {
    "CRITICAL": "#ff4b4b",
    "HIGH": "#ffa726",
    "MEDIUM": "#ffee58",
    "LOW": "#66bb6a",
}

CSS = f"""
<style>
:root {{
  --bg: {BG};
  --panel: {PANEL};
  --edge: {PANEL_EDGE};
  --ink: {INK};
  --muted: {INK_MUTED};
  --critical: {SEVERITY_COLORS['CRITICAL']};
  --high: {SEVERITY_COLORS['HIGH']};
  --medium: {SEVERITY_COLORS['MEDIUM']};
  --low: {SEVERITY_COLORS['LOW']};
}}

.stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); }}
[data-testid="stSidebar"] {{ background: #11161f; border-right: 1px solid var(--edge); }}
[data-testid="stHeader"] {{ background: transparent; }}

.stApp, .stApp p, .stApp li, .stApp label, .stApp span {{
  color: var(--ink);
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}}

/* ---- masthead ---- */
.tp-masthead {{
  border: 1px solid var(--edge);
  border-left: 3px solid var(--critical);
  border-radius: 10px;
  background: linear-gradient(135deg, #171d29 0%, #12161f 100%);
  padding: 1.15rem 1.4rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 0 22px rgba(255, 75, 75, 0.09);
}}
.tp-masthead h1 {{
  font-size: 1.55rem; font-weight: 650; letter-spacing: -0.015em;
  margin: 0 0 0.3rem 0; color: var(--ink);
}}
.tp-masthead .sub {{ color: var(--muted); font-size: 0.88rem; line-height: 1.5; margin: 0; }}

/* ---- metric cards ---- */
.tp-metric {{
  border: 1px solid var(--edge);
  border-radius: 10px;
  background: var(--panel);
  padding: 0.95rem 1.05rem;
  height: 100%;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}}
.tp-metric:hover {{ border-color: #35486b; box-shadow: 0 0 18px rgba(80, 130, 220, 0.14); }}
.tp-metric .k {{
  color: var(--muted); font-size: 0.72rem; letter-spacing: 0.04em;
  text-transform: none; margin-bottom: 0.4rem;
}}
.tp-metric .v {{
  font-size: 1.85rem; font-weight: 640; line-height: 1.1; color: var(--ink);
  font-variant-numeric: tabular-nums;
}}
.tp-metric .v.small {{ font-size: 1.1rem; font-weight: 600; word-break: break-word; }}
.tp-metric .n {{ color: var(--muted); font-size: 0.74rem; margin-top: 0.35rem; }}

/* ---- severity badge ---- */
.tp-badge {{
  display: inline-block; padding: 0.16rem 0.62rem; border-radius: 999px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
  border: 1px solid currentColor;
}}
.tp-badge.critical {{ color: var(--critical); background: rgba(255, 75, 75, 0.12); }}
.tp-badge.high     {{ color: var(--high);     background: rgba(255, 167, 38, 0.12); }}
.tp-badge.medium   {{ color: var(--medium);   background: rgba(255, 238, 88, 0.12); }}
.tp-badge.low      {{ color: var(--low);      background: rgba(102, 187, 106, 0.12); }}

/* ---- risk card internals ---- */
.tp-card {{
  border: 1px solid var(--edge); border-left: 3px solid var(--edge);
  border-radius: 10px; background: var(--panel); padding: 1rem 1.15rem;
  margin-bottom: 0.7rem;
}}
.tp-card.critical {{ border-left-color: var(--critical); box-shadow: 0 0 16px rgba(255,75,75,0.10); }}
.tp-card.high     {{ border-left-color: var(--high);     box-shadow: 0 0 16px rgba(255,167,38,0.09); }}
.tp-card.medium   {{ border-left-color: var(--medium);   box-shadow: 0 0 16px rgba(255,238,88,0.08); }}
.tp-card.low      {{ border-left-color: var(--low);      box-shadow: 0 0 16px rgba(102,187,106,0.08); }}

.tp-section {{
  color: var(--muted); font-size: 0.74rem; font-weight: 600;
  margin: 0.15rem 0 0.5rem 0;
}}
.tp-kv {{ display: flex; gap: 0.5rem; font-size: 0.86rem; padding: 0.16rem 0; }}
.tp-kv .kk {{ color: var(--muted); min-width: 9.5rem; }}
.tp-kv .vv {{ color: var(--ink); }}

.tp-driver {{
  font-family: "JetBrains Mono", "SF Mono", ui-monospace, monospace;
  font-size: 0.82rem; padding: 0.2rem 0; color: var(--ink);
}}
.tp-driver.off {{ color: #5d6a7f; }}
.tp-driver .ev {{ color: var(--muted); font-size: 0.76rem; }}

.tp-why {{
  border-left: 2px solid #3d5a8a; background: rgba(61, 90, 138, 0.08);
  border-radius: 0 6px 6px 0; padding: 0.6rem 0.85rem;
  font-size: 0.9rem; line-height: 1.55; color: var(--ink);
}}
.tp-nist {{
  border: 1px solid #2c3a52; border-radius: 8px;
  background: rgba(44, 58, 82, 0.22); padding: 0.75rem 0.95rem;
}}
.tp-nist .cid {{ font-weight: 660; color: #7fb4ff; font-size: 0.95rem; }}
.tp-nist .meta {{ color: var(--muted); font-size: 0.74rem; margin: 0.2rem 0 0.6rem 0; }}
.tp-nist .body {{ font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap; }}
.tp-prov {{
  display: inline-block; font-size: 0.68rem; padding: 0.1rem 0.45rem;
  border-radius: 4px; border: 1px solid #3a4a66; color: var(--muted);
  margin-left: 0.4rem;
}}

.tp-foot {{
  border-top: 1px solid var(--edge); margin-top: 2rem; padding-top: 0.9rem;
  color: var(--muted); font-size: 0.78rem; line-height: 1.6;
}}

div[data-testid="stExpander"] {{
  border: 1px solid var(--edge) !important; border-radius: 10px !important;
  background: #131824 !important; margin-bottom: 0.55rem;
}}
div[data-testid="stExpander"] summary {{ font-size: 0.95rem; }}
.stApp hr {{ border-color: var(--edge); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """HTML-escape any value for safe injection into the custom markup."""
    return html.escape(str(value if value is not None else ""), quote=True)


def metric_card(label: str, value: Any, note: str = "", small: bool = False) -> str:
    """Build one executive metric card."""
    return (
        f'<div class="tp-metric"><div class="k">{esc(label)}</div>'
        f'<div class="v{" small" if small else ""}">{esc(value)}</div>'
        f'<div class="n">{esc(note)}</div></div>'
    )


def badge(severity: str) -> str:
    """Build a severity pill."""
    level = (severity or "LOW").upper()
    return f'<span class="tp-badge {level.lower()}">{esc(level)}</span>'


def kv(key: str, value: Any) -> str:
    """Build one key/value row."""
    return f'<div class="tp-kv"><span class="kk">{esc(key)}</span><span class="vv">{esc(value)}</span></div>'


def yes_no(flag: bool) -> str:
    """Render a boolean the way a security reader expects."""
    return "Yes" if flag else "No"


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_report(data_dir: str, use_llm: bool, cache_token: int) -> RiskReport:
    """
    Run the backend pipeline once per (data_dir, use_llm) combination.

    `cache_token` is bumped by the Reload button to force a rebuild. The report
    holds live engine objects, so `cache_resource` is used rather than
    `cache_data`, which would try to serialise them.
    """
    return backend.generate_risk_report(data_dir=data_dir, top_n=5, use_llm=use_llm)


def sorted_unique(frame: pd.DataFrame, column: str) -> List[str]:
    """Distinct, sorted, non-empty values of a column for filter widgets."""
    if column not in frame.columns:
        return []
    values = {str(v).strip() for v in frame[column].dropna().tolist() if str(v).strip()}
    return sorted(values)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "cache_token" not in st.session_state:
    st.session_state.cache_token = 0

with st.sidebar:
    st.markdown("### Data source")
    data_dir = st.text_input(
        "Data pack directory",
        value=os.environ.get("TP_DATA_DIR", str(backend.DEFAULT_DATA_DIR)),
        help="Folder holding the CSVs, the KEV JSON and the MDR advisory.",
    )
    use_llm = st.toggle(
        "Summarise NIST guidance with Groq",
        value=bool(os.environ.get("GROQ_API_KEY", "").strip()),
        help="Off, or with no API key set, the dashboard shows the retrieved NIST "
             "control text verbatim. Scores never involve the model either way.",
    )
    if st.button("Reload data and rebuild index", **STRETCH):
        st.session_state.cache_token += 1
        st.cache_resource.clear()
        st.rerun()

try:
    with st.spinner("Loading data pack, scoring risks and building the NIST index…"):
        report = load_report(data_dir, use_llm, st.session_state.cache_token)
except SchemaValidationError as exc:
    st.markdown(
        '<div class="tp-masthead"><h1>Cannot start — data pack problem</h1>'
        '<p class="sub">The schema validator rejected the data pack. Fix the item below and reload.</p></div>',
        unsafe_allow_html=True,
    )
    st.code(str(exc), language="text")
    st.info(
        "Expected files: assets.csv, vulnerabilities.csv, threat_intelligence.csv, "
        "business_services.csv, remediation_guidance.csv, synthetic_threat_report.md, "
        "known_exploited_vulnerabilities.json, NIST_SP-800-53_rev5_catalog_load.csv"
    )
    st.stop()
except Exception as exc:  # unexpected failure — show it rather than a blank page
    st.error(f"Unexpected startup failure: {type(exc).__name__}: {exc}")
    st.stop()

frame = report.risk_frame

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")
    st.caption("Filters change which risks are shown. They never change how risks are scored.")

    severity_options = [s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
                        if s in set(frame.get("severity_band", pd.Series(dtype=str)))]
    sel_severity = st.multiselect("Severity band", severity_options, default=severity_options)
    sel_service = st.multiselect("Business service", sorted_unique(frame, "business_service"))
    sel_asset_type = st.multiselect("Asset type", sorted_unique(frame, "asset_type"))

    sel_internet = st.selectbox("Internet facing", ["Any", "True", "False"], index=0)
    sel_kev = st.selectbox("CISA KEV listed", ["Any", "True", "False"], index=0)
    sel_campaign = st.selectbox("Active regional campaign", ["Any", "True", "False"], index=0)

    st.markdown("---")
    st.markdown("### Pipeline status")
    st.caption(f"**Retrieval** — {report.nist_status}")
    st.caption(f"**Summarisation** — {report.groq_status}")
    if report.quality.missing_files:
        st.warning("Optional sources missing: " + ", ".join(report.quality.missing_files))


def tri_state(risks: Sequence[Risk], selection: str, attribute: str) -> List[Risk]:
    """Apply a True/False/Any filter over a boolean risk attribute."""
    if selection == "Any":
        return list(risks)
    want = selection == "True"
    return [r for r in risks if bool(getattr(r, attribute)) == want]


filtered: List[Risk] = list(report.all_risks)
if sel_severity:
    filtered = [r for r in filtered if r.severity_band in set(sel_severity)]
if sel_service:
    filtered = [r for r in filtered if r.business_service in set(sel_service)]
if sel_asset_type:
    filtered = [r for r in filtered if r.asset_type in set(sel_asset_type)]
filtered = tri_state(filtered, sel_internet, "internet_facing")
filtered = tri_state(filtered, sel_kev, "kev_listed")
filtered = tri_state(filtered, sel_campaign, "campaign_matches")

top_five = filtered[:5]
# Filtering can promote a risk that was not in the original Top 5, so retrieve
# NIST guidance for anything newly surfaced before rendering its card.
if top_five:
    with st.spinner("Retrieving NIST SP 800-53 guidance…"):
        report.enrich(top_five)


# ---------------------------------------------------------------------------
# Masthead and metrics
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="tp-masthead">'
    "<h1>TawasolPay cyber risk position</h1>"
    '<p class="sub">Composite risk across the asset estate, joined to CISA KEV, regional MDR '
    "campaign intelligence and business service context. Scores are computed deterministically "
    "in Python; remediation guidance is retrieved from NIST SP 800-53 Rev. 5.</p>"
    "</div>",
    unsafe_allow_html=True,
)

quality = report.quality
cols = st.columns(4, gap="small")
with cols[0]:
    st.markdown(
        metric_card("Assets analysed", quality.total_assets,
                    f"{len(quality.assets_without_owner)} without an assigned owner"),
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        metric_card("Vulnerabilities processed", quality.vulnerabilities_processed,
                    f"of {quality.total_vulnerabilities} records ingested"),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        metric_card("Threat intel matches", quality.threat_intel_matched,
                    f"{quality.threat_intel_unmatched} records were industry noise"),
        unsafe_allow_html=True,
    )
with cols[3]:
    leader = top_five[0] if top_five else None
    st.markdown(
        metric_card(
            "Top priority target",
            leader.asset_name if leader else "None in filter",
            f"score {leader.composite_score:.1f} · {leader.cve_display}" if leader else "widen the filters",
            small=True,
        ),
        unsafe_allow_html=True,
    )

if not filtered:
    st.warning("No risks match the current filters. Clear one of them in the sidebar to see results.")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def severity_chart(risks: Sequence[Risk]) -> None:
    """Distribution of scored risks across composite severity bands."""
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts = {band: 0 for band in order}
    for risk in risks:
        counts[risk.severity_band] = counts.get(risk.severity_band, 0) + 1

    if PLOTLY:
        figure = go.Figure(
            go.Bar(
                x=order,
                y=[counts[b] for b in order],
                marker_color=[SEVERITY_COLORS[b] for b in order],
                text=[counts[b] for b in order],
                textposition="outside",
                hovertemplate="%{x}: %{y} risks<extra></extra>",
            )
        )
        figure.update_layout(
            height=290,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK_MUTED, size=12),
            yaxis=dict(gridcolor=PANEL_EDGE, zerolinecolor=PANEL_EDGE, title="Risks"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(figure, **STRETCH)
    else:
        st.bar_chart(pd.DataFrame({"Risks": [counts[b] for b in order]}, index=order))


def driver_chart(risks: Sequence[Risk]) -> None:
    """How often each multiplier is actually firing across the filtered set."""
    counts = backend.driver_counts_for(risks)
    labels = list(counts.keys())[::-1]
    values = [counts[label] for label in labels]

    if PLOTLY:
        figure = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#4f7fd4",
                text=values,
                textposition="outside",
                hovertemplate="%{y}: %{x} risks<extra></extra>",
            )
        )
        figure.update_layout(
            height=290,
            margin=dict(l=8, r=28, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK_MUTED, size=12),
            xaxis=dict(gridcolor=PANEL_EDGE, zerolinecolor=PANEL_EDGE, title="Risks affected"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(figure, **STRETCH)
    else:
        st.bar_chart(pd.DataFrame({"Risks affected": values}, index=labels))


chart_left, chart_right = st.columns(2, gap="medium")
with chart_left:
    st.markdown('<div class="tp-section">Risk severity distribution</div>', unsafe_allow_html=True)
    severity_chart(filtered)
with chart_right:
    st.markdown('<div class="tp-section">Active risk drivers across the filtered estate</div>',
                unsafe_allow_html=True)
    driver_chart(filtered)


# ---------------------------------------------------------------------------
# Top 5 risk cards
# ---------------------------------------------------------------------------

def render_drivers(risk: Risk) -> str:
    """Render the multiplier lineage panel for one risk."""
    rows: List[str] = []
    for factor in risk.factors:
        mark = "✓" if factor.active else " "
        state = "" if factor.active else " off"
        rows.append(
            f'<div class="tp-driver{state}">[{mark}] {esc(factor.label)} '
            f'(x{factor.multiplier})<br><span class="ev">&nbsp;&nbsp;&nbsp;&nbsp;{esc(factor.evidence)}</span></div>'
        )
    product = " × ".join(f"{f.multiplier}" for f in risk.factors)
    rows.append(
        f'<div class="tp-driver" style="margin-top:0.5rem;color:#7fb4ff">'
        f"CVSS {risk.cvss:.1f} × {product} = {risk.composite_score:.2f}</div>"
    )
    return "".join(rows)


def render_nist(risk: Risk) -> None:
    """Render the retrieved-control section, labelling provenance honestly."""
    payload: Dict[str, Any] = risk.nist or {}
    control: Optional[Dict[str, Any]] = payload.get("control")
    source_dataset = payload.get("source_dataset", backend.FILE_NIST)
    guidance_source = payload.get("guidance_source", "none")

    provenance = {
        "groq": "AI summary of retrieved control",
        "nist_verbatim": "Retrieved control text, verbatim",
        "no_retrieval": "Nothing retrieved",
        "none": "Not retrieved",
    }.get(guidance_source, guidance_source)

    if control:
        header = (
            f'<div class="cid">{esc(control["control_id"])} — {esc(control["control_title"])}</div>'
            f'<div class="meta">{esc(control["family"])} family · retrieved from '
            f"{esc(source_dataset)} · {esc(payload.get('retrieval_method', ''))} · "
            f"cosine distance {control['distance']:.3f}"
            f'<span class="tp-prov">{esc(provenance)}</span></div>'
        )
    else:
        header = (
            '<div class="cid">No control retrieved</div>'
            f'<div class="meta">Searched {esc(source_dataset)}'
            f'<span class="tp-prov">{esc(provenance)}</span></div>'
        )

    st.markdown(
        f'<div class="tp-nist">{header}'
        f'<div class="body">{esc(payload.get("guidance", ""))}</div></div>',
        unsafe_allow_html=True,
    )

    if control:
        with st.expander("Source control text and retrieval trace"):
            st.markdown(f"**{control['control_id']} — {control['control_title']}** "
                        f"({control['family']})")
            st.text(control.get("control_text") or "No control text in the catalogue row.")
            if control.get("discussion"):
                st.markdown("**Discussion**")
                st.text(control["discussion"])
            if control.get("related"):
                st.caption(f"Related controls: {control['related']}")
            st.markdown("**Other candidates considered**")
            candidates = payload.get("candidates") or []
            if len(candidates) > 1:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Control": c["control_id"],
                                "Title": c["control_title"],
                                "Distance": round(c["distance"], 4),
                            }
                            for c in candidates
                        ]
                    ),
                    **STRETCH,
                    hide_index=True,
                )
            else:
                st.caption("No other candidate cleared the relevance threshold.")
            hint = payload.get("hint")
            if hint:
                st.caption(
                    f"Query seeded with the hint for '{hint.get('finding_type')}' "
                    f"(priority {hint.get('priority_hint') or 'n/a'}, "
                    f"token overlap {hint.get('match_score')}). "
                    "The hint steers retrieval only; it is not the guidance shown above."
                )


def render_risk_card(risk: Risk) -> None:
    """Render one expandable Top 5 entry."""
    title = (
        f"#{risk.rank}  ·  {risk.asset_name}  ·  {risk.cve_display}  ·  "
        f"score {risk.composite_score:.2f}  ·  {risk.severity_band}"
    )
    with st.expander(title, expanded=(risk is top_five[0])):
        st.markdown(
            f'<div class="tp-card {risk.severity_band.lower()}">'
            f"{badge(risk.severity_band)}"
            f'<span style="margin-left:0.6rem;color:{INK_MUTED};font-size:0.8rem">'
            f"composite {risk.composite_score:.2f} · CVSS {risk.cvss:.1f} · "
            f"{esc(risk.vulnerability_name)}</span></div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown('<div class="tp-section">Asset and threat context</div>', unsafe_allow_html=True)
            rows = [
                kv("Asset", f"{risk.asset_name} ({risk.asset_id})"),
                kv("Asset type", risk.asset_type),
                kv("Environment", f"{risk.environment} · {risk.location or 'location unknown'}"),
                kv("Platform", risk.vendor_product or "not recorded"),
                kv("Owning team", risk.owner_team),
                kv("Internet facing", yes_no(risk.internet_facing)),
                kv("EDR installed", yes_no(risk.edr_installed)),
                kv("Finding", f"{risk.vulnerability_name} ({risk.vuln_id})"),
                kv("CVE", risk.cve_display),
                kv("CVSS", f"{risk.cvss:.1f} ({risk.vuln_severity.title()})"),
                kv("Patch available", yes_no(risk.patch_available)),
                kv("Days open", risk.days_open if risk.days_open is not None else "not recorded"),
            ]
            st.markdown("".join(rows), unsafe_allow_html=True)

            st.markdown('<div class="tp-section" style="margin-top:0.7rem">Business service at risk</div>',
                        unsafe_allow_html=True)
            service_rows = [
                kv("Service", risk.business_service),
                kv("Service owner", risk.business_owner),
                kv("Customer facing", yes_no(risk.customer_facing)),
                kv("Compliance scope", risk.compliance_scope or "none recorded"),
                kv("Revenue impact", risk.revenue_impact or "not recorded"),
                kv("Recovery objective", f"{risk.rto_hours} h" if risk.rto_hours is not None else "not recorded"),
                kv("Risk appetite", risk.risk_appetite or "not recorded"),
                kv("Effective criticality", risk.effective_criticality.title()),
            ]
            if not risk.service_defined:
                service_rows.append(
                    kv("Data gap", "service not defined in business_services.csv; "
                                   "asset criticality used alone")
                )
            st.markdown("".join(service_rows), unsafe_allow_html=True)

        with right:
            st.markdown('<div class="tp-section">Risk drivers applied</div>', unsafe_allow_html=True)
            st.markdown(render_drivers(risk), unsafe_allow_html=True)

            st.markdown('<div class="tp-section" style="margin-top:0.8rem">Threat intelligence</div>',
                        unsafe_allow_html=True)
            if risk.kev_listed:
                st.markdown(
                    kv("CISA KEV", f"listed {risk.kev_date_added or 'date unknown'}"
                                   f"{' · ransomware campaign use' if risk.kev_ransomware else ''}"),
                    unsafe_allow_html=True,
                )
                if risk.kev_required_action:
                    st.markdown(kv("KEV required action", risk.kev_required_action),
                                unsafe_allow_html=True)
            else:
                st.markdown(kv("CISA KEV", "not listed in the catalogue supplied"),
                            unsafe_allow_html=True)

            if risk.campaign_matches:
                for match in risk.campaign_matches:
                    st.markdown(
                        kv(
                            "MDR advisory",
                            f"{match['threat_actor']} / {match['campaign_name']} · "
                            f"ransomware {match['ransomware']} · confidence "
                            f"{match['confidence'] or 'not stated'}",
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(kv("MDR advisory", "this CVE is not named in today's advisory"),
                            unsafe_allow_html=True)

            if risk.intel_matches:
                for match in risk.intel_matches:
                    st.markdown(
                        kv(
                            f"Intel {match['intel_id']}",
                            f"{match['threat_actor']} / {match['campaign_name']} · "
                            f"{match['exploit_maturity'] or 'maturity unknown'} · "
                            f"last seen {match['active_last_seen'] or 'unknown'} · "
                            f"{match['target_region'] or 'region unknown'}",
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(kv("Threat intel feed", "no campaign record matches this CVE"),
                            unsafe_allow_html=True)

        st.markdown('<div class="tp-section" style="margin-top:0.6rem">Why this ranks here</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="tp-why">{esc(risk.justification)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="tp-section" style="margin-top:0.9rem">Recommended NIST SP 800-53 control</div>',
                    unsafe_allow_html=True)
        render_nist(risk)


st.markdown("---")
st.markdown('<div class="tp-section">Top 5 composite risks</div>', unsafe_allow_html=True)
st.caption(
    f"Ranked out of {len(filtered)} risks in the current filter "
    f"({len(report.all_risks)} scored in total). Ordering is composite score, then CVSS, then CVE."
)
for risk in top_five:
    render_risk_card(risk)


# ---------------------------------------------------------------------------
# Full register and diagnostics
# ---------------------------------------------------------------------------

with st.expander(f"Full risk register ({len(filtered)} rows)"):
    if filtered:
        register = pd.DataFrame([r.as_row() for r in filtered])
        st.dataframe(register, hide_index=True, height=420, **STRETCH)
        st.download_button(
            "Download filtered register as CSV",
            data=register.to_csv(index=False).encode("utf-8"),
            file_name="tawasolpay_risk_register.csv",
            mime="text/csv",
        )
    else:
        st.caption("Nothing to show under the current filters.")

with st.expander("Data quality and diagnostics"):
    st.markdown("**Ingestion and join statistics**")
    stats = quality.as_display_dict()
    st.dataframe(
        pd.DataFrame({"Measure": list(stats.keys()), "Value": list(stats.values())}),
        **STRETCH,
        hide_index=True,
        height=min(430, 40 + 35 * len(stats)),
    )

    diag_left, diag_right = st.columns(2, gap="medium")
    with diag_left:
        st.markdown("**Missing files**")
        st.caption("\n".join(quality.missing_files) if quality.missing_files
                   else "All expected files were present.")

        st.markdown("**Malformed CVE identifiers**")
        if quality.malformed_cves:
            st.dataframe(pd.DataFrame(quality.malformed_cves),
                         hide_index=True, **STRETCH)
        else:
            st.caption("Every CVE identifier parsed cleanly.")

        st.markdown("**Vulnerabilities referencing an unknown asset_id**")
        st.caption(", ".join(quality.unmapped_asset_ids) if quality.unmapped_asset_ids
                   else "Every vulnerability mapped to a known asset.")

    with diag_right:
        st.markdown("**Assets with no assigned owner**")
        st.caption(", ".join(quality.assets_without_owner) if quality.assets_without_owner
                   else "Every asset has an owning team.")

        st.markdown("**Business services referenced but not defined**")
        st.caption(", ".join(quality.unmapped_business_services)
                   if quality.unmapped_business_services
                   else "Every referenced service exists in business_services.csv.")

        st.markdown("**Stale assets (last seen more than 30 days ago)**")
        if quality.stale_assets:
            st.dataframe(pd.DataFrame(quality.stale_assets),
                         hide_index=True, **STRETCH)
        else:
            st.caption("No stale assets in the inventory.")

    if quality.warnings:
        st.markdown("**Warnings raised during ingestion and scoring**")
        for warning in quality.warnings[:60]:
            st.caption(f"· {warning}")
        if len(quality.warnings) > 60:
            st.caption(f"… and {len(quality.warnings) - 60} more.")

with st.expander("Regional campaigns parsed from the MDR advisory"):
    if report.pack.campaigns:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Actor": c.threat_actor,
                        "Campaign": c.campaign_name,
                        "Target profile": c.target_profile,
                        "Ransomware": yes_no(c.ransomware),
                        "Confidence": c.confidence,
                        "CVEs named": ", ".join(c.cve_display) or "none extracted",
                    }
                    for c in report.pack.campaigns
                ]
            ),
            **STRETCH,
            hide_index=True,
        )
        st.caption(
            "Parsed deterministically with regular expressions. A CVE appearing here "
            "applies the x3.0 regional campaign multiplier to every matching finding."
        )
    else:
        st.caption("No campaigns were parsed, so the regional campaign multiplier never fired.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="tp-foot">'
    "This assistant is advisory only. It reads data and documents, and it does not execute "
    "automated remediation, change any configuration, scan, or take any network action. "
    "Risk scores and rankings are produced by deterministic Python logic; the language model "
    "only rewrites NIST control text that was already retrieved from the catalogue, and cannot "
    "alter any score. Validate every recommendation through your change management process "
    "before acting on it.<br>"
    "Data pack loaded from <code>" + esc(report.generated_from) + "</code>. "
    "Synthetic data, generated for assessment purposes."
    "</div>",
    unsafe_allow_html=True,
)
