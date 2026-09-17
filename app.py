"""
app.py — TawasolPay Executive Cyber Risk Dashboard.

A Streamlit front end over `backend.py`. This module renders only; it never
scores, ranks, or reorders anything. Every number shown here comes from the
deterministic engine, and every NIST control shown was retrieved from the
catalogue rather than generated.

UI approach
-----------
Native Streamlit components (`st.metric`, `st.columns`, `st.container(border=True)`,
`st.expander`) do the structural layout, so spacing, wrapping and responsiveness are
handled by Streamlit itself rather than by hand-rolled HTML that breaks on version
upgrades. Custom CSS is scoped to a handful of project-defined classes (`tp-badge`,
`tp-driver`, `tp-why`) for the small number of visual elements Streamlit has no
native equivalent for — it never targets Streamlit's internal DOM structure or
generated class names, which is what causes text like raw arrow glyphs or class
names to leak into the rendered page when an internal name changes.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st

import backend
from backend import Risk, RiskReport, SchemaValidationError

try:  # Plotly gives the requested executive-style charts; the app must not depend on it.
    import plotly.graph_objects as go

    PLOTLY = True
except Exception:  # pragma: no cover - optional dependency
    PLOTLY = False


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
CARD_BG = "#161b22"
BORDER = "#30363d"
INK = "#e6edf7"
MUTED = "#8b949e"

SEVERITY_COLORS: Dict[str, str] = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#22c55e",
}

# Streamlit renamed the container-width argument in 1.49 and deprecated the old
# spelling. Feature-detect once so the dashboard runs clean on either generation,
# rather than hand-writing width CSS that would need to track internal markup.
try:
    from packaging.version import Version

    _STRETCH: Dict[str, Any] = (
        {"width": "stretch"} if Version(str(st.__version__)) >= Version("1.49")
        else {"use_container_width": True}
    )
except Exception:
    _STRETCH = {"use_container_width": True}

# Scoped, project-owned CSS only. Every selector below is a class this file itself
# applies via `unsafe_allow_html=True`, or `.stApp`/`[data-testid="stSidebar"]`,
# which are documented, stable Streamlit theming hooks — nothing here reaches into
# Streamlit's internal component markup (no `.st-*` utility classes, no
# `:before`/`:after` pseudo-element overrides on framework-generated icons).
CSS = f"""
<style>
.stApp {{ background-color: {BG}; }}
[data-testid="stSidebar"] {{ background-color: #11161f; border-right: 1px solid {BORDER}; }}

.tp-badge {{
  display: inline-block;
  padding: 0.2rem 0.7rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border: 1px solid currentColor;
  line-height: 1.6;
}}
.tp-badge.critical {{ color: {SEVERITY_COLORS['CRITICAL']}; background: rgba(239, 68, 68, 0.12); }}
.tp-badge.high     {{ color: {SEVERITY_COLORS['HIGH']};     background: rgba(249, 115, 22, 0.12); }}
.tp-badge.medium   {{ color: {SEVERITY_COLORS['MEDIUM']};   background: rgba(234, 179, 8, 0.12); }}
.tp-badge.low      {{ color: {SEVERITY_COLORS['LOW']};      background: rgba(34, 197, 94, 0.12); }}

.tp-driver-row {{
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  padding: 0.3rem 0;
  border-bottom: 1px solid {BORDER};
  font-size: 0.86rem;
}}
.tp-driver-row:last-child {{ border-bottom: none; }}
.tp-driver-row .mark {{ width: 1.1rem; flex-shrink: 0; color: {SEVERITY_COLORS['LOW']}; }}
.tp-driver-row.off .mark {{ color: {MUTED}; }}
.tp-driver-row.off {{ color: {MUTED}; }}
.tp-driver-row .label {{ flex: 1 1 auto; }}
.tp-driver-row .mult {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
.tp-driver-evidence {{ color: {MUTED}; font-size: 0.76rem; padding: 0 0 0.3rem 1.6rem; }}

.tp-why {{
  border-left: 3px solid #3d5a8a;
  background: rgba(61, 90, 138, 0.10);
  border-radius: 0 8px 8px 0;
  padding: 0.75rem 1rem;
  font-size: 0.92rem;
  line-height: 1.6;
}}

.tp-prov {{
  display: inline-block;
  font-size: 0.7rem;
  padding: 0.1rem 0.5rem;
  border-radius: 4px;
  border: 1px solid {BORDER};
  color: {MUTED};
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------

def badge(severity: str) -> str:
    """Severity pill markup — the one label style Streamlit has no native form for."""
    level = (severity or "LOW").upper()
    return f'<span class="tp-badge {level.lower()}">{level}</span>'


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_report(data_dir: str, use_llm: bool, cache_token: int) -> RiskReport:
    """
    Run backend.generate_risk_report() once per (data_dir, use_llm) combination.

    `cache_token` is bumped by the sidebar Reload button to force a rebuild.
    """
    return backend.generate_risk_report(data_dir=data_dir, top_n=5, use_llm=use_llm)


def sorted_unique(frame: pd.DataFrame, column: str) -> List[str]:
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
    if st.button("Reload data and rebuild index", **_STRETCH):
        st.session_state.cache_token += 1
        st.cache_resource.clear()
        st.rerun()

try:
    with st.spinner("Loading data pack, scoring risks and building the NIST index…"):
        report = load_report(data_dir, use_llm, st.session_state.cache_token)
except SchemaValidationError as exc:
    st.title("Cannot start — data pack problem")
    st.caption("The schema validator rejected the data pack. Fix the item below and reload.")
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

    severity_options = [
        s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        if s in set(frame.get("severity_band", pd.Series(dtype=str)))
    ]
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
# Masthead
# ---------------------------------------------------------------------------

st.title("TawasolPay cyber risk position")
st.caption(
    "Composite risk across the asset estate, joined to CISA KEV, regional MDR campaign "
    "intelligence and business service context. Scores are computed deterministically in "
    "Python; remediation guidance is retrieved from NIST SP 800-53 Rev. 5."
)

quality = report.quality


# ---------------------------------------------------------------------------
# KPI metrics row — native st.metric, no hand-built HTML boxes
# ---------------------------------------------------------------------------

kpi_cols = st.columns(4, gap="medium")

with kpi_cols[0]:
    with st.container(border=True):
        st.metric("Assets analyzed", quality.total_assets)
        st.caption(f"{len(quality.assets_without_owner)} without an assigned owner")

with kpi_cols[1]:
    with st.container(border=True):
        st.metric("Vulnerabilities processed", quality.vulnerabilities_processed)
        st.caption(f"of {quality.total_vulnerabilities} records ingested")

with kpi_cols[2]:
    with st.container(border=True):
        st.metric("Threat intel matches", quality.threat_intel_matched)
        st.caption(f"{quality.threat_intel_unmatched} records were industry noise")

with kpi_cols[3]:
    with st.container(border=True):
        leader = top_five[0] if top_five else None
        st.metric("Top priority target", leader.asset_name if leader else "None in filter")
        st.caption(
            f"score {leader.composite_score:.1f} · {leader.cve_display}"
            if leader else "widen the filters"
        )

if not filtered:
    st.warning("No risks match the current filters. Clear one of them in the sidebar to see results.")

st.divider()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def severity_chart(risks: Sequence[Risk]) -> None:
    """Risk severity distribution, colored per the executive palette."""
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
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=12),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="Risks"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(figure, **_STRETCH)
    else:
        st.bar_chart(pd.DataFrame({"Risks": [counts[b] for b in order]}, index=order))


def driver_chart(risks: Sequence[Risk]) -> None:
    """Active risk driver counts: Exposure, KEV, Campaign, Criticality, Missing EDR."""
    counts = backend.driver_counts_for(risks)
    labels = list(counts.keys())[::-1]
    values = [counts[label] for label in labels]

    if PLOTLY:
        figure = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#3b82f6",
                text=values,
                textposition="outside",
                hovertemplate="%{y}: %{x} risks<extra></extra>",
            )
        )
        figure.update_layout(
            height=300,
            margin=dict(l=8, r=32, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=12),
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="Risks affected"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(figure, **_STRETCH)
    else:
        st.bar_chart(pd.DataFrame({"Risks affected": values}, index=labels))


chart_cols = st.columns(2, gap="large")
with chart_cols[0]:
    st.subheader("Risk severity distribution")
    with st.container(border=True):
        severity_chart(filtered)
with chart_cols[1]:
    st.subheader("Active risk drivers")
    with st.container(border=True):
        driver_chart(filtered)

st.divider()


# ---------------------------------------------------------------------------
# Top 5 risk cards
# ---------------------------------------------------------------------------

def render_drivers(risk: Risk) -> None:
    """Multiplier lineage panel: active/inactive state, weight, and evidence."""
    rows: List[str] = []
    for factor in risk.factors:
        state_class = "" if factor.active else " off"
        mark = "✓" if factor.active else "—"
        rows.append(
            f'<div class="tp-driver-row{state_class}">'
            f'<span class="mark">{mark}</span>'
            f'<span class="label">{factor.label}</span>'
            f'<span class="mult">×{factor.multiplier}</span></div>'
            f'<div class="tp-driver-evidence">{factor.evidence}</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption(
        f"CVSS {risk.cvss:.1f} × "
        + " × ".join(f"{f.multiplier}" for f in risk.factors)
        + f" = **{risk.composite_score:.2f}**"
    )


def render_nist(risk: Risk) -> None:
    """Retrieved-control section in a distinct bordered container."""
    payload: Dict[str, Any] = risk.nist or {}
    control = payload.get("control")
    source_dataset = payload.get("source_dataset", backend.FILE_NIST)
    guidance_source = payload.get("guidance_source", "none")
    provenance = {
        "groq": "AI summary of retrieved control",
        "nist_verbatim": "Retrieved control text, verbatim",
        "no_retrieval": "Nothing retrieved",
        "none": "Not retrieved",
    }.get(guidance_source, guidance_source)

    with st.container(border=True):
        if control:
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**{control['control_id']} — {control['control_title']}**")
                st.caption(f"{control['family']} family · retrieved from {source_dataset}")
            with top[1]:
                st.metric("Cosine distance", f"{control['distance']:.3f}")
            st.markdown(f'<span class="tp-prov">{provenance}</span>', unsafe_allow_html=True)
            st.write(payload.get("guidance", ""))

            with st.expander("Source control text and retrieval trace"):
                st.text(control.get("control_text") or "No control text in the catalogue row.")
                if control.get("discussion"):
                    st.markdown("**Discussion**")
                    st.text(control["discussion"])
                if control.get("related"):
                    st.caption(f"Related controls: {control['related']}")
                candidates = payload.get("candidates") or []
                if len(candidates) > 1:
                    st.markdown("**Other candidates considered**")
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
                        hide_index=True,
                        **_STRETCH,
                    )
                hint = payload.get("hint")
                if hint:
                    st.caption(
                        f"Query seeded with the hint for '{hint.get('finding_type')}' "
                        f"(priority {hint.get('priority_hint') or 'n/a'}). "
                        "The hint steers retrieval only; it is not the guidance shown above."
                    )
        else:
            st.markdown("**No control retrieved**")
            st.markdown(f'<span class="tp-prov">{provenance}</span>', unsafe_allow_html=True)
            st.write(payload.get("guidance", ""))


def render_risk_card(risk: Risk) -> None:
    """One expandable Top 5 entry, built from native containers and columns."""
    title = (
        f"#{risk.rank}  ·  {risk.asset_name}  ·  {risk.cve_display}  ·  "
        f"score {risk.composite_score:.2f}  ·  {risk.severity_band}"
    )
    with st.expander(title, expanded=(risk is top_five[0])):
        header_cols = st.columns([1, 4])
        with header_cols[0]:
            st.markdown(badge(risk.severity_band), unsafe_allow_html=True)
        with header_cols[1]:
            st.caption(f"composite {risk.composite_score:.2f} · CVSS {risk.cvss:.1f} · {risk.vulnerability_name}")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Asset and threat context**")
            with st.container(border=True):
                st.markdown(f"**Asset** — {risk.asset_name} ({risk.asset_id})")
                st.caption(f"{risk.asset_type} · {risk.environment} · {risk.location or 'location unknown'}")
                st.caption(f"Platform: {risk.vendor_product or 'not recorded'}")
                st.caption(f"Owning team: {risk.owner_team}")
                sub = st.columns(2)
                with sub[0]:
                    st.metric("Internet facing", yes_no(risk.internet_facing))
                with sub[1]:
                    st.metric("EDR installed", yes_no(risk.edr_installed))
                st.markdown(f"**Finding** — {risk.vulnerability_name} ({risk.vuln_id})")
                st.caption(f"{risk.cve_display} · CVSS {risk.cvss:.1f} ({risk.vuln_severity.title()})")
                st.caption(
                    f"Patch available: {yes_no(risk.patch_available)} · "
                    f"Days open: {risk.days_open if risk.days_open is not None else 'not recorded'}"
                )

            st.markdown("**Business service at risk**")
            with st.container(border=True):
                st.markdown(f"**{risk.business_service}**")
                st.caption(f"Owner: {risk.business_owner} · Customer facing: {yes_no(risk.customer_facing)}")
                st.caption(f"Compliance scope: {risk.compliance_scope or 'none recorded'}")
                st.caption(
                    f"Revenue impact: {risk.revenue_impact or 'not recorded'} · "
                    f"RTO: {f'{risk.rto_hours} h' if risk.rto_hours is not None else 'not recorded'}"
                )
                st.caption(f"Effective criticality used for scoring: **{risk.effective_criticality.title()}**")
                if not risk.service_defined:
                    st.caption(
                        "Data gap: service not defined in business_services.csv; "
                        "asset criticality used alone."
                    )

        with right:
            st.markdown("**Risk drivers applied**")
            with st.container(border=True):
                render_drivers(risk)

            st.markdown("**Threat intelligence**")
            with st.container(border=True):
                if risk.kev_listed:
                    st.markdown(
                        f"CISA KEV: listed {risk.kev_date_added or 'date unknown'}"
                        f"{' · ransomware campaign use' if risk.kev_ransomware else ''}"
                    )
                    if risk.kev_required_action:
                        st.caption(f"Required action: {risk.kev_required_action}")
                else:
                    st.caption("CISA KEV: not listed in the catalogue supplied.")

                if risk.campaign_matches:
                    for match in risk.campaign_matches:
                        st.markdown(
                            f"MDR advisory — {match['threat_actor']} / {match['campaign_name']}"
                        )
                        st.caption(
                            f"Ransomware {match['ransomware']} · confidence {match['confidence'] or 'not stated'}"
                        )
                else:
                    st.caption("MDR advisory: this CVE is not named in today's advisory.")

                if risk.intel_matches:
                    for match in risk.intel_matches:
                        st.markdown(f"Intel {match['intel_id']} — {match['threat_actor']} / {match['campaign_name']}")
                        st.caption(
                            f"{match['exploit_maturity'] or 'maturity unknown'} · "
                            f"last seen {match['active_last_seen'] or 'unknown'} · "
                            f"{match['target_region'] or 'region unknown'}"
                        )
                else:
                    st.caption("Threat intel feed: no campaign record matches this CVE.")

        st.markdown("**Why this ranks here**")
        st.markdown(f'<div class="tp-why">{risk.justification}</div>', unsafe_allow_html=True)

        st.markdown("**Recommended NIST SP 800-53 control**")
        render_nist(risk)


st.subheader("Top 5 composite risks")
st.caption(
    f"Ranked out of {len(filtered)} risks in the current filter "
    f"({len(report.all_risks)} scored in total). Ordering is composite score, then CVSS, then CVE."
)
for risk in top_five:
    render_risk_card(risk)

st.divider()


# ---------------------------------------------------------------------------
# Full register and diagnostics
# ---------------------------------------------------------------------------

with st.expander(f"Full risk register ({len(filtered)} rows)"):
    if filtered:
        register = pd.DataFrame([r.as_row() for r in filtered])
        st.dataframe(register, hide_index=True, height=420, **_STRETCH)
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
        hide_index=True,
        height=min(430, 40 + 35 * len(stats)),
        **_STRETCH,
    )

    diag_left, diag_right = st.columns(2, gap="large")
    with diag_left:
        st.markdown("**Missing files**")
        st.caption("\n".join(quality.missing_files) if quality.missing_files
                   else "All expected files were present.")

        st.markdown("**Malformed CVE identifiers**")
        if quality.malformed_cves:
            st.dataframe(pd.DataFrame(quality.malformed_cves), hide_index=True, **_STRETCH)
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
            st.dataframe(pd.DataFrame(quality.stale_assets), hide_index=True, **_STRETCH)
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
            hide_index=True,
            **_STRETCH,
        )
        st.caption(
            "Parsed deterministically with regular expressions. A CVE appearing here "
            "applies the ×3.0 regional campaign multiplier to every matching finding."
        )
    else:
        st.caption("No campaigns were parsed, so the regional campaign multiplier never fired.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "This assistant is advisory only. It reads data and documents, and it does not execute "
    "automated remediation, change any configuration, scan, or take any network action. "
    "Risk scores and rankings are produced by deterministic Python logic; the language model "
    "only rewrites NIST control text that was already retrieved from the catalogue, and cannot "
    "alter any score. Validate every recommendation through your change management process "
    "before acting on it."
)
st.caption(f"Data pack loaded from `{report.generated_from}`. Synthetic data, generated for assessment purposes.")