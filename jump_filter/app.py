"""Local Streamlit workbench. Run: streamlit run jump_filter/app.py."""
# SETUP LOGIC: Browser dependencies are loaded only by this optional application.
from dataclasses import asdict
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import numpy as np
import pandas as pd
import streamlit as st
from jump_filter import FilterConfig, METHODS, make_demo
from jump_filter.dashboard import ReviewWorkspace, evaluation_summary, method_comparison, audit_tables
from jump_filter.explanations import METHOD_HELP, METHOD_EXPLANATIONS, COMMON_STEPS, CLOCK_STEPS, FITTING_STEPS, PARAMETER_HELP, math_display_blocks
from jump_filter.plots import METHOD_LABELS, trade_figure, diagnostic_figure, comparison_figure

# UI LOGIC: Native Streamlit controls remain keyboard accessible and use a compact research layout.
CSS = """
<style>
.stApp{background:#F5F6F8;color:#142D3D;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stMainBlockContainer{padding-top:4rem;padding-bottom:3rem;max-width:1680px}
header[data-testid="stHeader"]{background:rgba(245,246,248,.92)}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid #E1E7EC}
section[data-testid="stSidebar"] h1{font-size:20px;letter-spacing:-.025em}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{font-size:12px;line-height:1.55;color:#657887}
.jf-hero{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:0 0 22px;margin:0 0 22px;border-bottom:1px solid #DFE5EB}
.jf-brand{display:flex;align-items:center;gap:15px}.jf-mark{display:grid;place-items:center;flex-shrink:0;width:46px;height:46px;background:#142D3D;color:#fff;border-radius:12px;font-size:16px;font-weight:750;letter-spacing:-.06em}
.jf-hero h1{font-size:29px;letter-spacing:-.045em;line-height:1.15;margin:4px 0;color:#142D3D;font-weight:700}
.jf-hero p{color:#657887;font-size:13px;margin:0;line-height:1.5}.jf-eyebrow{font-size:10px;font-weight:700;letter-spacing:.13em;color:#657887;text-transform:uppercase}
.jf-pill{display:inline-block;flex-shrink:0;padding:5px 10px;background:#E8F4F3;border:1px solid #CEE7E3;border-radius:20px;color:#137267;font-size:11px;font-weight:650}
.jf-section-label{margin:17px 0 7px;color:#657887;font-size:10px;font-weight:750;letter-spacing:.09em;text-transform:uppercase}
.jf-context{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:0 0 16px;color:#657887;font-size:12px}.jf-context strong{color:#142D3D;font-weight:600}.jf-context .jf-dot{color:#AAB7C0}
.jf-method-summary{border-left:3px solid #087F8C;padding:12px 16px;margin:0 0 18px;background:#EDF5F6;border-radius:0 8px 8px 0;color:#294D5C;font-size:14px;line-height:1.65}
.jf-method-context{padding-top:4px}.jf-method-context strong{display:block;color:#142D3D;font-size:14px;margin:4px 0}.jf-method-context span{color:#657887;font-size:12px}
.jf-section-heading{margin:18px 0 4px;font-weight:650;letter-spacing:-.02em;color:#142D3D;font-size:18px}.jf-section-note{margin:0 0 14px;color:#657887;font-size:12px;line-height:1.6}
.jf-empty{background:#fff;border:1px solid #DEE6EC;border-radius:12px;padding:24px;margin:8px 0 16px}.jf-empty h2{margin:0 0 8px;font-size:23px;letter-spacing:-.025em}.jf-empty p{color:#657887;margin:0;font-size:14px;line-height:1.65}
.jf-step{border-top:2px solid #D6E7E9;padding:13px 0;margin:8px 0 15px}.jf-step span{color:#087F8C;font-size:11px;font-weight:700}.jf-step strong{display:block;margin:5px 0;font-size:14px}.jf-step p{font-size:12px;line-height:1.6;color:#657887;margin:0}
div[data-testid="stMetric"]{background:white;border:1px solid #E0E6EC;border-radius:10px;padding:14px 15px;box-shadow:0 2px 4px rgba(20,45,61,.025)}
div[data-testid="stMetricLabel"] p{font-size:11px!important;color:#657887!important;font-weight:500}div[data-testid="stMetricValue"]{font-size:27px!important;color:#142D3D;letter-spacing:-.035em;font-weight:650}
.st-key-jf-metrics [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricValue"]{color:#D84C62}
.st-key-jf-metrics [data-testid="stColumn"]:nth-child(4) [data-testid="stMetricValue"]{color:#087F8C}
.stButton>button{font-size:13px;font-weight:600;border-radius:8px;min-height:38px}.stButton>button[kind="primary"]{background:#087F8C;border-color:#087F8C;box-shadow:0 2px 4px rgba(8,127,140,.1)}
.stButton>button[kind="primary"]:hover{background:#076F7B;border-color:#076F7B}.stButton>button:focus-visible{outline:3px solid #87B9C1;outline-offset:2px}
.st-key-jf-apply{position:sticky;bottom:0;background:#fff;padding-top:12px;border-top:1px solid #E1E7EC;z-index:5}
.stTabs [data-baseweb="tab-list"]{gap:22px;margin-top:15px}.stTabs [data-baseweb="tab"]{font-size:12px;font-weight:600;padding-bottom:13px}.stTabs [aria-selected="true"]{color:#087F8C}
div[data-testid="stExpander"]{background:white;border:1px solid #DEE6EC;border-radius:9px}div[data-testid="stExpander"] details summary p{font-size:13px;font-weight:600}
div[data-testid="stDataFrame"]{border-radius:9px;overflow:hidden}div[data-testid="stPlotlyChart"]{border:1px solid #E1E7EC;border-radius:11px;overflow:hidden;background:#fff}
div[data-testid="stLatex"]{background:#F7F9FB;border:1px solid #E9EEF2;border-radius:8px;max-width:100%;overflow-x:auto;padding:10px 14px;margin:7px 0 12px}
div[data-testid="stLatex"] .katex-display{margin:.3em 0;max-width:100%;overflow-x:auto}div[data-testid="stLatex"] .katex-display>.katex{text-align:left}
@media(max-width:900px){.jf-hero{align-items:flex-start}.jf-hero>.jf-pill{display:none}.jf-hero h1{font-size:25px}.stMainBlockContainer{padding-left:1.1rem;padding-right:1.1rem}.stTabs [data-baseweb="tab-list"]{gap:13px}}
</style>
"""


def _section(title, note=None):
    # UI LOGIC: Escaped headings provide the same hierarchy in tables, plots, and method documentation.
    st.markdown(f'<div class="jf-section-heading">{escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="jf-section-note">{escape(note)}</p>', unsafe_allow_html=True)


def _math_steps(steps):
    # UI LOGIC: Progressive disclosure keeps every source equation and explanatory clause available.
    for title, prose, formula in steps:
        st.markdown(f"**{title}**")
        st.write(prose)
        for block in math_display_blocks(formula):
            st.latex(block)


@st.cache_data(show_spinner=False)
def _demo(dataset_version):
    # CACHEING LOGIC: The public demo version participates in the cache key; user uploads are never cached here.
    return make_demo()


def _sync(source, target):
    # UI LOGIC: Streamlit callbacks run before the next render, so paired controls update safely.
    st.session_state[target] = st.session_state[source]


def _parameter(name, label, value, lower, upper, step, *, integer=False, disabled=False):
    # UI LOGIC: Stable keys keep the slider and exact input synchronized across reruns.
    slider_key, number_key = f"jf_slider_{name}", f"jf_number_{name}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(value) if integer else float(value)
        st.session_state[number_key] = int(value) if integer else float(value)
    cast = int if integer else float
    lower, upper, step = cast(lower), cast(upper), cast(step)
    st.slider(label, min_value=lower, max_value=upper, step=step, key=slider_key,
              on_change=_sync, args=(slider_key, number_key), disabled=disabled, help=PARAMETER_HELP[name])
    return st.number_input(f"Exact value · {label}", min_value=lower, max_value=upper, step=step,
                           key=number_key, on_change=_sync, args=(number_key, slider_key),
                           label_visibility="collapsed", disabled=disabled)


def _method_mathematics(method, pending_config, applied_method=None):
    # UI LOGIC: Native LaTeX rendering works immediately after selection, independently of Apply or plot state.
    card = METHOD_EXPLANATIONS[method]
    st.subheader(card["name"])
    st.caption(f'{card["mode"]} · Explanation follows selected method: {method} · Applied plots and exports: {applied_method or "none yet"}')
    st.markdown(f'<div class="jf-method-summary">{escape(card["summary"])}</div>', unsafe_allow_html=True)
    with st.expander("Algorithm · equations and decision rule", expanded=True):
        _math_steps(card["steps"])
    example, limits = st.columns(2)
    with example:
        _section("Numerical example")
        st.write(card["example"])
    with limits:
        _section("Assumptions and limitations")
        st.write(card["tradeoffs"])
    with st.expander("Shared foundations · neighborhoods, scales and strict boundaries"):
        _math_steps(COMMON_STEPS)
    with st.expander("Market clock · sessions, irregular events and liquidity"):
        _math_steps(CLOCK_STEPS)
    with st.expander("Downstream fitting · selection, influence and coverage"):
        _math_steps(FITTING_STEPS)
    _section("Selected parameters", "These values follow your current controls; plots and exports follow the last successful Apply.")
    records = [(name, str(pending_config[name]), PARAMETER_HELP[name]) for name in card["parameters"]]
    records += [(name, str(pending_config[name]), explanation) for name, explanation in
                [("horizon", "Reference-time distance limit; multiscale uses 1, 2 and 4 times this value."),
                 ("max_gap", "Split after this much inactivity in the selected clock; references never cross segments."),
                 ("time_basis", "Trading uses cumulative open-session time; wall uses calendar time."),
                 ("session_timezone", "Market session timezone, configured separately from parsing naive input timestamps."),
                 ("session_open", "Regular weekday session opening time."), ("session_close", "Regular weekday session closing time."),
                 ("holidays", "Explicit full-day closures; supply session_schedule for a complete custom calendar.")]]
    st.dataframe(pd.DataFrame(records, columns=["parameter", "pending value", "effect"]), width="stretch", hide_index=True)
    title, url = card["reference"]
    st.markdown(f"Reference: [{title}]({url})")


def _column_index(columns, aliases, fallback):
    # UI LOGIC: Suggested mappings are visible and editable; the numerical engine never infers units.
    matches = {str(column).lower(): i for i, column in enumerate(columns)}
    return next((matches[name.lower()] for name in aliases if name.lower() in matches), min(fallback, len(columns) - 1))


def _metrics(rows):
    # UI LOGIC: Counts distinguish algorithm evaluation from missing inputs and limited support.
    stats = evaluation_summary(rows)
    labels = [("Trades", f'{stats["total"]:,}'), ("Evaluated", f'{stats["evaluated"]:,}'),
              ("Flagged", f'{stats["outliers"]:,}'),
              ("Flag rate", f'{stats["flag_rate"]:.1%}' if np.isfinite(stats["flag_rate"]) else "—"),
              ("Low support", f'{stats["unsupported"]:,}'), ("Invalid", f'{stats["invalid"]:,}')]
    with st.container(key="jf-metrics"):
        for column, (label, value) in zip(st.columns(6), labels):
            column.metric(label, value)
    return stats


def _table(frame, *, height="auto"):
    # UI LOGIC: Human-readable headers and explicit units format the original audit values without altering them.
    labels = {"CUSIP": "CUSIP", "rows": "Trades", "total": "Trades", "supplied": "Supplied", "evaluated": "Evaluated", "flagged": "Flagged",
              "outliers": "Flagged", "accepted": "Accepted", "fit_eligible": "Fit eligible", "invalid": "Invalid",
              "insufficient": "Low support", "regime_candidates": "Regime candidates", "median_score": "Median score",
              "median_reference_count": "Median references", "flagged_rate": "Flag rate", "flag_rate": "Flag rate",
              "coverage": "Coverage", "hard_retention": "Hard retention", "hard_fit_rows": "Hard-fit rows",
              "soft_fit_rows": "Soft-fit rows", "soft_weight_sum": "Sum of influence weights", "jf_status": "Status",
              "jf_reason": "Reason", "jf_n_reference": "References", "jf_score": "Robust score", "jf_baseline": "Baseline",
              "jf_residual": "Residual", "jf_threshold": "Deviation cutoff", "jf_weight": "Influence weight",
              "jf_fit_eligible": "Fit eligible", "jf_row_id": "Source row", "jf_gap_minutes": "Active gap · min",
              "jf_wall_gap_minutes": "Wall gap · min", "jf_session_boundary": "Session boundary", "jf_scale": "Local scale",
              "jf_reference_density_per_hour": "References / hour", "jf_reference_span_minutes": "Reference span · min"}
    config = {name: st.column_config.Column(labels.get(name, name.replace("_", " ").capitalize())) for name in frame.columns}
    for name in ["coverage", "flagged_rate", "flag_rate", "hard_retention"]:
        if name in frame:
            config[name] = st.column_config.NumberColumn(labels[name], format="percent", width="small")
    if "jf_time" in frame:
        config["jf_time"] = st.column_config.DatetimeColumn("Time · UTC", format="YYYY-MM-DD HH:mm", width="medium")
    st.dataframe(frame, width="stretch", hide_index=True, height=height, column_config=config)


def _downloads(review, selected):
    # FILE IO LOGIC: Expensive CSV/standalone HTML files are serialized only after their individual prepare buttons are clicked.
    result, config, mapping = review["result"], review["config"], review["mapping"]
    settings = dict(config=asdict(config), mapping=mapping, unit=review["unit"], rows=len(result), source_rows=len(review["data"]), review_scope=review["scope"],
                    selected_cusip=str(review["bond"]), source_fingerprint=review["source_id"], source="applied result",
                    calendar=result.attrs.get("jump_filter", {}).get("calendar"), chart_display=review.get("chart_metadata"))
    files = [("Annotated trades CSV", "annotated_trades.csv", "text/csv", lambda: result.to_csv(index=False).encode("utf-8")),
             ("Bond statistics CSV", "bond_summary.csv", "text/csv", lambda: review["summary"].to_csv(index=False).encode("utf-8")),
             ("Applied settings JSON", "settings.json", "application/json", lambda: json.dumps(settings, indent=2)),
             ("Fitting coverage CSV", "fitting_coverage.csv", "text/csv", lambda: review["fitting"].to_csv(index=False).encode("utf-8")),
             ("Offline interactive chart", "selected_bond.html", "text/html", lambda: review["trade_chart"][1].to_html(include_plotlyjs=True))]
    downloads = review.setdefault("downloads", {})
    for column, (label, filename, mime, prepare) in zip(st.columns(5), files):
        cache_key = (filename, review["bond"] if filename.endswith(".html") or filename == "settings.json" else None)
        if column.button(f"Prepare {label.lower()}", key=f"jf_prepare_{filename}", width="stretch"):
            with st.spinner(f"Preparing {label.lower()} for {len(result):,} reviewed rows…"):
                downloads[cache_key] = prepare()
        if cache_key in downloads:
            column.download_button(label, downloads[cache_key], filename, mime, width="stretch")
    if review.get("session_schedule") is not None:
        st.download_button("Applied session schedule CSV", review["session_schedule"].to_csv(index=False), "session_schedule.csv", "text/csv")


def main():
    # UI LOGIC: Page setup precedes all content and file loading.
    st.set_page_config(page_title="Jump Filter · Bond trade review", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="jf-hero"><div class="jf-brand"><div class="jf-mark">jf</div><div><div class="jf-eyebrow">Bond spread research</div><h1>Jump Filter</h1><p>Review trade quality. Preserve meaningful market moves.</p></div></div><span class="jf-pill">Historical fitting</span></div>', unsafe_allow_html=True)
    st.sidebar.title("Configure review")
    st.sidebar.caption("Set up the filter, then Apply to publish a review.")
    st.sidebar.markdown('<div class="jf-section-label">01 · Data</div>', unsafe_allow_html=True)
    source = st.sidebar.radio("Data source", ["Synthetic demonstration", "Upload CSV"])
    # FILE IO LOGIC: Read uploads as text to retain CUSIP leading zeros; the engine converts only mapped fields.
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CUSIP / time / spread CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with CUSIP, timestamp and spread columns to begin.")
            return
        payload = uploaded.getvalue()
        source_id = sha256(payload).hexdigest()
        source_label = uploaded.name
        # CACHEING LOGIC: Parse one upload once per session; UI reruns reuse the immutable source rather than reading a million rows again.
        loaded = st.session_state.get("jf_loaded_source")
        if loaded is None or loaded[0] != source_id:
            try:
                loaded = (source_id, pd.read_csv(BytesIO(payload), dtype="string"))
                st.session_state["jf_loaded_source"] = loaded
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                return
        data = loaded[1]
    else:
        data, source_id, source_label = _demo("v2"), "synthetic-demo-v2-seed-42", "Synthetic demonstration v2 · not real trades"
    if len(data.columns) < 3 or data.columns.duplicated().any():
        st.error("The input needs at least three distinct columns.")
        return
    # UI LOGIC: Different datasets receive separate mapping widget identities.
    columns = list(data.columns)
    mapping_key = source_id[:16]
    with st.sidebar.expander("Column mappings and units", expanded=False):
        cusip_col = st.selectbox("CUSIP column", columns, index=_column_index(columns, ["CUSIP", "cusip", "bond_id"], 0), key=f"cusip_{mapping_key}")
        time_col = st.selectbox("Time column", columns, index=_column_index(columns, ["time", "timestamp", "trade_time"], 1), key=f"time_{mapping_key}")
        spread_col = st.selectbox("Spread column", columns, index=_column_index(columns, ["spread", "BM_SPREAD", "oas"], 2), key=f"spread_{mapping_key}")
        zone = st.text_input("Timezone for naive timestamps", "UTC", help="Aware timestamps retain their actual instant. Naive timestamps are interpreted in this timezone.")
        unit = st.text_input("Spread unit label", "bp", help="Display label only. Convert spreads upstream if needed; no automatic scaling.")
    mapping = dict(cusip_col=cusip_col, time_col=time_col, spread_col=spread_col, timezone=zone)
    # CACHEING LOGIC: Source and mapping identify a persistent bounded review cache and one positional CUSIP index.
    workspace_key = (source_id, tuple(mapping.items()))
    retained = st.session_state.get("jf_workspace")
    if retained is None or retained[0] != workspace_key:
        retained = (workspace_key, ReviewWorkspace(data, mapping))
        st.session_state["jf_workspace"] = retained
    workspace = retained[1]
    selected_value = st.sidebar.selectbox("Bond / CUSIP", workspace.bonds, format_func=str, key=f"focus_{source_id}_{cusip_col}")
    scope = st.sidebar.selectbox("Review population", ["selected", "all"], format_func=lambda value: "Selected bond · interactive" if value == "selected" else "All bonds · batch")
    st.sidebar.caption(f"{len(data):,} source trades · {len(workspace.bonds):,} CUSIPs. Selected bond evaluates one instrument; All bonds explicitly runs the full population.")
    st.sidebar.markdown('<div class="jf-section-label">02 · Detection</div>', unsafe_allow_html=True)
    method = st.sidebar.selectbox("Method", METHODS, format_func=lambda name: METHOD_LABELS[name], index=list(METHODS).index("consensus"))
    st.sidebar.caption(METHOD_EXPLANATIONS[method]["mode"])
    config_values = {}
    relevant = METHOD_EXPLANATIONS[method]["parameters"]
    # UI LOGIC: Hyperparameter values are shared between the exact input and its slider.
    with st.sidebar.expander("Hyperparameters", expanded=False):
        specs = [("window", "Reference timestamps", 31, 3, 201, 2, True),
                 ("min_neighbors", "Minimum reference timestamps", 6, 2, 60, 1, True),
                 ("threshold", "Score threshold", 4.5, .5, 12., .1, False),
                 ("abs_floor", f"Minimum deviation · {unit}", 1., .001, 50., .1, False),
                 ("reversion_tolerance", "Return tolerance", 2., .1, 6., .1, False),
                 ("iqr_multiplier", "Tukey IQR multiplier", 3., .1, 10., .1, False),
                 ("multiscale_votes", "Required multiscale votes", 2, 1, 3, 1, True),
                 ("trend_penalty", "TV level-change penalty λ", 8., .1, 100., .1, False),
                 ("huber_delta", "Huber clipping δ", 2.5, .1, 10., .1, False),
                 ("max_iter", "Solver iteration limit", 2000, 20, 5000, 20, True),
                 ("tolerance", "Solver convergence tolerance", .0001, .000001, .01, .000001, False),
                 ("alpha", "EWMA learning rate", .2, .01, 1., .01, False),
                 ("persistence", "Persistence cohorts", 3, 2, 10, 1, True)]
        for name, label, value, lower, upper, step, integer in specs:
            if name in relevant:
                config_values[name] = _parameter(name, label, value, lower, upper, step, integer=integer)
        with st.expander("Inactive parameters", expanded=False):
            st.caption("These settings belong to other methods and do not affect the selected filter.")
            for name, label, value, lower, upper, step, integer in specs:
                if name not in relevant:
                    config_values[name] = _parameter(name, label, value, lower, upper, step, integer=integer, disabled=True)
        horizon = st.text_input("Reference horizon", "3D", help="Pandas duration, for example 6h or 3D. In trading mode this means cumulative open-session time.")
        max_gap = st.text_input("Session break gap", "1D", help="Split after this much inactive elapsed time in the selected clock. Trading mode removes configured market closures.")
    with st.sidebar.expander("Trading calendar and liquidity gaps", expanded=False):
        time_basis = st.selectbox("Distance clock", ["trading", "wall"], format_func=lambda value: "Trading time / cumulative open-session time" if value == "trading" else "Wall clock / calendar time")
        use_schedule = st.checkbox("Use authoritative session schedule CSV", disabled=time_basis == "wall")
        schedule_upload = st.file_uploader("Session schedule · open / close timestamps", type=["csv"], disabled=not use_schedule or time_basis == "wall", help="Each row defines one open interval; open and close (or session_open and session_close) require timezone-aware timestamps, e.g. 2026-11-27T08:00:00-05:00,2026-11-27T13:00:00-05:00. Supplied intervals override regular weekday hours and holidays.")
        regular_disabled = time_basis == "wall" or use_schedule
        session_timezone = st.text_input("Market session timezone", "America/New_York", disabled=regular_disabled)
        session_open = st.text_input("Weekday session opens", "08:00", disabled=regular_disabled)
        session_close = st.text_input("Weekday session closes", "18:30", disabled=regular_disabled)
        holidays_text = st.text_input("Closed dates · YYYY-MM-DD, ...", "", disabled=regular_disabled, help="Comma-separated full-session closures. A configured weekday schedule is a research assumption; holidays and early closes are not automatically inferred.")
        st.caption("Trading time removes scheduled overnight, weekend and explicit holiday closures. No-trade gaps during open sessions remain. Local trends use the selected clock; plots show actual UTC timestamps. Supply an authoritative session schedule for a custom calendar.")
    # FILE IO LOGIC: Preserve the uploaded authoritative intervals and their identity with each successful review.
    session_schedule, schedule_error, schedule_id = None, None, None
    if use_schedule and time_basis == "trading":
        if schedule_upload is None:
            schedule_error = "Upload the authoritative session schedule or disable that option."
        else:
            try:
                schedule_payload = schedule_upload.getvalue()
                session_schedule = pd.read_csv(BytesIO(schedule_payload), dtype="string")
                schedule_id = sha256(schedule_payload).hexdigest()
            except Exception as exc:
                schedule_error = f"Could not read session schedule CSV: {exc}"
    pending_config = dict(method=method, horizon=horizon, max_gap=max_gap, time_basis=time_basis,
                          session_timezone=session_timezone, session_open=session_open, session_close=session_close,
                          holidays=tuple(value.strip() for value in holidays_text.split(",") if value.strip()), **config_values)
    with st.sidebar.container(key="jf-apply"):
        applied = st.button("Apply filter", type="primary", width="stretch")
        st.caption("Edits stay pending until you apply. Exports retain the applied settings.")
    signature = dict(source_id=source_id, mapping=mapping, config=pending_config, unit=unit, scope=scope,
                     schedule_fingerprint=schedule_id, schedule_requested=use_schedule and time_basis == "trading")
    # UI LOGIC: A failed Apply leaves the last valid snapshot downloadable and visible.
    if applied:
        try:
            if schedule_error:
                raise ValueError(schedule_error)
            config = FilterConfig(**pending_config)
            with st.spinner("Evaluating trades and preparing the applied review…"):
                record = workspace.review(config, scope=scope, bond=selected_value, session_schedule=session_schedule)
                result = record["result"]
                preview = result.iloc[:0] if selected_value is None else result.iloc[workspace.positions[selected_value]].copy() if scope == "all" else result
                preview_chart = trade_figure(preview, cusip_col=cusip_col, spread_col=spread_col, unit=unit,
                                             title=f'{selected_value} · {METHOD_LABELS[config.method]}', max_gap=config.max_gap)
                review = dict(**record, config=config, mapping=mapping, unit=unit, data=data, workspace=workspace,
                              source_id=source_id, source_label=source_label, signature=signature, downloads={}, trade_chart=(selected_value, preview_chart))
                review["session_schedule"] = session_schedule.copy(deep=True) if session_schedule is not None else None
            st.session_state["jf_review"] = review
            st.session_state.pop("jf_comparison", None)
        except Exception as exc:
            st.error(f"Could not apply settings: {exc}")
    review = st.session_state.get("jf_review")
    if review is None:
        preview_tab, math_tab = st.tabs(["Trade review", "Method & mathematics"])
        with preview_tab:
            st.markdown('<div class="jf-empty"><h2>Start a trade-quality review</h2><p>Choose a method in the sidebar and click <strong>Apply filter</strong>. Explore the synthetic dataset, or upload your own CUSIP, time and spread records.</p></div>', unsafe_allow_html=True)
            steps = [("01", "Choose your evidence", "Load a CSV or use the generated bond scenarios."),
                     ("02", "Set the method", "Tune the threshold, neighborhood and market calendar."),
                     ("03", "Review the result", "Inspect flagged trades, local support and fitting coverage.")]
            for column, (number, title, note) in zip(st.columns(3), steps):
                column.markdown(f'<div class="jf-step"><span>{number}</span><strong>{title}</strong><p>{note}</p></div>', unsafe_allow_html=True)
            _section("Input preview", f"{len(data):,} supplied rows · {source_label}")
            _table(data.head(12))
        with math_tab:
            _method_mathematics(method, pending_config)
        return
    # UI LOGIC: All panels use applied mappings and data, including after unsuccessful edits.
    same_source = source_id == review["source_id"] and mapping == review["mapping"]
    if same_source and selected_value is not None and selected_value != review["bond"]:
        try:
            if review["scope"] == "selected":
                with st.spinner(f"Reviewing {selected_value} using the applied settings…"):
                    record = review["workspace"].review(review["config"], bond=selected_value, session_schedule=review.get("session_schedule"))
                    review = dict(review, **record, downloads={})
                    st.session_state["jf_review"] = review
            else:
                review["bond"] = selected_value
                # CACHEING LOGIC: Only the current bond's prepared chart/settings payloads remain; portfolio CSVs stay reusable.
                review["downloads"] = {key: value for key, value in review.get("downloads", {}).items() if key[1] is None}
            st.session_state.pop("jf_comparison", None)
        except Exception as exc:
            st.error(f"Could not review this bond: {exc}")
    result, applied_mapping = review["result"], review["mapping"]
    active_cusip, active_spread = applied_mapping["cusip_col"], applied_mapping["spread_col"]
    population_label = "All bonds (batch)" if review["scope"] == "all" else f'Selected bond {review["bond"]}'
    st.markdown(f'<div class="jf-context"><span class="jf-pill">Applied review</span><strong>{escape(review["source_label"])}</strong><span class="jf-dot">·</span><span>{escape(population_label)}: {len(result):,} of {len(review["data"]):,} source rows</span><span class="jf-dot">·</span><span>{escape(review["unit"])}</span></div>', unsafe_allow_html=True)
    if signature != review["signature"]:
        st.warning("Pending changes. These results and downloads use the last successful Apply.")
    # UI LOGIC: Selector labels retain exact CUSIP values and do not hide sparse bonds.
    choices = review["workspace"].bonds
    if not choices:
        st.error("No nonmissing CUSIPs are available. Download annotations to inspect invalid inputs.")
        st.dataframe(result, width="stretch", hide_index=True)
        _downloads(review, result.iloc[:0])
        return
    selected_value = review["bond"]
    st.markdown(f'<div class="jf-method-context"><div class="jf-eyebrow">Applied method</div><strong>{escape(str(selected_value))} · {escape(METHOD_LABELS[review["config"].method])}</strong><span>{escape(review["config"].time_basis.title())} clock · {escape(str(review["config"].horizon))} horizon · {escape(str(review["config"].window))} reference timestamps</span></div>', unsafe_allow_html=True)
    selected = result.iloc[review["workspace"].positions[selected_value]].copy() if review["scope"] == "all" else result
    st.caption(f"Charts and headline counts: {selected_value}. Statistics and annotated exports: {population_label}.")
    _metrics(selected)
    trades_tab, statistics_tab, comparison_tab, math_tab, audit_tab = st.tabs(["Trade review", "Statistics", "Compare methods", "Method & mathematics", "Trade audit"])
    with trades_tab:
        # CACHEING LOGIC: Pending parameter edits and table toggles reuse the current view's figures without rebuilding chart arrays.
        cached_chart = review.get("trade_chart")
        if cached_chart is None or cached_chart[0] != selected_value:
            cached_chart = (selected_value, trade_figure(selected, cusip_col=active_cusip, spread_col=active_spread, unit=review["unit"],
                                                       title=f'{selected_value} · {METHOD_LABELS[review["config"].method]}', max_gap=review["config"].max_gap))
            review["trade_chart"] = cached_chart
        figure = cached_chart[1]
        review["chart_metadata"] = figure.layout.meta
        st.plotly_chart(figure, width="stretch")
        st.caption("Hover a trade for its reason, score and local support. Red crosses mark outliers; amber symbols mark transitions. Original spreads are preserved.")
        if figure.layout.meta.get("sampled"):
            counts = figure.layout.meta
            st.caption(f'Chart displays {counts["displayed_trades"]:,} of {counts["total_timed_trades"]:,} timed trades and {counts["displayed_outliers"]:,} of {counts["total_outliers"]:,} flags. Display sampling prioritizes review markers; filtering, statistics and annotated exports use every reviewed trade.')
        with st.expander("How to read this review"):
            st.write("The shaded band is the method's deviation cutoff around its diagnostic baseline. Empty markers are unscored records or ambiguous transitions; they are not automatically eligible for fitting. Session boundaries break the reference path.")
            st.write("Historical screening can use future trades. CUSIP, time and spread identify statistical deviations; they cannot establish retail origin, distress, markup or commission. The baseline is a diagnostic reference, not an observed market mid.")
    with math_tab:
        _method_mathematics(method, pending_config, review["config"].method)
    with statistics_tab:
        _section("Bond overview", "Flag rate uses evaluated trades; coverage uses every supplied row.")
        summary = review["summary"]
        overview = [active_cusip, "rows", "evaluated", "flagged", "fit_eligible", "flagged_rate", "coverage"]
        _table(summary[overview])
        with st.expander("Full bond statistics"):
            _table(summary)
        _section("Fitting coverage and retention")
        _table(review["fitting"])
        st.caption("Hard fit uses jf_fit_eligible (status ok and unflagged). Soft fit uses positive weights among status ok/outlier. Ambiguous transitions, provisional, invalid, unsupported and solver-failure records are excluded by default. Coverage includes all supplied rows; weight sum is not an effective sample size or inverse variance.")
        # CACHEING LOGIC: Keep only the current bond's diagnostic figure, rather than a chart for every source CUSIP.
        cached_diagnostic = review.get("diagnostic_chart")
        if cached_diagnostic is None or cached_diagnostic[0] != selected_value:
            cached_diagnostic = (selected_value, diagnostic_figure(selected, spread_col=active_spread, unit=review["unit"]))
            review["diagnostic_chart"] = cached_diagnostic
        st.plotly_chart(cached_diagnostic[1], width="stretch")
        daily, reasons = st.columns(2)
        audits = audit_tables(selected)
        with daily:
            _section("Observed daily support")
            _table(audits["daily"])
        with reasons:
            _section("Flag and support reasons")
            _table(audits["reasons"])
        st.caption("Flag rate uses evaluated records. Invalid and insufficient-history records remain in the audit table and exports.")
        support_columns = ["jf_time", "jf_status", "jf_n_reference", "jf_reference_span_minutes",
                           "jf_reference_density_per_hour", "jf_scale", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary"]
        with st.expander("Local liquidity and uncertainty · per-trade diagnostics"):
            _table(selected[[name for name in support_columns if name in selected]].head(2000), height=350)
            if len(selected) > 2000:
                st.caption(f"Showing the first 2,000 of {len(selected):,} bond records. Annotated CSV retains the complete review.")
        st.caption("Density counts distinct timestamp cohorts per reference-span hour, using the applied distance clock. Local volatility is the method's diagnostic scale. Unsupported transitions remain explicit; high volatility or low liquidity is not itself an outlier label.")
    with comparison_tab:
        _section("Method sensitivity", "Compare all nine methods on this bond using the applied settings. A higher flag count does not establish better accuracy.")
        compare_key = (review["source_id"], str(selected_value), json.dumps(asdict(review["config"]), sort_keys=True))
        if st.button("Compare methods for this bond"):
            with st.spinner("Evaluating all methods on the selected bond…"):
                try:
                    table = method_comparison(review["workspace"].source_bond(selected_value), review["config"], **applied_mapping,
                                               session_schedule=review.get("session_schedule"))
                    st.session_state["jf_comparison"] = (compare_key, table)
                except Exception as exc:
                    st.error(f"Could not compare methods: {exc}")
        comparison = st.session_state.get("jf_comparison")
        if comparison is not None and comparison[0] == compare_key:
            st.plotly_chart(comparison_figure(comparison[1]), width="stretch")
            _table(comparison[1])
            st.download_button("Method comparison CSV", comparison[1].to_csv(index=False), "method_comparison.csv", "text/csv")
    with audit_tab:
        _section("Trade audit", "Inspect original observations alongside flags, reasons, references and fitting eligibility.")
        flagged_only = st.checkbox("Show flagged trades only", value=False)
        # UI LOGIC: This display-only option never changes the applied flag denominator or downloadable rows.
        shown = selected.loc[selected["jf_is_outlier"]] if flagged_only else selected
        preferred = [active_cusip, applied_mapping["time_col"], active_spread, "jf_status", "jf_reason", "jf_score",
                     "jf_baseline", "jf_residual", "jf_threshold", "jf_n_reference", "jf_weight", "jf_fit_eligible",
                     "jf_n_votes", "jf_n_scales", "jf_solver_iterations", "jf_solver_converged",
                     "jf_reference_density_per_hour", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary", "jf_row_id"]
        _table(shown[[name for name in dict.fromkeys(preferred) if name in shown]].head(2000), height=440)
        if len(shown) > 2000:
            st.caption(f"Showing the first 2,000 of {len(shown):,} matching records. Download annotations for every reviewed row.")
    with st.expander("Export applied review · data, settings and standalone chart"):
        st.caption(f'Every file uses the applied settings. Annotations contain the explicit {review["scope"]} population: {len(result):,} of {len(review["data"]):,} source rows. Prepare each file only when needed; the chart follows the selected bond.')
        _downloads(review, selected)


if __name__ == "__main__":
    # UI LOGIC: Importing this module does not start a server or read trade data.
    main()
