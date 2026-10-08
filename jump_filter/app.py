"""Local Streamlit workbench. Run: streamlit run jump_filter/app.py."""
# SETUP LOGIC: Browser dependencies are loaded only by this optional application.
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import numpy as np
import pandas as pd
import streamlit as st
from jump_filter import FilterConfig, METHODS, filter_trades, make_demo, summarize
from jump_filter.dashboard import evaluation_summary, method_comparison, audit_tables, fitting_statistics
from jump_filter.explanations import METHOD_HELP, METHOD_EXPLANATIONS, COMMON_STEPS, CLOCK_STEPS, FITTING_STEPS, PARAMETER_HELP, math_display_blocks
from jump_filter.plots import METHOD_LABELS, trade_figure, diagnostic_figure, comparison_figure

# UI LOGIC: Native Streamlit controls remain keyboard accessible and use a compact research layout.
CSS = """
<style>
.stApp{background:#f4f7fb;color:#19364b}div[data-testid="stMetric"]{background:white;border:1px solid #dce6ee;border-radius:11px;padding:14px 16px}div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] p{color:#435f75!important}div[data-testid="stMetricValue"]{color:#087785}section[data-testid="stSidebar"]{background:#edf3f8}.jf-hero{background:linear-gradient(115deg,#15354c,#096977);color:white;padding:27px 30px;border-radius:15px;margin:0 0 20px}.jf-hero h1{font-size:34px;margin:2px 0 7px;color:white}.jf-hero p{color:#e0eef3;font-size:15px;margin:0}.jf-eyebrow{font-size:11px;font-weight:750;letter-spacing:.18em;color:#bce1e5;text-transform:uppercase}.jf-note{color:#526b7f;font-size:13px;padding:12px 0}.stButton>button[kind="primary"]{background:#087f8c;border-color:#087f8c}.stTabs [data-baseweb="tab-list"]{gap:17px}.stTabs [data-baseweb="tab"]{font-weight:650}div[data-testid="stExpander"]{background:white;border-radius:10px}
div[data-testid="stLatex"]{max-width:100%;overflow-x:auto;padding:2px 0}div[data-testid="stLatex"] .katex-display{margin:.3em 0;max-width:100%;overflow-x:auto}div[data-testid="stLatex"] .katex-display>.katex{text-align:left}
</style>
"""


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
    st.write(card["summary"])
    for title, prose, formula in COMMON_STEPS + CLOCK_STEPS + card["steps"] + FITTING_STEPS:
        st.markdown(f"**{title}**")
        st.write(prose)
        for block in math_display_blocks(formula):
            st.latex(block)
    st.markdown("**Numeric example / 数值例子**")
    st.write(card["example"])
    st.markdown("**Assumptions and limits / 假设与局限**")
    st.write(card["tradeoffs"])
    st.markdown("**Selected parameters / 当前待应用参数**")
    records = [(name, str(pending_config[name]), PARAMETER_HELP[name]) for name in card["parameters"]]
    records += [(name, str(pending_config[name]), explanation) for name, explanation in
                [("horizon", "真实时间距离限制；multiscale 使用其 1、2、4 倍。"),
                 ("max_gap", "超过此 inactivity gap 独立分段，任何方法都不跨段借用参考。"),
                 ("time_basis", "trading = 累计开市时间；wall = 日历时间。"),
                 ("session_timezone", "交易 session 时区，与原始 naive timestamps 的解析时区分别配置。"),
                 ("session_open", "常规工作日 session 开始时间。"), ("session_close", "常规工作日 session 结束时间。"),
                 ("holidays", "显式整日休市；完整特殊 calendar 可通过 API session_schedule 输入。")]]
    st.dataframe(pd.DataFrame(records, columns=["parameter", "pending value", "effect"]), width="stretch", hide_index=True)
    title, url = card["reference"]
    st.markdown(f"Reference: [{title}]({url})")


def _column_index(columns, aliases, fallback):
    # UI LOGIC: Suggested mappings are visible and editable; the numerical engine never infers units.
    matches = {str(column).lower(): i for i, column in enumerate(columns)}
    return next((matches[name.lower()] for name in aliases if name.lower() in matches), min(fallback, len(columns) - 1))


def _select_bond(frame, cusip_col, value):
    # Input: CUSIP=['A','B','A'], spread=[100,200,101], selected='A'.
    # Output: CUSIP=['A','A'],spread=[100,101], in source order.
    # Trick: Exact values preserve leading-zero strings; no string normalization or row-index joins are used.
    # CORE LOGIC: STEP 1 — Select one bond for charts and method comparison.
    return frame.loc[frame[cusip_col].eq(value)].copy()


def _metrics(rows):
    # UI LOGIC: Counts distinguish algorithm evaluation from missing inputs and limited support.
    stats = evaluation_summary(rows)
    labels = [("Trades", f'{stats["total"]:,}'), ("Evaluated", f'{stats["evaluated"]:,}'),
              ("Flagged", f'{stats["outliers"]:,}'),
              ("Flag rate", f'{stats["flag_rate"]:.1%}' if np.isfinite(stats["flag_rate"]) else "—"),
              ("Low support", f'{stats["unsupported"]:,}'), ("Invalid", f'{stats["invalid"]:,}')]
    for column, (label, value) in zip(st.columns(6), labels):
        column.metric(label, value)
    return stats


def _downloads(review, selected):
    # FILE IO LOGIC: Downloads are generated from the successful applied snapshot, never pending settings.
    result, config, mapping = review["result"], review["config"], review["mapping"]
    settings = dict(config=asdict(config), mapping=mapping, unit=review["unit"], rows=len(result),
                    source_fingerprint=review["source_id"], source="applied result",
                    calendar=result.attrs.get("jump_filter", {}).get("calendar"))
    csv = result.to_csv(index=False).encode("utf-8")
    summary_csv = summarize(result, cusip_col=mapping["cusip_col"]).to_csv(index=False).encode("utf-8")
    figure = trade_figure(selected, cusip_col=mapping["cusip_col"], spread_col=mapping["spread_col"], unit=review["unit"], max_gap=config.max_gap)
    columns = st.columns(4)
    columns[0].download_button("Annotated trades CSV", csv, "annotated_trades.csv", "text/csv", width="stretch")
    columns[1].download_button("Bond statistics CSV", summary_csv, "bond_summary.csv", "text/csv", width="stretch")
    columns[2].download_button("Applied settings JSON", json.dumps(settings, indent=2), "settings.json", "application/json", width="stretch")
    columns[3].download_button("Offline interactive chart", figure.to_html(include_plotlyjs=True), "selected_bond.html", "text/html", width="stretch")
    extra = st.columns(2)
    extra[0].download_button("Fitting coverage CSV", fitting_statistics(result, cusip_col=mapping["cusip_col"]).to_csv(index=False), "fitting_coverage.csv", "text/csv", width="stretch")
    if review.get("session_schedule") is not None:
        extra[1].download_button("Applied session schedule CSV", review["session_schedule"].to_csv(index=False), "session_schedule.csv", "text/csv", width="stretch")


def main():
    # UI LOGIC: Page setup precedes all content and file loading.
    st.set_page_config(page_title="Jump Filter · Bond trade review", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="jf-hero"><div class="jf-eyebrow">Bond trade quality / Research workbench</div><h1>Jump Filter</h1><p>Flag unusual spreads, inspect local evidence, and preserve meaningful spread moves.</p></div>', unsafe_allow_html=True)
    st.sidebar.title("Review controls")
    source = st.sidebar.radio("Data source", ["Synthetic demonstration", "Upload CSV"])
    # FILE IO LOGIC: Read uploads as text to retain CUSIP leading zeros; the engine converts only mapped fields.
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CUSIP / time / spread CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with CUSIP, timestamp and spread columns to begin.")
            return
        payload = uploaded.getvalue()
        try:
            data = pd.read_csv(BytesIO(payload), dtype="string")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            return
        source_id = sha256(payload).hexdigest()
        source_label = uploaded.name
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
    method = st.sidebar.selectbox("Method", METHODS, format_func=lambda name: METHOD_LABELS[name], index=list(METHODS).index("consensus"))
    st.sidebar.caption(METHOD_HELP[method])
    st.sidebar.caption("离线历史筛选可使用后续交易；Causal EWMA 仅作在线比较。无关参数已禁用。")
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
            config_values[name] = _parameter(name, label, value, lower, upper, step,
                                             integer=integer, disabled=name not in relevant)
        horizon = st.text_input("Reference horizon", "3D", help="Pandas duration, for example 6h or 3D. In trading mode this means cumulative open-session time.")
        max_gap = st.text_input("Session break gap", "1D", help="Split after this much inactive elapsed time in the selected clock. Trading mode removes configured market closures.")
    with st.sidebar.expander("Trading calendar and liquidity gaps", expanded=False):
        time_basis = st.selectbox("Distance clock", ["trading", "wall"], format_func=lambda value: "Trading time / 开市累计时间" if value == "trading" else "Wall clock / 日历时间")
        use_schedule = st.checkbox("Use authoritative session schedule CSV", disabled=time_basis == "wall")
        schedule_upload = st.file_uploader("Session schedule · open / close timestamps", type=["csv"], disabled=not use_schedule or time_basis == "wall", help="Each row defines one open interval; open and close (or session_open and session_close) require timezone-aware timestamps, e.g. 2026-11-27T08:00:00-05:00,2026-11-27T13:00:00-05:00. Supplied intervals override regular weekday hours and holidays.")
        regular_disabled = time_basis == "wall" or use_schedule
        session_timezone = st.text_input("Market session timezone", "America/New_York", disabled=regular_disabled)
        session_open = st.text_input("Weekday session opens", "08:00", disabled=regular_disabled)
        session_close = st.text_input("Weekday session closes", "18:30", disabled=regular_disabled)
        holidays_text = st.text_input("Closed dates · YYYY-MM-DD, ...", "", disabled=regular_disabled, help="Comma-separated full-session closures. A configured weekday schedule is a research assumption; holidays and early closes are not automatically inferred.")
        st.caption("Trading mode 压缩 scheduled overnight、weekend 和显式 holidays；开市时没有成交的 gap 仍保留。局部 trend 使用选定时钟，图仍显示真实 UTC。完整特殊 calendar 可通过 API session_schedule 输入。")
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
    applied = st.sidebar.button("Apply filter", type="primary", width="stretch")
    signature = dict(source_id=source_id, mapping=mapping, config=pending_config, unit=unit,
                     schedule_fingerprint=schedule_id, schedule_requested=use_schedule and time_basis == "trading")
    # UI LOGIC: A failed Apply leaves the last valid snapshot downloadable and visible.
    if applied:
        try:
            if schedule_error:
                raise ValueError(schedule_error)
            config = FilterConfig(**pending_config)
            with st.spinner("Evaluating trades and preparing the applied review…"):
                result = filter_trades(data, config, **mapping, session_schedule=session_schedule)
                preview_values = result[cusip_col].dropna().drop_duplicates()
                preview = _select_bond(result, cusip_col, preview_values.iloc[0]) if len(preview_values) else result.iloc[:0]
                trade_figure(preview, cusip_col=cusip_col, spread_col=spread_col, unit=unit, max_gap=config.max_gap)
                review = dict(result=result, config=config, mapping=mapping, unit=unit, data=data.copy(),
                              source_id=source_id, source_label=source_label, signature=signature)
                review["session_schedule"] = session_schedule.copy(deep=True) if session_schedule is not None else None
            st.session_state["jf_review"] = review
            st.session_state.pop("jf_comparison", None)
        except Exception as exc:
            st.error(f"Could not apply settings: {exc}")
    review = st.session_state.get("jf_review")
    if review is None:
        preview_tab, math_tab = st.tabs(["Trade review", "Method & mathematics"])
        with preview_tab:
            st.info("Choose a method and click Apply filter. Method & mathematics already explains your selected method, with formulas and numerical examples.")
            st.dataframe(data.head(12), width="stretch", hide_index=True)
        with math_tab:
            _method_mathematics(method, pending_config)
        return
    # UI LOGIC: All panels use applied mappings and data, including after unsuccessful edits.
    result, applied_mapping = review["result"], review["mapping"]
    active_cusip, active_spread = applied_mapping["cusip_col"], applied_mapping["spread_col"]
    st.caption(f'Applied source: {review["source_label"]} · {len(result):,} rows · {METHOD_LABELS[review["config"].method]} · {review["unit"]}')
    if signature != review["signature"]:
        st.warning("Pending changes. These results and downloads use the last successful Apply.")
    st.markdown('<div class="jf-note">历史拟合可使用前后交易确认异常。Three columns identify statistical anomalies; they cannot establish retail origin, distress, markup or commission. Baselines are diagnostic references. Hard fitting uses evaluated, unflagged records; see fitting coverage and the mathematics tab.</div>', unsafe_allow_html=True)
    # UI LOGIC: Selector labels retain exact CUSIP values and do not hide sparse bonds.
    choices = list(result[active_cusip].dropna().drop_duplicates())
    if not choices:
        st.error("No nonmissing CUSIPs are available. Download annotations to inspect invalid inputs.")
        st.dataframe(result, width="stretch", hide_index=True)
        _downloads(review, result.iloc[:0])
        return
    selected_value = st.selectbox("Bond / CUSIP", choices, format_func=str, key=f'focus_{review["source_id"]}_{active_cusip}')
    selected = _select_bond(result, active_cusip, selected_value)
    _metrics(selected)
    trades_tab, math_tab, statistics_tab, comparison_tab, audit_tab = st.tabs(["Trade review", "Method & mathematics", "Statistical dashboard", "Method comparison", "Trade audit"])
    with trades_tab:
        st.plotly_chart(trade_figure(selected, cusip_col=active_cusip, spread_col=active_spread, unit=review["unit"],
                                    title=f'{selected_value} · {METHOD_LABELS[review["config"].method]}',
                                    max_gap=review["config"].max_gap), width="stretch")
        st.caption("Red crosses: flagged trades. Amber diamonds: level change candidates or ambiguous transitions. Amber triangles: provisional causal jumps. All finite timed trades are shown; hover for local support, volatility and trading-time gaps.")
    with math_tab:
        _method_mathematics(method, pending_config, review["config"].method)
    with statistics_tab:
        st.markdown("**All bonds · applied method**")
        st.dataframe(summarize(result, cusip_col=active_cusip), width="stretch", hide_index=True)
        st.markdown("**Fitting coverage and retention / 拟合样本统计**")
        st.dataframe(fitting_statistics(result, cusip_col=active_cusip), width="stretch", hide_index=True)
        st.caption("Hard fit uses jf_fit_eligible (status ok and unflagged). Soft fit uses positive weights among status ok/outlier. Ambiguous transitions, provisional, invalid, unsupported and solver-failure records are excluded by default. Coverage includes all supplied rows; weight sum is not an effective sample size or inverse variance.")
        st.plotly_chart(diagnostic_figure(selected, spread_col=active_spread, unit=review["unit"]), width="stretch")
        daily, reasons = st.columns(2)
        audits = audit_tables(selected)
        with daily:
            st.markdown("**Selected bond · daily support**")
            st.dataframe(audits["daily"], width="stretch", hide_index=True)
        with reasons:
            st.markdown("**Selected bond · flag and support reasons**")
            st.dataframe(audits["reasons"], width="stretch", hide_index=True)
        st.caption("Flag rate uses evaluated records. Invalid and insufficient-history records remain in the audit table and exports.")
        support_columns = ["jf_time", "jf_status", "jf_n_reference", "jf_reference_span_minutes",
                           "jf_reference_density_per_hour", "jf_scale", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary"]
        st.markdown("**Selected bond · local liquidity and uncertainty diagnostics**")
        st.dataframe(selected[[name for name in support_columns if name in selected]], width="stretch", hide_index=True)
        st.caption("Density counts distinct timestamp cohorts per reference-span hour, using the applied distance clock. Local volatility is the method's diagnostic scale. Unsupported transitions remain explicit; high volatility or low liquidity is not itself an outlier label.")
    with comparison_tab:
        st.caption("Run all methods on this bond with the applied hyperparameters. Differences describe sensitivity; a higher flag rate does not establish better filtering.")
        compare_key = (review["source_id"], str(selected_value), json.dumps(asdict(review["config"]), sort_keys=True))
        if st.button("Compare methods for this bond"):
            with st.spinner("Evaluating all methods on the selected bond…"):
                try:
                    table = method_comparison(_select_bond(review["data"], active_cusip, selected_value), review["config"], **applied_mapping,
                                               session_schedule=review.get("session_schedule"))
                    st.session_state["jf_comparison"] = (compare_key, table)
                except Exception as exc:
                    st.error(f"Could not compare methods: {exc}")
        comparison = st.session_state.get("jf_comparison")
        if comparison is not None and comparison[0] == compare_key:
            st.plotly_chart(comparison_figure(comparison[1]), width="stretch")
            st.dataframe(comparison[1], width="stretch", hide_index=True)
            st.download_button("Method comparison CSV", comparison[1].to_csv(index=False), "method_comparison.csv", "text/csv")
    with audit_tab:
        flagged_only = st.checkbox("Show flagged trades only", value=False)
        # UI LOGIC: This display-only option never changes the applied flag denominator or downloadable rows.
        shown = selected.loc[selected["jf_is_outlier"]] if flagged_only else selected
        preferred = [active_cusip, applied_mapping["time_col"], active_spread, "jf_status", "jf_reason", "jf_score",
                     "jf_baseline", "jf_residual", "jf_threshold", "jf_n_reference", "jf_weight", "jf_fit_eligible",
                     "jf_n_votes", "jf_n_scales", "jf_solver_iterations", "jf_solver_converged",
                     "jf_reference_density_per_hour", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary", "jf_row_id"]
        st.dataframe(shown[[name for name in dict.fromkeys(preferred) if name in shown]], width="stretch", hide_index=True)
    st.markdown("**Export applied review**")
    _downloads(review, selected)


if __name__ == "__main__":
    # UI LOGIC: Importing this module does not start a server or read trade data.
    main()
