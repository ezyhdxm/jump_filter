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
from jump_filter.dashboard import METHOD_HELP, evaluation_summary, method_comparison, audit_tables
from jump_filter.plots import METHOD_LABELS, trade_figure, diagnostic_figure, comparison_figure

# UI LOGIC: Native Streamlit controls remain keyboard accessible and use a compact research layout.
CSS = """
<style>
.stApp{background:#f4f7fb;color:#19364b}div[data-testid="stMetric"]{background:white;border:1px solid #dce6ee;border-radius:11px;padding:14px 16px}div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] p{color:#435f75!important}div[data-testid="stMetricValue"]{color:#087785}section[data-testid="stSidebar"]{background:#edf3f8}.jf-hero{background:linear-gradient(115deg,#15354c,#096977);color:white;padding:27px 30px;border-radius:15px;margin:0 0 20px}.jf-hero h1{font-size:34px;margin:2px 0 7px;color:white}.jf-hero p{color:#e0eef3;font-size:15px;margin:0}.jf-eyebrow{font-size:11px;font-weight:750;letter-spacing:.18em;color:#bce1e5;text-transform:uppercase}.jf-note{color:#526b7f;font-size:13px;padding:12px 0}.stButton>button[kind="primary"]{background:#087f8c;border-color:#087f8c}.stTabs [data-baseweb="tab-list"]{gap:17px}.stTabs [data-baseweb="tab"]{font-weight:650}div[data-testid="stExpander"]{background:white;border-radius:10px}
</style>
"""


@st.cache_data(show_spinner=False)
def _demo():
    # CACHEING LOGIC: Only generated demo data are cached; user uploads stay in the local session.
    return make_demo()


def _sync(source, target):
    # UI LOGIC: Streamlit callbacks run before the next render, so paired controls update safely.
    st.session_state[target] = st.session_state[source]


def _parameter(name, label, value, lower, upper, step, *, integer=False):
    # UI LOGIC: Stable keys keep the slider and exact input synchronized across reruns.
    slider_key, number_key = f"jf_slider_{name}", f"jf_number_{name}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(value) if integer else float(value)
        st.session_state[number_key] = int(value) if integer else float(value)
    cast = int if integer else float
    lower, upper, step = cast(lower), cast(upper), cast(step)
    st.slider(label, min_value=lower, max_value=upper, step=step, key=slider_key,
              on_change=_sync, args=(slider_key, number_key))
    return st.number_input(f"Exact value · {label}", min_value=lower, max_value=upper, step=step,
                           key=number_key, on_change=_sync, args=(number_key, slider_key),
                           label_visibility="collapsed")


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
                    source_fingerprint=review["source_id"], source="applied result")
    csv = result.to_csv(index=False).encode("utf-8")
    summary_csv = summarize(result, cusip_col=mapping["cusip_col"]).to_csv(index=False).encode("utf-8")
    figure = trade_figure(selected, cusip_col=mapping["cusip_col"], spread_col=mapping["spread_col"], unit=review["unit"], max_gap=config.max_gap)
    columns = st.columns(4)
    columns[0].download_button("Annotated trades CSV", csv, "annotated_trades.csv", "text/csv", width="stretch")
    columns[1].download_button("Bond statistics CSV", summary_csv, "bond_summary.csv", "text/csv", width="stretch")
    columns[2].download_button("Applied settings JSON", json.dumps(settings, indent=2), "settings.json", "application/json", width="stretch")
    columns[3].download_button("Offline interactive chart", figure.to_html(include_plotlyjs=True), "selected_bond.html", "text/html", width="stretch")


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
        data, source_id, source_label = _demo(), "synthetic-demo-seed-42", "Synthetic demonstration · not real trades"
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
    config_values = {}
    # UI LOGIC: Hyperparameter values are shared between the exact input and its slider.
    with st.sidebar.expander("Hyperparameters", expanded=False):
        config_values["window"] = _parameter("window", "Reference timestamps", 31, 3, 201, 2, integer=True)
        config_values["min_neighbors"] = _parameter("min_neighbors", "Minimum reference timestamps", 6, 2, 60, 1, integer=True)
        config_values["threshold"] = _parameter("threshold", "Score threshold", 4.5, .5, 12., .1)
        config_values["abs_floor"] = _parameter("abs_floor", f"Minimum deviation · {unit}", 1., .001, 50., .1)
        config_values["reversion_tolerance"] = _parameter("reversion_tolerance", "Return tolerance", 2., .1, 6., .1)
        config_values["alpha"] = _parameter("alpha", "EWMA learning rate", .2, .01, 1., .01)
        config_values["persistence"] = _parameter("persistence", "Persistence cohorts", 3, 2, 10, 1, integer=True)
        horizon = st.text_input("Reference horizon", "3D", help="Pandas duration, for example 6h or 3D.")
        max_gap = st.text_input("Session break gap", "1D", help="Do not compare trades across an inactivity gap longer than this duration.")
    pending_config = dict(method=method, horizon=horizon, max_gap=max_gap, **config_values)
    applied = st.sidebar.button("Apply filter", type="primary", width="stretch")
    signature = dict(source_id=source_id, mapping=mapping, config=pending_config, unit=unit)
    # UI LOGIC: A failed Apply leaves the last valid snapshot downloadable and visible.
    if applied:
        try:
            config = FilterConfig(**pending_config)
            with st.spinner("Evaluating trades and preparing the applied review…"):
                result = filter_trades(data, config, **mapping)
                preview_values = result[cusip_col].dropna().drop_duplicates()
                preview = _select_bond(result, cusip_col, preview_values.iloc[0]) if len(preview_values) else result.iloc[:0]
                trade_figure(preview, cusip_col=cusip_col, spread_col=spread_col, unit=unit, max_gap=config.max_gap)
                review = dict(result=result, config=config, mapping=mapping, unit=unit, data=data.copy(),
                              source_id=source_id, source_label=source_label, signature=signature)
            st.session_state["jf_review"] = review
            st.session_state.pop("jf_comparison", None)
        except Exception as exc:
            st.error(f"Could not apply settings: {exc}")
    review = st.session_state.get("jf_review")
    if review is None:
        st.info("Choose a method and click Apply filter. The demonstration includes isolated bad prints, trends, level shifts and sparse periods.")
        st.dataframe(data.head(12), width="stretch", hide_index=True)
        return
    # UI LOGIC: All panels use applied mappings and data, including after unsuccessful edits.
    result, applied_mapping = review["result"], review["mapping"]
    active_cusip, active_spread = applied_mapping["cusip_col"], applied_mapping["spread_col"]
    st.caption(f'Applied source: {review["source_label"]} · {len(result):,} rows · {METHOD_LABELS[review["config"].method]} · {review["unit"]}')
    if signature != review["signature"]:
        st.warning("Pending changes. These results and downloads use the last successful Apply.")
    st.markdown('<div class="jf-note">Three columns identify statistical anomalies; they cannot establish retail origin, distress, markup or commission. Centered methods use future observations. Causal robust EWMA uses past observations and marks unconfirmed jumps for review. Robust baselines are diagnostic estimates.</div>', unsafe_allow_html=True)
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
    trades_tab, statistics_tab, comparison_tab, audit_tab = st.tabs(["Trade review", "Statistical dashboard", "Method comparison", "Trade audit"])
    with trades_tab:
        st.plotly_chart(trade_figure(selected, cusip_col=active_cusip, spread_col=active_spread, unit=review["unit"],
                                    title=f'{selected_value} · {METHOD_LABELS[review["config"].method]}',
                                    max_gap=review["config"].max_gap), width="stretch")
        st.caption("Red crosses: flagged trades. Amber diamonds: level change candidates. Amber triangles: provisional causal jumps. All finite timed trades are shown; zoom or box-select to inspect a period.")
    with statistics_tab:
        st.markdown("**All bonds · applied method**")
        st.dataframe(summarize(result, cusip_col=active_cusip), width="stretch", hide_index=True)
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
    with comparison_tab:
        st.caption("Run all methods on this bond with the applied hyperparameters. Differences describe sensitivity; a higher flag rate does not establish better filtering.")
        compare_key = (review["source_id"], str(selected_value), json.dumps(asdict(review["config"]), sort_keys=True))
        if st.button("Compare methods for this bond"):
            with st.spinner("Evaluating all methods on the selected bond…"):
                try:
                    table = method_comparison(_select_bond(review["data"], active_cusip, selected_value), review["config"], **applied_mapping)
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
                     "jf_baseline", "jf_residual", "jf_threshold", "jf_n_reference", "jf_weight", "jf_row_id"]
        st.dataframe(shown[list(dict.fromkeys(preferred))], width="stretch", hide_index=True)
    st.markdown("**Export applied review**")
    _downloads(review, selected)


if __name__ == "__main__":
    # UI LOGIC: Importing this module does not start a server or read trade data.
    main()
