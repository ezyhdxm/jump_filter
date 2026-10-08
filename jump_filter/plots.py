"""Shared interactive charts for the notebook and browser dashboards."""
# SETUP LOGIC: Plotly remains an optional dependency until a dashboard is requested.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONFIGURATION LOGIC: Distinct markers complement the colors for accessible status reading.
INK = "#19364b"
TEAL = "#087f8c"
RED = "#db4759"
AMBER = "#bc7b19"
METHOD_LABELS = {
    "hampel": "Local Hampel", "local_linear": "Robust local linear",
    "jump_reversion": "Jump and reversion", "causal_ewma": "Causal robust EWMA",
    "consensus": "Conservative consensus",
}


def _style(figure, *, height=660, trade=False):
    # PLOTTING LOGIC: Shared typography and white cards match both dashboard surfaces.
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color=INK),
        paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=65, r=28, t=160 if trade else 65, b=45),
        legend=dict(orientation="h", y=1.20 if trade else 1.10, yanchor="top" if trade else "bottom",
                    x=0, font=dict(size=11)),
        hovermode="closest", uirevision="jump-filter-chart",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False)
    if trade:
        # PLOTTING LOGIC: reserve space for wrapped legends in a narrow notebook/browser pane.
        figure.update_layout(title=dict(y=.98, yanchor="top"))
    return figure


def _band_coordinates(rows, max_gap):
    # PLOTTING LOGIC: Insert drawing breaks across inactive sessions without adding or modifying trade records.
    coordinates = [[], [], [], [], []]
    previous, gap = None, pd.Timedelta(max_gap)
    for _, row in rows.iterrows():
        timestamp, baseline, cutoff = row["jf_time"], row["jf_baseline"], row["jf_threshold"]
        if previous is not None and timestamp - previous > gap:
            for values in coordinates:
                values.append(None)
        for values, value in zip(coordinates, [timestamp, baseline, baseline + cutoff, baseline - cutoff, cutoff]):
            values.append(value)
        previous = timestamp
    return coordinates


def trade_figure(annotated, *, cusip_col="CUSIP", spread_col="spread", unit="bp", title=None, max_gap="1D"):
    """Show every timed trade, algorithm flags, baseline and signed residual."""
    # PLOTTING LOGIC: Sort only the display copy; exported rows retain their original order and index.
    rows = annotated.loc[annotated["jf_time"].notna()].sort_values(["jf_time", "jf_row_id"], kind="stable")
    value = pd.to_numeric(rows[spread_col], errors="coerce")
    finite = np.isfinite(value.to_numpy(dtype=float, na_value=np.nan))
    rows, value = rows.loc[finite], value.loc[finite]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                           row_heights=[.68, .32], subplot_titles=("Trade spread and robust baseline", "Signed deviation from baseline"))
    if rows.empty:
        # PLOTTING LOGIC: Missing chart coordinates receive an explicit empty state.
        figure.add_annotation(text="No trades with a finite spread and usable timestamp", x=.5, y=.6,
                              xref="paper", yref="paper", showarrow=False)
        return _style(figure, height=760, trade=True)
    # PLOTTING LOGIC: Exact source positions make hover identifiers safe with duplicate dataframe indexes.
    hover = np.column_stack([
        rows["jf_row_id"].astype(str), rows[cusip_col].astype(str), rows["jf_status"],
        rows["jf_reason"], rows["jf_score"].round(3), rows["jf_n_reference"],
    ])
    hover_text = ("CUSIP %{customdata[1]}<br>%{x}<br>Spread %{y:.4f} " + unit +
                  "<br>Status %{customdata[2]}<br>Score %{customdata[4]}<br>References %{customdata[5]}"
                  "<br>Row position %{customdata[0]}<br>%{customdata[3]}<extra></extra>")
    evaluated = rows["jf_status"].isin(["ok", "outlier", "provisional_jump"])
    classes = [
        ("Evaluated / unflagged", ~rows["jf_is_outlier"] & evaluated, TEAL, "circle", 6),
        ("Insufficient / invalid support", ~rows["jf_is_outlier"] & ~evaluated, "#94a3b3", "circle-open", 7),
        ("Flagged outlier", rows["jf_is_outlier"], RED, "x", 10),
    ]
    for label, mask, color, symbol, size in classes:
        # PLOTTING LOGIC: Outliers remain visible at their original spread values.
        figure.add_trace(go.Scattergl(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                     marker=dict(color=color, size=size, symbol=symbol, opacity=.8),
                                     customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: The raw-unit cutoff band is diagnostic; line breaks preserve inactive sessions.
    band_time, baseline, upper, lower, raw_cutoff = _band_coordinates(rows, max_gap)
    figure.add_trace(go.Scatter(x=band_time, y=upper, mode="lines", line=dict(width=0),
                               name="Local cutoff band", showlegend=False, hoverinfo="skip", connectgaps=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=lower, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor="rgba(8,127,140,.08)", name="Local cutoff band",
                               hoverinfo="skip", connectgaps=False), row=1, col=1)
    # PLOTTING LOGIC: Baselines are diagnostics rather than quoted executable mid prices.
    figure.add_trace(go.Scatter(x=band_time, y=baseline, mode="lines", name="Robust baseline",
                               line=dict(color=INK, width=1.8), connectgaps=False,
                               hovertemplate="%{x}<br>Baseline %{y:.4f} " + unit + "<extra></extra>"), row=1, col=1)
    regime = rows["jf_regime_change"].fillna(False)
    provisional = rows["jf_status"].eq("provisional_jump")
    for label, mask, symbol in [("Level change candidate", regime, "diamond-open"),
                                ("Provisional jump", provisional, "triangle-up-open")]:
        figure.add_trace(go.Scatter(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                   marker=dict(color=AMBER, symbol=symbol, size=12, line=dict(width=2)),
                                   customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: The residual plot uses the engine's raw-unit diagnostic without recomputing it.
    figure.add_trace(go.Scattergl(x=rows["jf_time"], y=rows["jf_residual"], mode="markers", showlegend=False,
                                 marker=dict(color=np.where(rows["jf_is_outlier"], RED, TEAL), size=5),
                                 hovertemplate="%{x}<br>Residual %{y:.4f} " + unit + "<extra></extra>"), row=2, col=1)
    figure.add_hline(y=0, line_color="#8798a8", line_dash="dot", row=2, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=raw_cutoff, mode="lines", showlegend=False,
                               line=dict(color="#8fb5be", width=1, dash="dot"), hoverinfo="skip"), row=2, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=[-value if value is not None else None for value in raw_cutoff], mode="lines", showlegend=False,
                               line=dict(color="#8fb5be", width=1, dash="dot"), hoverinfo="skip"), row=2, col=1)
    figure.update_yaxes(title_text=f"Spread · {unit}", row=1, col=1)
    figure.update_yaxes(title_text=f"Residual · {unit}", row=2, col=1)
    figure.update_xaxes(title_text="Time · UTC", row=2, col=1)
    figure.update_layout(title=dict(text=title or "Selected bond", font=dict(size=17), x=.02))
    return _style(figure, height=760, trade=True)


def diagnostic_figure(annotated, *, spread_col="spread", unit="bp"):
    """Contrast retained and flagged deviations and show status coverage."""
    # PLOTTING LOGIC: Histogram counts use all finite residuals, without trimming tails.
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Residual distribution", "Evaluation coverage"),
                           column_widths=[.65, .35], horizontal_spacing=.12)
    for flag, label, color in [(False, "Retained / reviewed", TEAL), (True, "Flagged outlier", RED)]:
        rows = annotated.loc[annotated["jf_is_outlier"].eq(flag)]
        figure.add_trace(go.Histogram(x=rows["jf_residual"], name=label, marker_color=color,
                                     opacity=.75, nbinsx=45), row=1, col=1)
    # Input: jf_status=['ok','outlier','ok','insufficient_history'].
    # Output: counts={'ok':2,'outlier':1,'insufficient_history':1}.
    # Trick: Missing diagnostic fields do not become zero residuals; all statuses retain explicit counts.
    # CORE LOGIC: STEP 1 — Count evaluation statuses before displaying their coverage.
    counts = annotated["jf_status"].value_counts(dropna=False)
    # PLOTTING LOGIC: Draw the already calculated coverage table with stable status colors.
    figure.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, showlegend=False,
                           marker_color=[RED if name == "outlier" else TEAL if name == "ok" else AMBER for name in counts.index]), row=1, col=2)
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text=f"Residual · {unit}", row=1, col=1)
    figure.update_yaxes(title_text="Trades", row=1, col=1)
    figure.update_yaxes(title_text="Trades", row=1, col=2)
    return _style(figure, height=390)


def comparison_figure(table):
    """Display method flag counts without implying that more flags are better."""
    # PLOTTING LOGIC: Common methods use a stable display order and exact counts in hover text.
    figure = go.Figure(go.Bar(x=[METHOD_LABELS.get(name, name) for name in table["method"]],
                             y=table["outliers"], marker_color=TEAL,
                             customdata=table[["flag_rate", "evaluated"]].to_numpy(),
                             hovertemplate="%{x}<br>Outliers %{y}<br>Flag rate %{customdata[0]:.1%}"
                                           "<br>Evaluated %{customdata[1]}<extra></extra>"))
    figure.update_yaxes(title_text="Flagged trades")
    figure.update_layout(title="Method sensitivity on the selected bond", showlegend=False)
    return _style(figure, height=370)
