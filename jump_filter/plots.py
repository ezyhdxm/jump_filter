"""Shared interactive charts for the notebook and browser dashboards."""
# SETUP LOGIC: Plotly remains an optional dependency until a dashboard is requested.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONFIGURATION LOGIC: Distinct markers complement the colors for accessible status reading.
INK = "#142D3D"
TEAL = "#087F8C"
RED = "#D84C62"
AMBER = "#B78028"
MUTED = "#657887"
GRID = "#EDF1F4"
METHOD_LABELS = {
    "hampel": "Local Hampel", "local_linear": "Robust local linear",
    "rolling_iqr": "Rolling IQR fences", "multiscale": "Multiscale Hampel confirmation",
    "local_piecewise": "Two-sided local piecewise trend",
    "robust_trend": "Offline robust TV trend",
    "jump_reversion": "Jump and reversion", "causal_ewma": "Causal robust EWMA · optional online",
    "consensus": "Conservative consensus",
}


def _style(figure, *, height=520, trade=False):
    # PLOTTING LOGIC: Shared typography and white cards match both dashboard surfaces.
    figure.update_layout(
        template="plotly_white", height=height, autosize=True,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color=INK, size=12),
        paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=64, r=24, t=104 if trade else 64, b=48),
        legend=dict(orientation="h", y=1.12 if trade else 1.08, yanchor="bottom",
                    x=0, font=dict(size=10, color=MUTED), itemsizing="constant", tracegroupgap=4, traceorder="normal"),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="white", size=12), align="left"),
        hovermode="closest", uirevision="jump-filter-chart", modebar=dict(color=MUTED, activecolor=TEAL),
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color=MUTED),
                        title_font=dict(size=11, color=MUTED), automargin=True, ticklen=0)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color=MUTED),
                        title_font=dict(size=11, color=MUTED), automargin=True, ticklen=0)
    figure.update_annotations(font=dict(size=12, color=INK))
    if trade:
        # PLOTTING LOGIC: Keep the title, compact legend and panel labels in separate vertical bands.
        figure.update_layout(title=dict(y=.985, yanchor="top", x=.015, xanchor="left"))
    return figure


def _band_coordinates(rows, max_gap):
    # PLOTTING LOGIC: Insert drawing breaks across inactive sessions without adding or modifying trade records.
    coordinates = [[], [], [], [], []]
    previous, gap = None, pd.Timedelta(max_gap)
    for _, row in rows.iterrows():
        timestamp, baseline, cutoff = row["jf_time"], row["jf_baseline"], row["jf_threshold"]
        if previous is not None and (timestamp - previous > gap or bool(row.get("jf_session_boundary", False))):
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
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.14,
                           row_heights=[.7, .3], subplot_titles=("Trades, baseline & cutoff band", "Deviation from baseline"))
    if rows.empty:
        # PLOTTING LOGIC: Missing chart coordinates receive an explicit empty state.
        figure.add_annotation(text="No trades with a finite spread and usable timestamp", x=.5, y=.6,
                              xref="paper", yref="paper", showarrow=False)
        return _style(figure, height=600, trade=True)
    # PLOTTING LOGIC: Exact source positions make hover identifiers safe with duplicate dataframe indexes.
    hover = np.column_stack([
        rows["jf_row_id"].astype(str), rows[cusip_col].astype(str), rows["jf_status"],
        rows["jf_reason"], rows["jf_score"].round(3), rows["jf_n_reference"],
        rows.get("jf_reference_density_per_hour", pd.Series(np.nan, index=rows.index)).round(3),
        rows.get("jf_local_volatility", rows["jf_scale"]).round(4),
        rows.get("jf_gap_minutes", pd.Series(np.nan, index=rows.index)).round(2),
        rows.get("jf_wall_gap_minutes", pd.Series(np.nan, index=rows.index)).round(2),
        rows.get("jf_session_boundary", pd.Series(False, index=rows.index)).astype(str),
    ])
    hover_text = ("<b>%{customdata[1]}</b> · %{customdata[2]}<br>%{x}<br><b>Spread %{y:.4f} " + unit +
                  "</b> · Score %{customdata[4]}<br>References %{customdata[5]} · Cohorts/hour %{customdata[6]}"
                  "<br>Local volatility %{customdata[7]} " + unit +
                  "<br>Active-clock gap %{customdata[8]} min · Wall-clock gap %{customdata[9]} min"
                  "<br>Session boundary %{customdata[10]} · Row position %{customdata[0]}"
                  "<br>%{customdata[3]}<extra></extra>")
    evaluated = rows["jf_status"].isin(["ok", "outlier", "provisional_jump"])
    ambiguous = rows["jf_status"].eq("ambiguous_transition")
    classes = [
        ("Evaluated", ~rows["jf_is_outlier"] & evaluated, TEAL, "circle", 5.5),
        ("Unscored", ~rows["jf_is_outlier"] & ~evaluated & ~ambiguous, MUTED, "circle-open", 7),
        ("Ambiguous turn", ambiguous, AMBER, "diamond-open", 9),
        ("Outlier", rows["jf_is_outlier"], RED, "x", 10),
    ]
    for label, mask, color, symbol, size in classes:
        # PLOTTING LOGIC: Outliers remain visible at their original spread values.
        figure.add_trace(go.Scattergl(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                     marker=dict(color=color, size=size, symbol=symbol, opacity=.85), showlegend=bool(mask.any()),
                                     customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: The raw-unit cutoff band is diagnostic; line breaks preserve inactive sessions.
    band_time, baseline, upper, lower, raw_cutoff = _band_coordinates(rows, max_gap)
    figure.add_trace(go.Scatter(x=band_time, y=upper, mode="lines", line=dict(width=0),
                               name="Cutoff band", showlegend=False, hoverinfo="skip", connectgaps=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=lower, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor="rgba(8,127,140,.075)", name="Cutoff band",
                               hoverinfo="skip", connectgaps=False), row=1, col=1)
    # PLOTTING LOGIC: Baselines are diagnostics rather than quoted executable mid prices.
    figure.add_trace(go.Scatter(x=band_time, y=baseline, mode="lines", name="Baseline",
                               line=dict(color=INK, width=1.8), connectgaps=False,
                               hovertemplate="%{x}<br>Baseline %{y:.4f} " + unit + "<extra></extra>"), row=1, col=1)
    regime = rows["jf_regime_change"].fillna(False)
    provisional = rows["jf_status"].eq("provisional_jump")
    for label, mask, symbol in [("Level change", regime & ~ambiguous, "diamond-open"),
                                ("Provisional", provisional, "triangle-up-open")]:
        figure.add_trace(go.Scatter(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                   marker=dict(color=AMBER, symbol=symbol, size=11, line=dict(width=1.5)), showlegend=bool(mask.any()),
                                   customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: The residual plot uses the engine's raw-unit diagnostic without recomputing it.
    figure.add_trace(go.Scattergl(x=rows["jf_time"], y=rows["jf_residual"], mode="markers", name="Residual", showlegend=False,
                                 marker=dict(color=np.where(rows["jf_is_outlier"], RED, np.where(ambiguous, AMBER, np.where(evaluated, TEAL, MUTED))),
                                             symbol=np.where(rows["jf_is_outlier"], "x", "circle"), size=np.where(rows["jf_is_outlier"], 8, 4.5), opacity=.85),
                                 customdata=hover, hovertemplate=hover_text.replace("Spread", "Residual")), row=2, col=1)
    figure.add_hline(y=0, line_color=MUTED, line_width=1, line_dash="dot", row=2, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=raw_cutoff, mode="lines", showlegend=False,
                               line=dict(color="#8fb5be", width=1, dash="dot"), hoverinfo="skip"), row=2, col=1)
    figure.add_trace(go.Scatter(x=band_time, y=[-value if value is not None else None for value in raw_cutoff], mode="lines", showlegend=False,
                               line=dict(color="#8fb5be", width=1, dash="dot"), hoverinfo="skip"), row=2, col=1)
    figure.update_yaxes(title_text=f"Spread · {unit}", row=1, col=1)
    figure.update_yaxes(title_text=f"Residual · {unit}", row=2, col=1)
    figure.update_xaxes(title_text="Time · UTC", row=2, col=1)
    # PLOTTING LOGIC: The surrounding dashboard supplies context; the chart title stays compact.
    figure.update_layout(title=dict(text=title or "Selected bond", font=dict(size=15), x=.015))
    return _style(figure, height=600, trade=True)


def diagnostic_figure(annotated, *, spread_col="spread", unit="bp"):
    """Contrast unflagged/flagged diagnostics without implying fit eligibility."""
    # PLOTTING LOGIC: Histogram counts use all finite residuals, without trimming tails.
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Residual distribution", "Evaluation coverage"),
                           column_widths=[.57, .43], horizontal_spacing=.22)
    for flag, label, color in [(False, "Unflagged diagnostic", TEAL), (True, "Flagged outlier", RED)]:
        rows = annotated.loc[annotated["jf_is_outlier"].eq(flag)]
        figure.add_trace(go.Histogram(x=rows["jf_residual"], name=label, marker_color=color,
                                     opacity=.75, nbinsx=45,
                                     hovertemplate="Residual %{x:.4f} " + unit + "<br>Trades %{y}<extra>%{fullData.name}</extra>"), row=1, col=1)
    # Input: jf_status=['ok','outlier','ok','insufficient_history'].
    # Output: counts={'ok':2,'outlier':1,'insufficient_history':1}.
    # Trick: Missing diagnostic fields do not become zero residuals; all statuses retain explicit counts.
    # CORE LOGIC: STEP 1 — Count evaluation statuses before displaying their coverage.
    counts = annotated["jf_status"].value_counts(dropna=False)
    # PLOTTING LOGIC: Draw the already calculated coverage table with stable status colors.
    figure.add_trace(go.Bar(x=counts.values, y=[str(name).replace("_", " ") for name in counts.index], orientation="h", showlegend=False,
                           text=counts.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Trades %{x}<extra></extra>",
                           marker_color=[RED if name == "outlier" else TEAL if name == "ok" else AMBER for name in counts.index]), row=1, col=2)
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text=f"Residual · {unit}", row=1, col=1)
    figure.update_yaxes(title_text="Trades", row=1, col=1)
    figure.update_xaxes(title_text="Trades", row=1, col=2)
    figure.update_yaxes(autorange="reversed", row=1, col=2)
    return _style(figure, height=380)


def comparison_figure(table):
    """Display method flag counts without implying that more flags are better."""
    # PLOTTING LOGIC: Common methods use a stable display order and exact counts in hover text.
    # PLOTTING LOGIC: Older supplied comparison tables can omit total while retaining their exact flag counts.
    hover = np.column_stack([table["flag_rate"], table["evaluated"], table.get("total", pd.Series(np.nan, index=table.index))])
    figure = go.Figure(go.Bar(y=[METHOD_LABELS.get(name, name) for name in table["method"]],
                             x=table["outliers"], orientation="h", marker_color=TEAL,
                             text=table["outliers"], textposition="outside", cliponaxis=False,
                             customdata=hover,
                             hovertemplate="<b>%{y}</b><br>Flagged trades %{x}<br>Flag rate / evaluated %{customdata[0]:.1%}"
                                           "<br>Evaluation coverage %{customdata[1]} / %{customdata[2]} supplied trades<extra></extra>"))
    figure.update_xaxes(title_text="Flagged trades", rangemode="tozero")
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(title=dict(text="Method sensitivity on the selected bond", font=dict(size=15), x=.02), showlegend=False, bargap=.42)
    _style(figure, height=430)
    figure.update_layout(margin=dict(l=240, r=48, t=64, b=48))
    return figure
