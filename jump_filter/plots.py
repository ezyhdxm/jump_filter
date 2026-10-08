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


def _drawing_breaks(rows, max_gap):
    # Input: times=['2026-10-01T14:00Z','2026-10-01T14:01Z','2026-10-02T14:00Z'], boundaries=[False,False,True],max_gap='1D'.
    # Output: drawing breaks=[False,False,True].
    # Trick: A session boundary breaks the path even when its wall-clock gap is shorter than max_gap; the first row never starts with a separator.
    # CORE LOGIC: STEP 1 — Find actual path breaks before reducing the drawing population.
    stamps = rows["jf_time"].dt.as_unit("ns").array.asi8
    boundaries = rows.get("jf_session_boundary", pd.Series(False, index=rows.index)).fillna(False).to_numpy(dtype=bool)
    breaks = boundaries.copy()
    breaks[1:] |= np.diff(stamps) > pd.Timedelta(max_gap).value
    if len(breaks):
        breaks[0] = False
    return breaks


def _sampled_drawing_breaks(rows, positions, max_gap):
    # Input: session breaks=[False,False,False,True,False,False],selected positions=[0,1,4,5].
    # Output: closure prefix=[0,0,0,1,1,1],displayed breaks=[False,False,True,False].
    # Trick: Prefix counts retain every original closure even when its boundary trade is omitted from the drawing population.
    # CORE LOGIC: STEP 1 — Transfer actual path boundaries to displayed source positions.
    closures = np.cumsum(_drawing_breaks(rows, max_gap))
    breaks = np.r_[False, np.diff(closures[positions]) > 0] if len(positions) else np.array([], dtype=bool)
    # Input: supported=[True,True,False,True,True,True],selected positions=[0,1,4,5],existing breaks=[False,False,False,False].
    # Output: missing prefix=[0,0,1,1,1,1],displayed breaks=[False,False,True,False].
    # Trick: A retained unsupported endpoint already becomes NaN; add a separator only when a hidden unsupported record lies between two supported displayed endpoints.
    # CORE LOGIC: STEP 2 — Preserve omitted reference holes without duplicating visible NaN boundaries.
    supported = np.isfinite(rows["jf_baseline"].to_numpy(dtype=float)) & np.isfinite(rows["jf_threshold"].to_numpy(dtype=float))
    missing = np.cumsum(~supported)
    breaks[1:] |= supported[positions[1:]] & supported[positions[:-1]] & (np.diff(missing[positions]) > 0)
    return breaks


def _band_coordinates(rows, max_gap, *, session_breaks=None):
    # Input: rows times=['2026-10-01T14:00Z','2026-10-02T14:00Z'],baseline=[100,101],threshold=[1,2],session_breaks=[False,True].
    # Output: times=[Oct1 14:00UTC,None,Oct2 14:00UTC], baseline=[100,NaN,101], upper=[101,NaN,103], lower=[99,NaN,99], cutoff=[1,NaN,2].
    # Trick: Drawing separators are not trade records. Numeric NaNs serialize as null; source-row identifiers never change.
    # CORE LOGIC: STEP 1 — Allocate one separator at each real closure or inactive gap.
    breaks = _drawing_breaks(rows, max_gap) if session_breaks is None else session_breaks
    positions = np.arange(len(rows)) + np.cumsum(breaks)
    size = len(rows) + int(np.sum(breaks))
    timestamps = np.full(size, None, dtype=object)
    timestamps[positions] = rows["jf_time"].to_numpy(dtype=object)
    baseline, cutoff = rows["jf_baseline"].to_numpy(dtype=float), rows["jf_threshold"].to_numpy(dtype=float)
    # Input: baseline=[100,101,NaN],cutoff=[1,NaN,1].
    # Output: supported=[True,False,False],drawn baseline=[100,NaN,NaN],drawn cutoff=[1,NaN,NaN].
    # Trick: This changes only chart coordinates; an incomplete reference must not create a baseline segment or a residual cutoff that contradicts its missing band.
    # CORE LOGIC: STEP 2 — Align all rendered reference paths to their actual diagnostic support.
    supported = np.isfinite(baseline) & np.isfinite(cutoff)
    baseline, cutoff = np.where(supported, baseline, np.nan), np.where(supported, cutoff, np.nan)
    # PLOTTING LOGIC: Vectorized assignment avoids constructing a pandas Series for every displayed trade.
    coordinates = [timestamps]
    for values in (baseline, baseline + cutoff, baseline - cutoff, cutoff):
        drawn = np.full(size, np.nan)
        drawn[positions] = values
        coordinates.append(drawn)
    return coordinates


def _band_polygons(timestamps, upper, lower):
    # Input: times=[Oct1 14:00UTC,Oct1 14:01UTC,None,Oct2 14:00UTC],upper=[101,102,NaN,103],lower=[99,100,NaN,101].
    # Output: supported=[True,True,False,True],starts=[0,3],stops=[2,4].
    # Trick: Finite band edges establish support; separators and unavailable references never acquire a synthetic zero-height band.
    # CORE LOGIC: STEP 1 — Identify each contiguous supported run independently.
    supported = np.isfinite(upper) & np.isfinite(lower)
    starts = np.flatnonzero(supported & ~np.r_[False, supported[:-1]])
    stops = np.flatnonzero(supported & ~np.r_[supported[1:], False]) + 1
    # Input: times=[Oct1 14:00UTC,Oct1 14:01UTC],upper=[101,102],lower=[99,100],starts=[0],stops=[2].
    # Output: x=[Oct1 14:00UTC,Oct1 14:01UTC,Oct1 14:01UTC,Oct1 14:00UTC,Oct1 14:00UTC,None],y=[101,102,100,99,101,None].
    # Trick: Upper edges run forward and lower edges backward to enclose only that run; explicit closure plus null separators makes Plotly fill='toself' respect every gap. A singleton encloses zero area.
    # CORE LOGIC: STEP 2 — Close separate band polygons without joining their endpoints.
    polygon_time, polygon_values = [], []
    for start, stop in zip(starts, stops):
        section = timestamps[start:stop].tolist()
        polygon_time.extend(section + section[::-1] + [section[0], None])
        polygon_values.extend(upper[start:stop].tolist() + lower[start:stop][::-1].tolist() + [float(upper[start]), None])
    return polygon_time, polygon_values


def _policy_reference_rows(rows, config):
    # Input: support trend=[1.00,1.01,1.02],reliable=[True,False,True],max_deviation_bps=10,spread_units_per_bp=.01.
    # Output: supported=[True,False,True],raw-unit limit=.1.
    # Trick: Spread labels do not convert values; the explicit units-per-bp configuration converts the policy limit, and unreliable trends never acquire a visual limit.
    # CORE LOGIC: STEP 1 — Select reliable policy references and convert the optional distance limit.
    trend = rows["jf_support_trend"].to_numpy(dtype=float)
    reliable = rows.get("jf_support_reliable", pd.Series(False, index=rows.index)).fillna(False).to_numpy(dtype=bool)
    supported = reliable & np.isfinite(trend)
    raw_limit = float(config.get("max_deviation_bps", 10.)) * float(config.get("spread_units_per_bp", 1.))
    # Input: times=[Oct1 14:00UTC,Oct1 14:01UTC,Oct1 14:02UTC],trend=[1.00,1.01,1.02],supported=[True,False,True],raw limit=.1,boundaries=[False,False,False].
    # Output: drawing baseline=[1.00,NaN,1.02],drawing threshold=[.1,NaN,.1],times/boundaries unchanged.
    # Trick: Construct only four drawing columns; full source annotations remain unchanged and source positional ordering survives duplicate index labels.
    # CORE LOGIC: STEP 2 — Map policy support to the shared gap-preserving coordinate interface.
    return pd.DataFrame({
        "jf_time": rows["jf_time"].array,
        "jf_baseline": np.where(supported, trend, np.nan),
        "jf_threshold": np.where(supported, raw_limit, np.nan),
        "jf_session_boundary": rows.get("jf_session_boundary", pd.Series(False, index=rows.index)).to_numpy(),
    }, index=rows.index)


def _extrema_positions(values, limit):
    # Input: values=[0,9,1,8,2,7],limit=4 -> Output: positions=[0,1,2,5].
    # Trick: The first/last records and bucket minima/maxima retain spikes and corners; ties select the earliest original position. This is display sampling, never fitting data.
    # CORE LOGIC: STEP 1 — Budget endpoints and deterministic interior extrema.
    count = len(values)
    if count <= limit:
        return np.arange(count)
    if limit <= 3:
        return np.array([0, count - 1])[:limit]
    buckets = max(1, (limit - 2) // 2)
    edges = np.linspace(1, count - 1, buckets + 1, dtype=int)
    selected = [0, count - 1]
    # Input: interior bucket values=[9,1,8,2] at positions1..4 -> Output: selected=[0,5,2,1], sorted unique=[0,1,2,5].
    # Trick: Every bucket is nonempty because downsampling is only entered when the count exceeds the budget.
    # CORE LOGIC: STEP 2 — Select actual trade positions without inventing aggregated spreads.
    for left, right in zip(edges[:-1], edges[1:]):
        bucket = values[left:right]
        selected.extend([left + int(np.argmin(bucket)), left + int(np.argmax(bucket))])
    return np.unique(selected)


def _display_positions(rows, values, max_points):
    # CONFIGURATION LOGIC: None explicitly requests the complete interactive chart; bounded charts require a useful positive budget.
    if max_points is not None and (isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 4):
        raise ValueError("max_points must be None or an integer of at least 4")
    # Input: flags=[False,False,True,False],statuses=['ok','ok','outlier','ok'],regime=[False,False,False,False],max_points=4 -> Output: positions=[0,1,2,3].
    # Trick: Small bonds keep every record. Review markers use original positional identity, independently of duplicate index labels.
    # CORE LOGIC: STEP 1 — Identify real review markers before budgeting the drawing population.
    if max_points is None or len(rows) <= max_points:
        return np.arange(len(rows))
    review = rows["jf_is_outlier"].to_numpy(dtype=bool) | rows["jf_regime_change"].fillna(False).to_numpy(dtype=bool)
    review |= rows["jf_status"].isin(["ambiguous_transition", "provisional_jump"]).to_numpy()
    priority = np.flatnonzero(review)
    # Input: values=[0,9,20,8,2,7],priority=[1,2,3],max_points=4 -> Output: required=[0,1,2,3,5],sampled=[1,3],positions=[0,1,3,5].
    # Trick: Always reserve the full period's endpoints. If their union with markers exceeds the budget, markers are sampled and displayed/total counts disclose omissions.
    # CORE LOGIC: STEP 2 — Preserve time coverage even when interior review markers consume the budget.
    endpoints = np.array([0, len(rows) - 1])
    required = np.union1d(priority, endpoints)
    if len(required) > max_points:
        interior = priority[(priority > 0) & (priority < len(rows) - 1)]
        sampled = interior[_extrema_positions(values[interior], max_points - 2)]
        return np.union1d(endpoints, sampled)
    # Input: values=[0,9,20,8,2,7],required=[0,2,5],max_points=4 -> Output: background=[1,3,4],retained=[1],positions=[0,1,2,5].
    # Trick: Min/max pairing can leave one budget slot unused; all review markers are preserved when their union with endpoints fits.
    # CORE LOGIC: STEP 3 — Fill the remaining budget with background shape evidence.
    background = np.setdiff1d(np.arange(len(rows)), required, assume_unique=True)
    retained = background[_extrema_positions(values[background], max_points - len(required))]
    return np.sort(np.concatenate((required, retained)))


def trade_figure(annotated, *, cusip_col="CUSIP", spread_col="spread", unit="bp", title=None, max_gap="1D", max_points=20_000):
    """Show trade evidence within a display budget; full annotations remain intact."""
    # PLOTTING LOGIC: Sort only the display copy; exported rows retain their original order and index.
    rows = annotated.loc[annotated["jf_time"].notna()].sort_values(["jf_time", "jf_row_id"], kind="stable")
    value = pd.to_numeric(rows[spread_col], errors="coerce")
    finite = np.isfinite(value.to_numpy(dtype=float, na_value=np.nan))
    rows, value = rows.loc[finite], value.loc[finite]
    # PLOTTING LOGIC: The algorithm and statistics always use the full source; only the drawing population is reduced.
    total, total_flags = len(rows), int(rows["jf_is_outlier"].sum())
    positions = _display_positions(rows, value.to_numpy(dtype=float), max_points)
    displayed_breaks = _sampled_drawing_breaks(rows, positions, max_gap)
    # PLOTTING LOGIC: Policy references have their own reliable-support mask, so their omitted holes must be preserved before display sampling as well.
    policy_config = annotated.attrs.get("jump_filter", {}).get("config", {})
    policy_coordinates = None
    if policy_config.get("max_deviation_rule") and "jf_support_trend" in rows:
        policy_rows = _policy_reference_rows(rows, policy_config)
        policy_breaks = _sampled_drawing_breaks(policy_rows, positions, max_gap)
        policy_coordinates = _band_coordinates(policy_rows.iloc[positions], max_gap, session_breaks=policy_breaks)
    rows, value = rows.iloc[positions], value.iloc[positions]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.14,
                           row_heights=[.7, .3], subplot_titles=("Trades, baseline & statistical cutoff band", "Deviation from baseline"))
    figure.update_layout(meta=dict(total_timed_trades=total, displayed_trades=len(rows),
                                   total_outliers=total_flags, displayed_outliers=int(rows["jf_is_outlier"].sum()),
                                   sampled=len(rows) < total, max_points=max_points))
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
    # PLOTTING LOGIC: Policy evidence appears only when enabled; raw method diagnostics retain their own cutoff band and baseline.
    if (policy_config.get("quantity_rule") or policy_config.get("max_deviation_rule")) and "jf_support_distance_bps" in rows:
        hover = np.column_stack((hover,
            rows.get("jf_quantity_notional", pd.Series(np.nan, index=rows.index)),
            rows.get("jf_support_trend", pd.Series(np.nan, index=rows.index)),
            rows["jf_support_distance_bps"],
            rows.get("jf_support_reliable", pd.Series(False, index=rows.index)).astype(str),
            rows.get("jf_algorithm_status", rows["jf_status"]),
            rows.get("jf_support_reason", pd.Series("not_recorded", index=rows.index)),
            rows.get("jf_policy_cap_status", pd.Series("not_recorded", index=rows.index)),
            rows.get("jf_policy_cap_reason", pd.Series("not_recorded", index=rows.index)),
        ))
        hover_text = hover_text.replace("<extra></extra>",
            "<br>Quantity (notional) %{customdata[11]:,.0f}<br>Policy support trend %{customdata[12]:.4f} " + unit +
            "<br>Distance to support %{customdata[13]:.3f} bp · Reliable support %{customdata[14]}" +
            "<br>Original algorithm %{customdata[15]}<br>Support reason %{customdata[16]}" +
            "<br>Cap assessment %{customdata[17]}<br>Cap reason %{customdata[18]}<extra></extra>")
    # PLOTTING LOGIC: A retained trade without a reliable cap check must not look like a verified within-limit trade.
    # Trick: Older annotations can lack the explicit cap audit; reliable support supplies a display-only fallback without changing any fitting decision.
    evaluated = rows["jf_status"].isin(["ok", "outlier", "provisional_jump"])
    ambiguous = rows["jf_status"].eq("ambiguous_transition")
    retained = rows["jf_status"].eq("retained_uncertain")
    unverified = rows["jf_status"].eq("unverified_support")
    cap_enabled = bool(policy_config.get("max_deviation_rule", False))
    cap_assessed = rows.get("jf_policy_cap_assessed", rows.get("jf_support_reliable", pd.Series(False, index=rows.index))).fillna(False)
    retained_unchecked = retained & cap_enabled & ~cap_assessed
    classes = [
        ("Evaluated", ~rows["jf_is_outlier"] & evaluated, TEAL, "circle", 5.5),
        ("Unscored", ~rows["jf_is_outlier"] & ~evaluated & ~ambiguous & ~retained & ~unverified, MUTED, "circle-open", 7),
        ("Retained · within cap" if cap_enabled else "Retained · uncertain", retained & ~retained_unchecked & ~rows["jf_is_outlier"], AMBER, "circle-open", 7),
        ("Retained · cap unchecked", retained_unchecked & ~rows["jf_is_outlier"], AMBER, "square-open", 8),
        ("Unverified · excluded", unverified & ~rows["jf_is_outlier"], MUTED, "square-open", 8),
        ("Ambiguous turn", ambiguous, AMBER, "diamond-open", 9),
        ("Outlier", rows["jf_is_outlier"], RED, "x", 10),
    ]
    for label, mask, color, symbol, size in classes:
        # PLOTTING LOGIC: Outliers remain visible at their original spread values.
        figure.add_trace(go.Scattergl(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                     marker=dict(color=color, size=size, symbol=symbol, opacity=.85), showlegend=bool(mask.any()),
                                     customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: Separate closed polygons preserve actual closures and unsupported-reference holes; tonexty can bridge interrupted traces.
    band_time, baseline, upper, lower, raw_cutoff = _band_coordinates(rows, max_gap, session_breaks=displayed_breaks)
    polygon_time, polygon_values = _band_polygons(band_time, upper, lower)
    figure.add_trace(go.Scatter(x=polygon_time, y=polygon_values, mode="lines", line=dict(width=0),
                               fill="toself", fillcolor="rgba(8,127,140,.075)", name="Cutoff band",
                               hoverinfo="skip", connectgaps=False), row=1, col=1)
    # PLOTTING LOGIC: Baselines are diagnostics rather than quoted executable mid prices.
    figure.add_trace(go.Scatter(x=band_time, y=baseline, mode="lines", name="Baseline",
                               line=dict(color=INK, width=1.8), connectgaps=False,
                               hovertemplate="%{x}<br>Baseline %{y:.4f} " + unit + "<extra></extra>"), row=1, col=1)
    # PLOTTING LOGIC: The optional hard distance limit is distinct from the method's statistical band; unreliable support remains visibly interrupted.
    if policy_coordinates is not None:
        support_time, support_trend, support_upper, support_lower, _ = policy_coordinates
        if np.isfinite(support_trend).any():
            duplicate_baseline = list(support_time) == list(band_time) and np.array_equal(support_trend, baseline, equal_nan=True)
            if not duplicate_baseline:
                figure.add_trace(go.Scatter(x=support_time, y=support_trend, mode="lines", name="Support trend",
                    line=dict(color=TEAL, width=1.5, dash="dash"), connectgaps=False,
                    hovertemplate="%{x}<br>Policy support trend %{y:.4f} " + unit + "<extra></extra>"), row=1, col=1)
            limit_label = f"{float(policy_config.get('max_deviation_bps', 10.)):g} bp limit"
            for boundary, showlegend in [(support_upper, True), (support_lower, False)]:
                figure.add_trace(go.Scatter(x=support_time, y=boundary, mode="lines", name=limit_label,
                    legendgroup="policy-deviation-limit", showlegend=showlegend,
                    line=dict(color=AMBER, width=1.1, dash="dash"), connectgaps=False,
                    hovertemplate="%{x}<br>Policy support limit %{y:.4f} " + unit + "<extra></extra>"), row=1, col=1)
    regime = rows["jf_regime_change"].fillna(False)
    provisional = rows["jf_status"].eq("provisional_jump")
    for label, mask, symbol in [("Level change", regime & ~ambiguous, "diamond-open"),
                                ("Provisional", provisional, "triangle-up-open")]:
        figure.add_trace(go.Scatter(x=rows.loc[mask, "jf_time"], y=value.loc[mask], mode="markers", name=label,
                                   marker=dict(color=AMBER, symbol=symbol, size=11, line=dict(width=1.5)), showlegend=bool(mask.any()),
                                   customdata=hover[mask.to_numpy()], hovertemplate=hover_text), row=1, col=1)
    # PLOTTING LOGIC: The residual plot uses the engine's raw-unit diagnostic without recomputing it.
    figure.add_trace(go.Scattergl(x=rows["jf_time"], y=rows["jf_residual"], mode="markers", name="Residual", showlegend=False,
                                 marker=dict(color=np.where(rows["jf_is_outlier"], RED, np.where(ambiguous | retained, AMBER, np.where(evaluated, TEAL, MUTED))),
                                             symbol=np.where(rows["jf_is_outlier"], "x", np.where(unverified | retained_unchecked, "square-open", np.where(retained, "circle-open", "circle"))), size=np.where(rows["jf_is_outlier"], 8, 4.5), opacity=.85),
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
    edges, unflagged, flagged = _residual_histograms(annotated["jf_residual"].to_numpy(dtype=float),
                                                  annotated["jf_is_outlier"].to_numpy(dtype=bool))
    # PLOTTING LOGIC: Send a fixed-size count table, rather than every residual, to the notebook/browser.
    centres, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    for counts, label, color in [(unflagged, "Unflagged diagnostic", TEAL), (flagged, "Flagged outlier", RED)]:
        figure.add_trace(go.Bar(x=centres, y=counts, width=widths, name=label, marker_color=color,
                               opacity=.75, customdata=np.column_stack((edges[:-1], edges[1:])),
                               hovertemplate="Residual [%{customdata[0]:.4f}, %{customdata[1]:.4f}] " + unit +
                                             "<br>Trades %{y}<extra>%{fullData.name}</extra>"), row=1, col=1)
    # Input: jf_status=['ok','outlier','ok','insufficient_history'].
    # Output: counts={'ok':2,'outlier':1,'insufficient_history':1}.
    # Trick: Missing diagnostic fields do not become zero residuals; all statuses retain explicit counts.
    # CORE LOGIC: STEP 1 — Count evaluation statuses before displaying their coverage.
    counts = annotated["jf_status"].value_counts(dropna=False)
    # PLOTTING LOGIC: Draw the already calculated coverage table with stable status colors.
    figure.add_trace(go.Bar(x=counts.values, y=[str(name).replace("_", " ") for name in counts.index], orientation="h", showlegend=False,
                           text=counts.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Trades %{x}<extra></extra>",
                           marker_color=[RED if name in ("outlier", "policy_outlier") else TEAL if name == "ok" else AMBER for name in counts.index]), row=1, col=2)
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text=f"Residual · {unit}", row=1, col=1)
    figure.update_yaxes(title_text="Trades", row=1, col=1)
    figure.update_xaxes(title_text="Trades", row=1, col=2)
    figure.update_yaxes(autorange="reversed", row=1, col=2)
    return _style(figure, height=380)


def _residual_histograms(values, flags, *, bins=45):
    # Input: values=[0,0,1,NaN],flags=[False,True,False,False],bins=2 -> Output: edges=[0,.5,1],finite mask=[True,True,True,False].
    # Trick: Shared full-population edges make the two distributions comparable; missing residuals are not converted to zero.
    # CORE LOGIC: STEP 1 — Establish a common finite-residual population and binning.
    finite = np.isfinite(values)
    edges = np.histogram_bin_edges(values[finite], bins=bins) if finite.any() else np.linspace(-.5, .5, bins + 1)
    # Input: values=[0,0,1,NaN],flags=[False,True,False,False],edges=[0,.5,1] -> Output: unflagged=[1,1],flagged=[1,0].
    # Trick: NumPy bins are [left,right), with an inclusive final right boundary; counts include every finite residual, including tails.
    # CORE LOGIC: STEP 2 — Aggregate full-data diagnostic counts without sampling.
    unflagged = np.histogram(values[finite & ~flags], bins=edges)[0]
    flagged = np.histogram(values[finite & flags], bins=edges)[0]
    return edges, unflagged, flagged


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
