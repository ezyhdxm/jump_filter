"""Optional notebook workbench and reusable dashboard statistics."""
# SETUP LOGIC: The notebook dependency is imported only when constructing a workbench.
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .explanations import METHOD_HELP, METHOD_EXPLANATIONS, COMMON_STEPS, CLOCK_STEPS, FITTING_STEPS, PARAMETER_HELP, math_display_blocks

# UI LOGIC: CSS is scoped to this workbench and never changes another notebook's controls.
# Trick: Native widget inputs use a horizontal 148px flex basis; column labels require an explicit 38px vertical basis.
STYLE = """
.jf-workbench{--jf-ink:#142D3D;--jf-teal:#087F8C;--jf-muted:#657887;--jf-line:#DDE4E8;background:#F5F6F8;color:var(--jf-ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px;border-radius:16px;border:1px solid var(--jf-line);max-width:1500px;width:100%;box-sizing:border-box}
.jf-workbench *{box-sizing:border-box}.jf-workbench .widget-vbox{gap:12px;min-width:0}.jf-workbench .widget-html-content{max-width:100%}.jf-workbench h2,.jf-workbench h3,.jf-workbench h4{color:var(--jf-ink);line-height:1.3}.jf-workbench h3{font-size:20px;margin:0 0 10px}.jf-workbench h4{font-size:14px;margin:22px 0 8px}.jf-workbench p{margin:0 0 8px}.jf-workbench .jf-hero{padding:0 0 17px;border-bottom:1px solid var(--jf-line)}.jf-workbench .jf-hero h2{margin:5px 0 6px;font-size:30px;font-weight:750;letter-spacing:-.035em}.jf-workbench .jf-hero p{margin:0;color:var(--jf-muted);font-size:14px;max-width:800px}.jf-workbench .jf-eyebrow{color:var(--jf-teal);font-size:10px;letter-spacing:.16em;font-weight:750;text-transform:uppercase}.jf-workbench .jf-hero-top{display:flex;align-items:center;justify-content:space-between;gap:12px}.jf-workbench .jf-badge{display:inline-block;padding:3px 9px;border:1px solid #C8E3E5;border-radius:20px;color:#076A74;background:#E9F5F5;font-size:11px;font-weight:650;white-space:nowrap}
.jf-workbench .jf-card{background:white;border:1px solid var(--jf-line);border-radius:10px;padding:16px;min-width:0}.jf-workbench .jf-help{color:var(--jf-muted);font-size:12px;line-height:1.55}.jf-workbench .jf-row{display:flex;flex-flow:row wrap;gap:12px;align-items:flex-end;min-width:0}.jf-workbench .jf-selection{padding:15px 17px;background:white;border:1px solid var(--jf-line);border-radius:10px}.jf-workbench .jf-control{flex:1 1 230px;min-width:0;max-width:100%;height:auto}.jf-workbench .jf-control:not(.widget-checkbox){display:flex;flex-direction:column;align-items:stretch}.jf-workbench .jf-control .widget-label{width:100%!important;text-align:left;white-space:normal;overflow:visible;height:auto;font-size:11px;font-weight:700;color:var(--jf-ink);margin-bottom:5px;letter-spacing:.015em}.jf-workbench input:not([type=checkbox]),.jf-workbench select{border:1px solid #CDD7DE;border-radius:6px;color:var(--jf-ink);background:#FFF;padding:7px 10px;min-height:37px;font-size:13px}.jf-workbench input:focus,.jf-workbench select:focus,.jf-workbench button:focus-visible{outline:2px solid var(--jf-teal);outline-offset:2px}.jf-workbench .widget-button{height:38px;border:1px solid #CDD7DE;background:white;color:var(--jf-ink);border-radius:7px;font-weight:650;padding:7px 15px;transition:background .15s}.jf-workbench .widget-button:hover{background:#EEF3F5}.jf-workbench .widget-button.mod-primary{background:var(--jf-teal);color:white;border-color:var(--jf-teal)}.jf-workbench .widget-button.mod-primary:hover{background:#076A74}.jf-workbench .widget-button:disabled{opacity:.5}.jf-workbench .jf-actionbar{align-items:center}.jf-workbench .jf-actionbar .widget-button{flex:0 1 auto}.jf-workbench .jf-actionbar .jf-state-area{flex:1 1 230px}.jf-workbench .jf-status{padding:9px 12px;border:1px solid #C9E2E4;background:#F0F8F8;border-radius:7px;color:#27636B;font-size:12px}.jf-workbench .jf-status:empty{display:none}.jf-workbench .jf-status-error{border-color:#ECC7CE;background:#FFF1F3;color:#9F3043}.jf-workbench .jf-state{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--jf-muted)}.jf-workbench .jf-state-pending .jf-badge{background:#FFF5E3;border-color:#EAD3A4;color:#865E20}.jf-workbench .jf-state-ready .jf-badge{background:#EEF2F5;border-color:#DDE4E8;color:#596E7D}
.jf-workbench .jf-control select,.jf-workbench .jf-control input:not([type=checkbox]){flex:0 0 38px;height:38px;min-height:38px;max-height:38px;width:100%;margin:0}.jf-workbench .jf-method-context{padding:0 2px}.jf-workbench .jf-method-context p{margin:6px 0 0;max-width:1000px}.jf-workbench .jf-kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px}.jf-workbench .jf-kpi{background:white;border:1px solid var(--jf-line);padding:13px 12px;border-radius:9px;font-variant-numeric:tabular-nums}.jf-workbench .jf-kpi strong{display:block;font-size:23px;line-height:1.25;font-weight:700;letter-spacing:-.025em;margin-bottom:5px;color:var(--jf-ink)}.jf-workbench .jf-kpi span{font-size:10px;line-height:1.3;color:var(--jf-muted)}.jf-workbench .jf-kpi-alert strong{color:#BA3C51}.jf-workbench .jf-kpi-rate strong{color:var(--jf-teal)}.jf-workbench .jf-table{overflow:auto;max-height:400px;border:1px solid var(--jf-line);border-radius:8px}.jf-workbench table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;font-variant-numeric:tabular-nums}.jf-workbench th,.jf-workbench td{white-space:nowrap;padding:9px 11px!important;border:0!important;border-bottom:1px solid #EAF0F3!important;text-align:right}.jf-workbench thead th{position:sticky;top:0;background:#F1F5F7;z-index:1;color:#4D6371;font-size:11px;font-weight:650}.jf-workbench tbody tr:nth-child(even){background:#FAFCFD}.jf-workbench tbody tr:hover{background:#EFF8F8}.jf-workbench td:first-child,.jf-workbench th:first-child{text-align:left}
.jf-workbench .widget-tab{width:100%;min-width:0}.jf-workbench .widget-tab>.p-TabBar,.jf-workbench .widget-tab>.lm-TabBar{overflow:auto;min-height:41px}.jf-workbench .p-TabBar-tab,.jf-workbench .lm-TabBar-tab{background:transparent;border:0;color:var(--jf-muted);padding:10px 14px;font-size:12px;font-weight:650;min-width:110px}.jf-workbench .p-TabBar-tab.p-mod-current,.jf-workbench .lm-TabBar-tab.lm-mod-current{background:white;color:var(--jf-teal);border-bottom:2px solid var(--jf-teal)}.jf-workbench .widget-tab-contents{padding:16px;border:1px solid var(--jf-line);background:white;border-radius:0 9px 9px 9px;min-width:0}.jf-workbench .widget-accordion{width:100%;min-width:0}.jf-workbench .widget-accordion .p-Collapse-header,.jf-workbench .widget-accordion .lm-Collapse-header{background:white;border:1px solid var(--jf-line);color:var(--jf-ink);padding:11px 13px;font-size:12px;font-weight:650}.jf-workbench .widget-accordion .p-Collapse-contents,.jf-workbench .widget-accordion .lm-Collapse-contents{background:white;border:1px solid var(--jf-line);padding:15px}.jf-workbench .jf-param{flex:1 1 275px;min-width:0;border:1px solid var(--jf-line);border-radius:8px;padding:12px;gap:6px;background:#FAFCFD}.jf-workbench .jf-param-title{font-size:11px;font-weight:700;color:var(--jf-ink)}.jf-workbench .jf-param .widget-hslider{width:100%;min-width:0}.jf-workbench .jf-param .widget-text{width:110px;flex:0 0 110px}.jf-workbench .jf-param-values{display:flex;flex-direction:row;align-items:center;gap:12px;width:100%;min-width:0}.jf-workbench .jf-param-values .widget-hslider{flex:1 1 auto;min-width:80px}.jf-workbench .jf-param .jf-help{font-size:11px;margin:0}.jf-workbench .jf-note{font-size:12px;color:var(--jf-muted)}.jf-workbench .jf-note summary{cursor:pointer;color:var(--jf-muted);font-weight:600}.jf-workbench .jf-note p{margin:9px 0 0;line-height:1.65}.jf-workbench .jf-export{border-top:1px solid var(--jf-line);padding-top:15px}.jf-workbench .jf-export .jf-control{flex:1 1 300px}.jf-workbench .jf-math-section{padding:0 3px}.jf-workbench .jf-math-section h4:first-child{margin-top:5px}.jf-workbench .jf-math-section p{max-width:980px;line-height:1.7}.jf-workbench .jf-example{padding:14px;background:#F0F8F8;border-left:3px solid var(--jf-teal);border-radius:5px;margin:16px 0}.jf-workbench .jf-example h4{margin:0 0 6px}.jf-workbench .jf-section-heading{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:4px 0 12px}.jf-workbench .jf-section-heading h4{margin:0}.jf-workbench .jf-empty{padding:48px 18px;text-align:center;color:var(--jf-muted)}.jf-workbench .jf-empty strong{display:block;color:var(--jf-ink);font-size:16px;margin-bottom:6px}
.jf-workbench .widget-output,.jf-workbench .jp-OutputArea,.jf-workbench .jp-OutputArea-child,.jf-workbench .jp-OutputArea-output,.jf-workbench .output_subarea,.jf-workbench .widget-subarea,.jf-workbench .jf-figure{width:100%!important;max-width:100%;min-width:0}.jf-workbench .jf-figure{display:block!important}.jf-workbench .jp-RenderedMath,.jf-workbench mjx-container[display="true"]{max-width:100%;overflow-x:auto}.jf-workbench .MathJax_Display{max-width:100%;overflow-x:auto;text-align:left!important}
@media(max-width:760px){.jf-workbench{padding:14px}.jf-workbench .jf-kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.jf-workbench .jf-hero h2{font-size:26px}.jf-workbench .jf-hero-top{align-items:flex-start}.jf-workbench .jf-hero-top .jf-badge{font-size:10px}.jf-workbench .widget-tab-contents{padding:10px}.jf-workbench .jf-param{flex-basis:100%}}
.jf-workbench .jupyter-widget-Collapse-header{background:white;border:1px solid var(--jf-line);border-radius:7px;color:var(--jf-ink);padding:11px 13px;font-size:12px;font-weight:650}.jf-workbench .jupyter-widget-Collapse-contents{background:white;border:1px solid var(--jf-line);border-top:0;padding:15px;border-radius:0 0 7px 7px}.jf-workbench .widget-tab .p-TabBar-tab,.jf-workbench .widget-tab .lm-TabBar-tab{flex:0 0 auto!important;min-width:0!important;width:auto!important;max-width:none!important;padding:10px 14px!important;border:0!important;background:transparent!important;color:var(--jf-muted)!important;font-size:12px!important;font-weight:650!important}.jf-workbench .widget-tab .p-TabBar-tab.p-mod-current,.jf-workbench .widget-tab .lm-TabBar-tab.lm-mod-current{background:white!important;color:var(--jf-teal)!important;border-bottom:2px solid var(--jf-teal)!important}.jf-workbench .widget-tab .p-TabBar-tabLabel,.jf-workbench .widget-tab .lm-TabBar-tabLabel{white-space:nowrap!important;overflow:visible!important;text-overflow:clip!important}
"""

# PLOTTING LOGIC: Decorate the bundled Plotly module using anywidget's supported factory and lifecycle hooks.
# Trick: Native outputs may render while detached or hidden; observe their real width after attachment and tab reveals.
# Plotly's responsive handler recalculates axes and SVG geometry; no chart is visually stretched by CSS.
_RESPONSIVE_WIDGET_ESM = """
// PLOTTING LOGIC: Load the existing locally bundled module; every original model and event callback remains intact.
export default async () => {
  const moduleUrl = URL.createObjectURL(new Blob([__PLOTLY_MODULE__], {type: "text/javascript"}));
  let bundled;
  try { bundled = await import(moduleUrl); }
  finally { URL.revokeObjectURL(moduleUrl); }
  const original = typeof bundled.default === "function" ? await bundled.default() : bundled.default;
  return {
    initialize(context) { return original.initialize?.(context); },
    render(context) {
      const cleanup = original.render(context);
      const {el} = context;
      let ready = false, lastWidth = -1, frame = null;
      // PLOTTING LOGIC: Coalesce visible host-width changes into Plotly's normal responsive resize event.
      const resize = () => {
        if (!ready || !el.isConnected || el.clientWidth <= 0 || el.clientWidth === lastWidth) return;
        lastWidth = el.clientWidth;
        if (frame !== null) cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => { frame = null; window.dispatchEvent(new Event("resize")); });
      };
      const observer = new ResizeObserver(resize);
      observer.observe(el);
      // PLOTTING LOGIC: The bundled renderer emits this event after newPlot finishes, including for hidden views.
      const rendered = (event) => {
        if (event.detail?.element !== el) return;
        ready = true;
        resize();
      };
      document.addEventListener("plotlywidget-after-render", rendered);
      // PLOTTING LOGIC: Dispose every observer/listener/frame with the original widget view.
      return () => {
        observer.disconnect();
        document.removeEventListener("plotlywidget-after-render", rendered);
        if (frame !== null) cancelAnimationFrame(frame);
        cleanup?.();
      };
    },
  };
};
"""


def evaluation_summary(annotated):
    """Return explicit review counts for one applied population."""
    # Input: jf_status=['ok','outlier','invalid_input','insufficient_history','provisional_jump']; flags=[False,True,False,False,True],regime=[False,False,False,False,False].
    # Output: total=5,evaluated=3,outliers=2,flag_rate=2/3,invalid=1,unsupported=1,provisional=1,regime_changes=0.
    # Trick: Provisional jumps are evaluable review cases; invalid/unsupported rows never count as accepted evidence.
    # CORE LOGIC: STEP 1 — Build separate evaluation and review denominators.
    status = annotated["jf_status"]
    evaluated = status.isin(["ok", "outlier", "provisional_jump"])
    count = int(evaluated.sum())
    outliers = int(annotated["jf_is_outlier"].sum())
    return dict(total=len(annotated), evaluated=count, outliers=outliers,
                flag_rate=outliers / count if count else np.nan,
                invalid=int(status.eq("invalid_input").sum()),
                unsupported=int(status.eq("insufficient_history").sum()),
                provisional=int(status.eq("provisional_jump").sum()),
                regime_changes=int(annotated["jf_regime_change"].sum()))


def fitting_statistics(annotated, *, cusip_col="CUSIP"):
    """Report hard/soft fitting coverage without treating abstentions as clean."""
    # Input: CUSIP=['A','A','A','A'],jf_row_id=[0,1,2,3],jf_status=['ok','outlier','insufficient_history','provisional_jump'],jf_is_outlier=[False,True,False,True],jf_weight=[1,.2,0,.1].
    # Output: grouped table indexed by A has supplied=4,evaluated=3,hard_fit_rows=1,soft_fit_rows=2,soft_weight_sum=1.2,solver_failures=0,ambiguous_transitions=0.
    # Trick: Provisional is evaluated for review but excluded from default fits; a weight sum is not an effective sample size or inverse variance.
    # CORE LOGIC: STEP 1 — Separate assessed, retained, and softly weighted observations.
    status = annotated["jf_status"]
    hard = status.eq("ok") & ~annotated["jf_is_outlier"]
    soft = status.isin(["ok", "outlier"]) & annotated["jf_weight"].gt(0)
    frame = annotated.assign(_evaluated=status.isin(["ok", "outlier", "provisional_jump"]),
                             _hard=hard, _soft=soft,
                             _weight=annotated["jf_weight"].where(soft, 0.0),
                             _solver=status.eq("solver_not_converged"), _ambiguous=status.eq("ambiguous_transition"))
    table = frame.groupby(cusip_col, dropna=False, sort=False).agg(
        supplied=("jf_row_id", "size"), evaluated=("_evaluated", "sum"), hard_fit_rows=("_hard", "sum"),
        soft_fit_rows=("_soft", "sum"), soft_weight_sum=("_weight", "sum"), solver_failures=("_solver", "sum"), ambiguous_transitions=("_ambiguous", "sum"))
    # Input: table index=['A'],supplied=[4],evaluated=[3],hard_fit_rows=[1],soft_fit_rows=[2],soft_weight_sum=[1.2],solver_failures=[0],ambiguous_transitions=[0].
    # Output: one row CUSIP='A' retains the seven input statistics, adds coverage=.75,hard_retention=.25.
    # Trick: Denominators include every supplied row, including invalid and unsupported records.
    # CORE LOGIC: STEP 2 — Expose both evaluation and downstream retention coverage.
    table["coverage"] = table["evaluated"] / table["supplied"]
    table["hard_retention"] = table["hard_fit_rows"] / table["supplied"]
    return table.reset_index()


def method_comparison(data, config, *, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC", session_schedule=None):
    """Evaluate all methods on an explicitly selected input population."""
    # SETUP LOGIC: Public engine imports avoid a dashboard/engine initialization cycle.
    from . import METHODS, filter_trades
    records = []
    # Input: data={CUSIP:['A'],time:['2026-10-01T10:00Z'],spread:[100]}, config=FilterConfig().
    # Output: each supported method (e.g. hampel) produces
    # each row has total=1,evaluated=0,outliers=0,flag_rate=NaN,invalid=0,
    # unsupported=1,provisional=0,regime_changes=0.
    # Trick: Every method sees the same rows and hyperparameters; no result is chosen by its flag rate.
    # CORE LOGIC: STEP 1 — Recompute each algorithm on the same review population.
    for method in METHODS:
        result = filter_trades(data, replace(config, method=method), cusip_col=cusip_col,
                               time_col=time_col, spread_col=spread_col, timezone=timezone, session_schedule=session_schedule)
        records.append(dict(method=method, **evaluation_summary(result)))
    return pd.DataFrame(records)


def audit_tables(annotated):
    """Return daily support and exact reason counts for one review population."""
    # Input: jf_time=['2026-10-01T10:00Z','2026-10-01T11:00Z','2026-10-02T10:00Z']; statuses=['ok','outlier','invalid_input']; flags=[False,True,False].
    # Output: daily UTC rows=[('2026-10-01',trades=2,evaluated=2,flagged=1),('2026-10-02',trades=1,evaluated=0,flagged=0)].
    # Trick: Only observed UTC dates are listed; untimed records remain in the reason table without fabricated calendar bins.
    # CORE LOGIC: STEP 1 — Aggregate actual calendar support and flags on observed days.
    frame = annotated.assign(_day=annotated["jf_time"].dt.floor("D"),
                             _evaluated=annotated["jf_status"].isin(["ok", "outlier", "provisional_jump"]))
    daily = frame.loc[frame["_day"].notna()].groupby("_day", sort=True).agg(
        trades=("jf_row_id", "size"), evaluated=("_evaluated", "sum"), flagged=("jf_is_outlier", "sum"),
    ).reset_index().rename(columns={"_day": "date_utc"})
    # Input: status=['ok','outlier','invalid_input'],reason=['within_local_band','hampel_threshold_exceeded','missing_spread'],flags=[False,True,False].
    # Output: reason rows=[('invalid_input','missing_spread',trades=1,flagged=0),('ok','within_local_band',trades=1,flagged=0),('outlier','hampel_threshold_exceeded',trades=1,flagged=1)].
    # Trick: Status/reason groups sort deterministically; missing reasons stay explicit instead of being dropped.
    # CORE LOGIC: STEP 2 — Keep every explanatory status and reason in the review audit.
    reasons = annotated.groupby(["jf_status", "jf_reason"], dropna=False, sort=True).agg(
        trades=("jf_row_id", "size"), flagged=("jf_is_outlier", "sum"),
    ).reset_index()
    # REPORTING LOGIC: Exact tables are reusable by both dashboard surfaces and exports.
    return {"daily": daily, "reasons": reasons}


def _table_html(frame, *, limit=200):
    # UI LOGIC: Format a bounded presentation copy with readable headers; complete raw values remain in exports.
    # Trick: Percentage strings belong only to this copy, so downstream fitting and exports retain numeric fractions.
    shown = frame.head(limit).copy()
    labels = {"CUSIP": "CUSIP", "rows": "Trades", "total": "Trades", "supplied": "Supplied", "evaluated": "Evaluated",
              "flagged": "Flagged", "outliers": "Flagged", "accepted": "Accepted", "fit_eligible": "Fit eligible",
              "invalid": "Invalid", "insufficient": "Low support", "unsupported": "Low support", "provisional": "Provisional",
              "regime_candidates": "Regime candidates", "regime_changes": "Regime changes", "median_score": "Median score",
              "median_reference_count": "Median references", "flagged_rate": "Flag rate", "flag_rate": "Flag rate",
              "coverage": "Coverage", "hard_retention": "Hard retention", "hard_fit_rows": "Hard-fit rows",
              "soft_fit_rows": "Soft-fit rows", "soft_weight_sum": "Sum of influence weights", "solver_failures": "Solver failures",
              "solver_failed": "Solver failures", "ambiguous_transitions": "Ambiguous transitions", "jf_status": "Status",
              "jf_reason": "Reason", "jf_time": "Time · UTC", "date_utc": "Date · UTC", "jf_n_reference": "References",
              "jf_score": "Robust score", "jf_baseline": "Baseline", "jf_residual": "Residual", "jf_threshold": "Deviation cutoff",
              "jf_weight": "Influence weight", "jf_fit_eligible": "Fit eligible", "jf_row_id": "Source row",
              "jf_gap_minutes": "Active gap · min", "jf_wall_gap_minutes": "Wall gap · min", "jf_session_boundary": "Session boundary",
              "jf_scale": "Local scale", "jf_reference_density_per_hour": "References / hour", "jf_reference_span_minutes": "Reference span · min"}
    for name in ["coverage", "flagged_rate", "flag_rate", "hard_retention"]:
        if name in shown:
            shown[name] = shown[name].map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
    shown = shown.rename(columns=lambda name: labels.get(name, str(name).replace("_", " ").capitalize()))
    return (f'<p class="jf-help">Showing {len(shown):,} of {len(frame):,} rows.</p><div class="jf-table">' +
            shown.to_html(index=False, escape=True, border=0, na_rep="—", float_format=lambda x: f"{x:.5g}") + "</div>")


def _kpi_html(stats):
    # UI LOGIC: Words, exact counts and explicit denominators accompany status colors.
    rate = f'{stats["flag_rate"]:.1%}' if np.isfinite(stats["flag_rate"]) else "—"
    items = [("Supplied trades", f'{stats["total"]:,}'), ("Evaluated", f'{stats["evaluated"]:,}'),
             ("Flagged outliers", f'{stats["outliers"]:,}'), ("Flag rate / evaluated", rate),
             ("Insufficient support", f'{stats["unsupported"]:,}'), ("Invalid input", f'{stats["invalid"]:,}')]
    classes = ["", "", " jf-kpi-alert", " jf-kpi-rate", "", ""]
    return '<div class="jf-kpis">' + ''.join(f'<div class="jf-kpi{style}"><strong>{value}</strong><span>{label}</span></div>' for (label, value), style in zip(items, classes)) + '</div>'


def _notebook_figure(figure):
    # PLOTTING LOGIC: Canonical Plotly JSON maps missing hover diagnostics to null before the widget comm is opened.
    # Trick: Object-array NaNs can become illegal JSON primitives; numerical arrays may remain valid binary buffers.
    # The presentation copy alone is serialized; engine results and applied-data exports retain their original values.
    import plotly.graph_objects as go
    widget = go.FigureWidget(json.loads(figure.to_json(remove_uids=False)))
    # PLOTTING LOGIC: Unset an inherited fixed width so the native view fills its actual notebook output pane.
    # Trick: FigureWidget.layout is Plotly's chart layout, not ipywidgets.Layout; a scoped DOM class sizes its host.
    widget.layout.width = None
    widget.layout.autosize = True
    widget.add_class("jf-figure")
    widget._config = {**widget._config, "responsive": True, "displaylogo": False}
    # PLOTTING LOGIC: Embed the original installed Plotly module without network requests or a frontend extension.
    # Trick: Plotly 5 uses its legacy Lumino resize lifecycle and has no anywidget _esm trait to decorate.
    if hasattr(widget, "_esm"):
        widget._esm = _RESPONSIVE_WIDGET_ESM.replace("__PLOTLY_MODULE__", json.dumps(widget._esm), 1)
    return widget


class FilterDashboard:
    """Live Jupyter widgets with an explicit applied result and bundled Plotly charts.

    Install ``jump-filter[notebook]`` for a complete Notebook 7/JupyterLab 4 setup,
    or ``jump-filter[dashboard]`` inside an existing Jupyter kernel. No Streamlit
    process is required. A running kernel handles filtering and control callbacks.
    """

    def __init__(self, data, *, config=None, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC", unit="bp", session_schedule=None):
        # SETUP LOGIC: Copy caller data; keep optional notebook dependencies outside package imports.
        import ipywidgets as w
        from . import FilterConfig, METHODS
        from .plots import METHOD_LABELS
        self.w, self.data = w, data.copy()
        self.config = config or FilterConfig(time_basis="trading")
        self.mapping = dict(cusip_col=cusip_col, time_col=time_col, spread_col=spread_col, timezone=timezone)
        self.unit, self.result, self.applied_config = unit, None, None
        self.session_schedule = session_schedule.copy(deep=True) if session_schedule is not None else None
        self.applied_schedule = None
        self.comparison, self._applied_state, self.busy = None, None, False
        self._figure_widgets, self._comparison_widget = (), None
        self._explanation_widgets = ()
        self._statistics_widgets = ()
        # VALIDATION LOGIC: Fail early for a bad mapping, before showing controls that cannot run.
        if not all(column in data for column in [cusip_col, time_col, spread_col]):
            raise ValueError("CUSIP, time and spread mappings must name existing columns.")
        # UI LOGIC: Exact values drive selection; text labels are presentation only.
        values = list(data[cusip_col].dropna().drop_duplicates())
        self.cusip = w.Dropdown(description="Bond / CUSIP", options=[(str(value), value) for value in values])
        self.method = w.Dropdown(description="Detection method", options=[(METHOD_LABELS[name], name) for name in METHODS], value=self.config.method)
        self.help = w.HTML()
        self.pending, self.status, self.kpis = w.HTML(), w.HTML(), w.HTML()
        self.chart, self.statistics, self.method_output, self.explanation = [w.Output(layout=w.Layout(width="100%")) for _ in range(4)]
        self.apply_button = w.Button(description="Apply filter", button_style="primary", icon="check")
        self.compare_button = w.Button(description="Compare methods", icon="bar-chart", layout=w.Layout(width="170px"),
                                       tooltip="Compare all methods on the selected bond using the last applied settings.")
        self.export_button = w.Button(description="Export review", icon="download", disabled=True, tooltip="Save annotations, statistics, charts and settings from the last successful Apply.")
        self.export_path = w.Text(value="reports/jump_filter", description="Export folder")
        self.params, self._param_sliders, self._param_cards, param_cards = {}, {}, {}, []
        # UI LOGIC: Every numerical slider has a synchronized input box for exact values.
        specs = [("window", "Maximum reference timestamps", 3, 201, 2, "int"),
                 ("min_neighbors", "Minimum reference timestamps", 2, 60, 1, "int"),
                 ("threshold", "Robust score threshold", .5, 12., .1, "float"),
                 ("abs_floor", f"Minimum deviation · {unit}", .001, 50., .1, "float"),
                 ("reversion_tolerance", "Return tolerance · robust scale", .1, 6., .1, "float"),
                 ("iqr_multiplier", "Tukey IQR multiplier", .1, 10., .1, "float"),
                 ("multiscale_votes", "Required multiscale votes", 1, 3, 1, "int"),
                 ("trend_penalty", "TV level-change penalty λ", .1, 100., .1, "float"),
                 ("huber_delta", "Huber clipping δ", .1, 10., .1, "float"),
                 ("max_iter", "Solver iteration limit", 20, 5000, 20, "int"),
                 ("tolerance", "Solver convergence tolerance", .000001, .01, .000001, "float"),
                 ("alpha", "EWMA learning rate", .01, 1., .01, "float"),
                 ("persistence", "Persistence confirmation cohorts", 2, 10, 1, "int")]
        for name, label, lower, upper, step, kind in specs:
            value = getattr(self.config, name)
            slider_cls, input_cls = (w.IntSlider, w.BoundedIntText) if kind == "int" else (w.FloatSlider, w.BoundedFloatText)
            slider = slider_cls(value=value, min=min(lower, value), max=max(upper, value), step=step, readout=False,
                                continuous_update=False, description=label, tooltip=PARAMETER_HELP[name], style={"description_width": "0px"})
            number = input_cls(value=value, min=min(lower, value), max=max(upper, value), step=step,
                               description=label, tooltip="Exact value. " + PARAMETER_HELP[name], style={"description_width": "0px"})
            w.link((slider, "value"), (number, "value"))
            self.params[name] = number
            self._param_sliders[name] = slider
            values_row = w.HBox([slider, number], layout=w.Layout(width="100%")).add_class("jf-param-values")
            card = w.VBox([w.HTML(f'<div class="jf-param-title">{escape(label)}</div>'), values_row,
                           w.HTML(f'<p class="jf-help">{escape(PARAMETER_HELP[name])}</p>')]).add_class("jf-param")
            self._param_cards[name] = card
            param_cards.append(card)
        self.horizon = w.Text(value=str(self.config.horizon), description="Reference horizon (e.g. 3D)")
        self.max_gap = w.Text(value=str(self.config.max_gap), description="Session break gap (e.g. 1D)")
        self.time_basis = w.Dropdown(options=[("Cumulative trading time", "trading"), ("Wall-clock time", "wall")], value=self.config.time_basis, description="Distance clock")
        self.session_timezone = w.Text(value=self.config.session_timezone, description="Market session timezone")
        self.session_open = w.Text(value=self.config.session_open, description="Weekday session opens")
        self.session_close = w.Text(value=self.config.session_close, description="Weekday session closes")
        self.holidays = w.Text(value=", ".join(self.config.holidays), description="Closed dates · YYYY-MM-DD, ...")
        self._calendar_controls = [self.time_basis, self.session_timezone, self.session_open, self.session_close, self.holidays]
        self._controls = [self.method, self.horizon, self.max_gap, *self._calendar_controls, *self.params.values()]
        for control in [self.cusip, self.method, self.horizon, self.max_gap, self.export_path, *self._calendar_controls]:
            control.add_class("jf-control")
            control.style.description_width = "initial"
        # UI LOGIC: CUSIP selection changes the view only; parameter edits await explicit Apply.
        self.cusip.observe(self._focus_changed, names="value")
        self.method.observe(self._help_changed, names="value")
        for control in self._controls:
            control.observe(self._pending_changed, names="value")
        self.apply_button.on_click(self.run)
        self.compare_button.on_click(self.compare)
        self.export_button.on_click(self.export)
        self.tabs = w.Tab(children=[self.chart, self.statistics, self.method_output, self.explanation])
        for i, name in enumerate(["Trade review", "Statistics", "Comparison", "Method & mathematics"]):
            self.tabs.set_title(i, name)
        hero = w.HTML('<div class="jf-hero"><div class="jf-hero-top"><div class="jf-eyebrow">Bond research / trade quality</div><span class="jf-badge">Jupyter workbench</span></div><h2>Jump Filter</h2><p>Screen noisy trades. Preserve meaningful moves. Review every flag against its local evidence.</p></div>')
        note = w.HTML('<details class="jf-note"><summary>How to interpret this review</summary><p>Historical fit screening can use earlier and later trades. Offline methods are available alongside Causal EWMA as an online comparator. CUSIP, time and spread identify statistical anomalies; they cannot determine retail, distress or commission causes. Method &amp; mathematics contains every equation, numerical example and parameter effect.</p></details>')
        calendar = w.VBox([self._row(self.time_basis, self.session_timezone), self._row(self.session_open, self.session_close), self.holidays,
                           w.HTML('<p class="jf-help">Trading time compresses scheduled nights, weekends and listed holidays while retaining inactivity during open sessions. The session calendar is a configurable research assumption. In trading mode, horizon and max_gap measure cumulative open time. Charts always use actual UTC timestamps. Supply an authoritative session_schedule for exact holidays and early closes.</p>')])
        parameter_grid = w.Box(param_cards, layout=w.Layout(display="flex", flex_flow="row wrap")).add_class("jf-row")
        settings = w.Accordion(children=[w.VBox([parameter_grid, self._row(self.horizon, self.max_gap)]), calendar], selected_index=None)
        settings.set_title(0, "Tune detection · relevant parameters and exact values")
        settings.set_title(1, "Trading calendar · sessions, closures and gaps")
        self.pending.add_class("jf-state-area")
        actions = self._row(self.apply_button, self.compare_button, self.pending).add_class("jf-actionbar")
        selection = w.VBox([self._row(self.cusip, self.method), self.help]).add_class("jf-selection")
        export_row = self._row(self.export_path, self.export_button).add_class("jf-export")
        self.widget = w.VBox([w.HTML("<style>" + STYLE + "</style>"), hero,
                              selection, actions, settings, self.status, self.kpis, self.tabs, note, export_row],
                              layout=w.Layout(width="100%", max_width="1500px")).add_class("jf-workbench")
        self._empty_outputs()
        self._help_changed()
        self._pending_changed()

    def _row(self, *children):
        # UI LOGIC: Wrapping respects notebook pane width rather than browser viewport width.
        return self.w.Box(list(children), layout=self.w.Layout(display="flex", flex_flow="row wrap", width="100%")).add_class("jf-row")

    def _empty_outputs(self):
        # UI LOGIC: The first view explains the next action and gives each empty results tab a useful purpose.
        from IPython.display import HTML, display
        prompts = [(self.chart, "Ready to review", "Choose a bond and method, then Apply filter to see trades and outlier flags."),
                   (self.statistics, "Statistics follow your applied review", "Coverage, fitting retention and liquidity diagnostics will appear here."),
                   (self.method_output, "Compare detection methods", "Apply a filter, then Compare methods to inspect sensitivity on the selected bond.")]
        for output, title, message in prompts:
            with output:
                display(HTML(f'<div class="jf-empty"><strong>{escape(title)}</strong>{escape(message)}</div>'))

    def _state(self):
        # UI LOGIC: Only algorithm settings determine pending state; view focus and export path do not.
        fingerprint = sha256(self.session_schedule.to_csv(index=False).encode("utf-8")).hexdigest() if self.session_schedule is not None else None
        return tuple(control.value for control in self._controls) + (fingerprint,)

    def _configuration(self):
        # CONFIGURATION LOGIC: Validation belongs to FilterConfig and the engine rather than the widgets.
        from . import FilterConfig
        return FilterConfig(method=self.method.value, horizon=self.horizon.value, max_gap=self.max_gap.value,
                            time_basis=self.time_basis.value, session_timezone=self.session_timezone.value,
                            session_open=self.session_open.value, session_close=self.session_close.value,
                            holidays=tuple(value.strip() for value in self.holidays.value.split(",") if value.strip()),
                            **{name: control.value for name, control in self.params.items()})

    def _help_changed(self, _=None):
        # UI LOGIC: Show and enable only parameters used by the currently selected method.
        card = METHOD_EXPLANATIONS[self.method.value]
        mode = card["mode"].partition(" · ")[0]
        self.help.value = (f'<div class="jf-method-context"><span class="jf-badge">{escape(mode)}</span>'
                           '<p class="jf-help">' + escape(METHOD_HELP[self.method.value]) + '</p></div>')
        for control in [self.session_timezone, self.session_open, self.session_close, self.holidays]:
            control.disabled = self.busy or self.session_schedule is not None or self.time_basis.value == "wall"
        relevant = METHOD_EXPLANATIONS[self.method.value]["parameters"]
        for name, control in self.params.items():
            control.disabled = self.busy or name not in relevant
            self._param_sliders[name].disabled = self.busy or name not in relevant
            self._param_cards[name].layout.display = "" if name in relevant else "none"

    def _render_explanation(self):
        # UI LOGIC: Group native mathematics into progressive sections without removing any formula or example.
        from IPython.display import HTML, Math, display
        card = METHOD_EXPLANATIONS[self.method.value]
        applied = self.applied_config.method if self.applied_config else "none yet"
        groups = [("Detection method · equations and worked example", card["steps"]),
                  ("Shared preparation · timestamps, cohorts and support", COMMON_STEPS),
                  ("Trading clock · irregular events and session boundaries", CLOCK_STEPS),
                  ("Fitting policy · eligibility and suggested influence weights", FITTING_STEPS)]
        sections = []
        for index, (_, steps) in enumerate(groups):
            section = self.w.Output(layout=self.w.Layout(width="100%")).add_class("jf-math-section")
            with section:
                for title, prose, formula in steps:
                    display(HTML(f'<h4>{escape(title)}</h4><p>{escape(prose)}</p>'))
                    for block in math_display_blocks(formula):
                        display(Math(block))
                if index == 0:
                    title, url = card["reference"]
                    display(HTML(f'<div class="jf-example"><h4>Worked numerical example</h4><p>{escape(card["example"])}</p></div>'
                                 f'<h4>Assumptions and limitations</h4><p>{escape(card["tradeoffs"])}</p>'
                                 f'<p class="jf-help">Reference: <a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(title)}</a></p>'))
            sections.append(section)
        # UI LOGIC: The parameter audit follows the current selection; charts and exports keep their applied snapshot.
        parameters = self.w.Output(layout=self.w.Layout(width="100%")).add_class("jf-math-section")
        rows = [(name, self.params[name].value, PARAMETER_HELP[name]) for name in card["parameters"]]
        rows += [("horizon", self.horizon.value, "Maximum reference distance in the selected clock"),
                 ("max_gap", self.max_gap.value, "Reference break gap in the selected clock"),
                 ("time_basis", self.time_basis.value, "Trading exposure or wall-clock distance"),
                 ("session_timezone", self.session_timezone.value, "Market calendar timezone"),
                 ("session_open", self.session_open.value, "Weekday regular open"),
                 ("session_close", self.session_close.value, "Weekday regular close"),
                 ("holidays", self.holidays.value, "Explicit full-day closures")]
        with parameters:
            display(HTML(_table_html(pd.DataFrame(rows, columns=["parameter", "pending value", "effect"]))))
            if self.session_schedule is not None:
                display(HTML('<p class="jf-help">An authoritative session_schedule was supplied; trading mode uses only its intervals and overrides weekday calendar controls.</p>' + _table_html(self.session_schedule)))
        sections.append(parameters)
        accordion = self.w.Accordion(children=sections, selected_index=0)
        for index, (title, _) in enumerate(groups):
            accordion.set_title(index, title)
        accordion.set_title(4, "Selected parameter values · complete audit")
        with self.explanation:
            self.explanation.clear_output(wait=True)
            display(HTML(f'<h3>{escape(card["name"])}</h3><p class="jf-help">{escape(card["mode"])} · Explanation: selected method {escape(self.method.value)} · Applied plots/exports: {escape(applied)}</p><p>{escape(card["summary"])}</p>'))
            display(accordion)
        # UI LOGIC: Release superseded widget models after replacing their views, avoiding stale math/control audits.
        previous_widgets = self._explanation_widgets
        self._explanation_widgets = (*sections, accordion)
        for previous_widget in previous_widgets:
            previous_widget.close()

    def _pending_changed(self, _=None):
        # UI LOGIC: Reverting edits restores the applied badge without running algorithms.
        if self.result is None:
            state, badge, label = "ready", "Ready", "Choose settings and Apply."
        elif self._state() != self._applied_state:
            state, badge, label = "pending", "Pending settings", "Apply to update plots and exports."
        else:
            state, badge, label = "applied", "Applied", "Plots and exports match these settings."
        self.pending.value = f'<div class="jf-state jf-state-{state}"><span class="jf-badge">{badge}</span>{label}</div>'
        self._help_changed()
        self._render_explanation()

    def _lock(self, busy):
        # UI LOGIC: Keep state fixed during computation and allow export only after a successful Apply.
        self.busy = busy
        for control in [*self._controls, self.cusip, self.apply_button, self.compare_button]:
            control.disabled = busy
        self.export_button.disabled = busy or self.result is None
        self._help_changed()

    def _selected(self, frame):
        # Input: CUSIP=['A','B','A'], jf_row_id=[0,1,2], selected='A'.
        # Output: retained jf_row_id=[0,2], in source order.
        # Trick: Exact equality keeps string CUSIPs distinct from numeric IDs; original dataframe index labels are irrelevant.
        # CORE LOGIC: STEP 1 — Restrict display records to the exact selected instrument.
        return frame.loc[frame[self.mapping["cusip_col"]].eq(self.cusip.value)].copy()

    def _render(self, result):
        # PLOTTING LOGIC: Native widget views use bundled Plotly assets, avoiding scripts inside sanitized HTML outputs.
        from .plots import trade_figure, diagnostic_figure, METHOD_LABELS
        from . import summarize
        selected = self._selected(result)
        figure = trade_figure(selected, cusip_col=self.mapping["cusip_col"], spread_col=self.mapping["spread_col"],
                              unit=self.unit, max_gap=self.applied_config.max_gap if self.applied_config else "1D",
                              title=f'{self.cusip.value} · {METHOD_LABELS[self.applied_config.method] if self.applied_config else "Review"}')
        diagnostic = diagnostic_figure(selected, spread_col=self.mapping["spread_col"], unit=self.unit)
        summary = summarize(result, cusip_col=self.mapping["cusip_col"])
        return selected, _notebook_figure(figure), _notebook_figure(diagnostic), summary, fitting_statistics(result, cusip_col=self.mapping["cusip_col"])

    def _publish(self, selected, figure, diagnostic, summary, fitting):
        # UI LOGIC: Replace output areas only after engine and plot construction both succeed.
        from IPython.display import HTML, display
        self.kpis.value = _kpi_html(evaluation_summary(selected))
        with self.chart:
            self.chart.clear_output(wait=True)
            display(HTML('<div class="jf-section-heading"><h4>Spread history &amp; review flags</h4><span class="jf-help">Actual trade timestamps · UTC</span></div>'))
            display(figure)
            display(HTML('<p class="jf-help">All timed finite trades are shown. Red crosses indicate statistical outliers; amber markers identify level changes or provisional jumps. Hover for scores, reference support and the reason for each decision.</p>'))
        # UI LOGIC: Keep the liquidity chart immediately visible and progressively disclose the complete audit tables.
        audits = audit_tables(selected)
        support_columns = ["jf_time", "jf_status", "jf_n_reference", "jf_reference_span_minutes",
                           "jf_reference_density_per_hour", "jf_scale", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary"]
        overview_columns = [self.mapping["cusip_col"], "rows", "evaluated", "flagged", "fit_eligible", "flagged_rate", "coverage"]
        overview = summary[[name for name in overview_columns if name in summary]]
        full_summary = ('<details class="jf-note"><summary>Full bond statistics · all diagnostic columns</summary>' +
                        _table_html(summary) + '</details>')
        groups = [("All bonds · evaluation and fitting coverage", '<h4>Bond overview · applied method</h4>' + _table_html(overview) + full_summary +
                   '<h4>All bonds · fitting coverage and retention</h4>' + _table_html(fitting) +
                   '<p class="jf-help">Hard fit: jf_fit_eligible / status ok and unflagged. Soft fit: status ok or outlier with positive suggested influence weight. Provisional jumps, ambiguous transitions, invalid inputs, unsupported rows and solver failures are excluded by default. Coverage includes all supplied rows; weight sum is not an effective sample size.</p>'),
                  ("Selected bond · daily support and decision reasons", '<h4>Observed daily support</h4>' + _table_html(audits["daily"]) +
                   '<h4>Flag and support reasons</h4>' + _table_html(audits["reasons"])),
                  ("Selected bond · liquidity and local uncertainty", _table_html(selected[[name for name in support_columns if name in selected]])),
                  ("Selected bond · complete trade audit", _table_html(selected))]
        sections = []
        for _, contents in groups:
            section = self.w.Output(layout=self.w.Layout(width="100%"))
            with section:
                display(HTML(contents))
            sections.append(section)
        accordion = self.w.Accordion(children=sections, selected_index=0)
        for index, (title, _) in enumerate(groups):
            accordion.set_title(index, title)
        with self.statistics:
            self.statistics.clear_output(wait=True)
            display(HTML('<div class="jf-section-heading"><h4>Liquidity &amp; local uncertainty</h4><span class="jf-help">Selected bond · applied method</span></div>'))
            display(diagnostic)
            display(HTML('<p class="jf-help">Review support density and uncertainty before using flags for fitting. Expand a section below for exact counts and original-order trade diagnostics.</p>'))
            display(accordion)
        # UI LOGIC: Keep expandable views alive while closing every superseded statistics widget.
        previous_statistics = self._statistics_widgets
        self._statistics_widgets = (*sections, accordion)
        for previous_widget in previous_statistics:
            previous_widget.close()
        # UI LOGIC: Keep live plot models referenced and close superseded views after publishing replacements.
        previous_widgets = self._figure_widgets
        self._figure_widgets = (figure, diagnostic)
        for previous_widget in previous_widgets:
            previous_widget.close()

    def run(self, _=None):
        """Apply the settings to every supplied row and retain a successful snapshot."""
        # UI LOGIC: Failed pending changes preserve the previous valid result and export settings.
        from . import filter_trades
        if self.busy:
            return self
        self._lock(True)
        self.status.value = '<div class="jf-status">Evaluating supplied trades…</div>'
        previous = self.applied_config
        try:
            config = self._configuration()
            schedule = self.session_schedule.copy(deep=True) if self.session_schedule is not None else None
            result = filter_trades(self.data, config, **self.mapping, session_schedule=schedule)
            self.applied_config = config
            rendered = self._render(result)
            self._publish(*rendered)
            self.result, self._applied_state, self.comparison = result, self._state(), None
            self.applied_schedule = schedule
            self.method_output.clear_output()
            self._clear_comparison_widget()
            self.status.value = f'<div class="jf-status">Applied {escape(config.method)} to {len(result):,} supplied rows. Charts focus on {escape(str(self.cusip.value))}.</div>'
        except Exception as exc:
            self.applied_config = previous
            self.status.value = '<div class="jf-status jf-status-error">Could not apply settings: ' + escape(str(exc)) + '</div>'
        finally:
            self._lock(False)
            self._pending_changed()
        return self

    def _focus_changed(self, _=None):
        # UI LOGIC: Reuse applied annotations when browsing bonds; no algorithm is rerun.
        if self.result is None or self.busy:
            return
        self._publish(*self._render(self.result))
        self.status.value = f'<div class="jf-status">Viewing {escape(str(self.cusip.value))} · applied {escape(self.applied_config.method)} · {len(self.result):,} supplied rows.</div>'
        self.comparison = None
        self.method_output.clear_output()
        self._clear_comparison_widget()

    def _clear_comparison_widget(self):
        # UI LOGIC: Closed comparison models cannot retain stale applied settings in the notebook frontend.
        if self._comparison_widget is not None:
            self._comparison_widget.close()
            self._comparison_widget = None

    def compare(self, _=None):
        # UI LOGIC: Compare explicitly using the applied settings, even when new edits are pending.
        from IPython.display import HTML, display
        from .plots import comparison_figure
        if self.result is None or self.busy:
            self.status.value = '<div class="jf-status">Apply a filter before comparing methods.</div>'
            return
        self._lock(True)
        try:
            table = method_comparison(self._selected(self.data), self.applied_config, **self.mapping, session_schedule=self.applied_schedule)
            figure = _notebook_figure(comparison_figure(table))
            with self.method_output:
                self.method_output.clear_output(wait=True)
                display(HTML('<p class="jf-help">Same selected bond, same applied hyperparameters. Differences measure sensitivity, not accuracy. Centered and causal methods use different information sets.</p>'))
                display(figure)
                display(HTML(_table_html(table)))
            self._clear_comparison_widget()
            self._comparison_widget = figure
            self.comparison = table
            self.tabs.selected_index = 2
        except Exception as exc:
            self.status.value = '<div class="jf-status jf-status-error">Could not compare methods: ' + escape(str(exc)) + '</div>'
        finally:
            self._lock(False)

    def export(self, _=None):
        """Save original-order annotations and the last successfully applied settings."""
        # FILE IO LOGIC: A timestamped folder keeps prior reviews; exports never use pending controls.
        from . import summarize
        from .plots import trade_figure
        if self.result is None:
            return None
        output = Path(self.export_path.value).expanduser() / datetime.now(timezone.utc).strftime("review_%Y%m%dT%H%M%S_%fZ")
        try:
            output.mkdir(parents=True, exist_ok=False)
            self.result.to_csv(output / "annotated_trades.csv", index=False)
            summarize(self.result, cusip_col=self.mapping["cusip_col"]).to_csv(output / "bond_summary.csv", index=False)
            fitting_statistics(self.result, cusip_col=self.mapping["cusip_col"]).to_csv(output / "fitting_coverage.csv", index=False)
            for name, table in audit_tables(self._selected(self.result)).items():
                table.to_csv(output / f"selected_bond_{name}.csv", index=False)
            settings = dict(config=asdict(self.applied_config), mapping=self.mapping, unit=self.unit,
                            selected_cusip=str(self.cusip.value), rows=len(self.result), source="applied result",
                            calendar=self.result.attrs.get("jump_filter", {}).get("calendar"))
            if self.applied_schedule is not None:
                self.applied_schedule.to_csv(output / "session_schedule.csv", index=False)
            (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
            figure = trade_figure(self._selected(self.result), cusip_col=self.mapping["cusip_col"],
                                  spread_col=self.mapping["spread_col"], unit=self.unit, title=str(self.cusip.value),
                                  max_gap=self.applied_config.max_gap)
            figure.write_html(output / "selected_bond.html", include_plotlyjs=True)
            if self.comparison is not None:
                self.comparison.to_csv(output / "method_comparison.csv", index=False)
            self.status.value = '<div class="jf-status">Saved applied review: ' + escape(str(output.resolve())) + '</div>'
            return output
        except Exception as exc:
            self.status.value = '<div class="jf-status jf-status-error">Could not export: ' + escape(str(exc)) + '</div>'
            return None

    def _ipython_display_(self):
        # UI LOGIC: Displaying a workbench does not rerun filtering or change caller data.
        from IPython.display import display
        display(self.widget)


def show_filter(data, **kwargs):
    """Display the complete dashboard inside Jupyter; call .run() to apply settings.

    Return a FilterDashboard with native ipywidgets/Plotly views, including method
    explanations, mathematical formulas, statistics and applied-data exports.
    """
    # UI LOGIC: Return the workbench so callers can inspect its applied result and exports.
    from IPython.display import display
    panel = FilterDashboard(data, **kwargs)
    display(panel.widget)
    return panel
