"""Optional notebook workbench and reusable dashboard statistics."""
# SETUP LOGIC: The notebook dependency is imported only when constructing a workbench.
from dataclasses import asdict, replace
from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .explanations import METHOD_HELP, METHOD_EXPLANATIONS, COMMON_STEPS, CLOCK_STEPS, FITTING_STEPS, POLICY_STEPS, PARAMETER_HELP, math_display_blocks

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
.jf-workbench .jf-policy-controls>.jf-control{flex:0 0 auto}
.jf-workbench .jf-policy-toggle{flex:0 0 auto;width:100%;min-height:34px;display:flex;align-items:center}
.jf-workbench .jf-policy-toggle>.widget-label-basic{width:auto!important;overflow:visible;white-space:normal}
"""

# PLOTTING LOGIC: Decorate the bundled Plotly module using anywidget's supported factory and lifecycle hooks.
# Trick: Native outputs may render while detached or hidden; observe their real width after attachment and tab reveals.
# Plotly's responsive handler recalculates axes and SVG geometry; no chart is visually stretched by CSS.
_RESPONSIVE_WIDGET_ESM = """
// CACHEING LOGIC: Identical anywidget modules share one bundled import instead of reparsing Plotly for every chart.
// Trick: Cache only the imported module; its default factory below creates separate model state for each figure.
let bundledImport;
const loadBundle = () => {
  if (!bundledImport) {
    const moduleUrl = URL.createObjectURL(new Blob([__PLOTLY_MODULE__], {type: "text/javascript"}));
    bundledImport = import(moduleUrl).finally(() => URL.revokeObjectURL(moduleUrl));
  }
  return bundledImport;
};
// PLOTTING LOGIC: Each widget retains the original factory, initialize, render, and cleanup lifecycle.
export default async () => {
  const bundled = await loadBundle();
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
    # Trick: Algorithm statuses preserve this denominator when an optional policy retains or excludes an unsupported trade.
    # CORE LOGIC: STEP 1 — Build separate evaluation and review denominators.
    status = annotated.get("jf_algorithm_status", annotated["jf_status"])
    evaluated = status.isin(["ok", "outlier", "provisional_jump"])
    count = int(evaluated.sum())
    outliers = int(annotated["jf_is_outlier"].sum())
    stats = dict(total=len(annotated), evaluated=count, outliers=outliers,
                flag_rate=int((annotated["jf_is_outlier"] & evaluated).sum()) / count if count else np.nan,
                invalid=int(status.eq("invalid_input").sum()),
                unsupported=int(status.eq("insufficient_history").sum()),
                provisional=int(status.eq("provisional_jump").sum()),
                regime_changes=int(annotated["jf_regime_change"].sum()))
    # Input: Step 1's five records plus fit_eligible=[True,False,False,False,False]; algorithm flags equal final flags and all optional masks are absent.
    # Output: total=5,evaluated=3,outliers=2,flag_rate=2/3,invalid=1,unsupported=1,provisional=1,regime_changes=0,fit_eligible=1,algorithm_outliers=2,retained_uncertain=0,quantity_rejected=0,max_deviation_rejected=0.
    # Trick: Optional masks default to false; an unassessed policy rejection stays outside the evaluated flag-rate denominator.
    # CORE LOGIC: STEP 2 — Report final retention independently from algorithm assessment.
    empty = pd.Series(False, index=annotated.index)
    stats.update(fit_eligible=int(annotated.get("jf_fit_eligible", status.eq("ok") & ~annotated["jf_is_outlier"]).sum()),
                 algorithm_outliers=int(annotated.get("jf_algorithm_is_outlier", annotated["jf_is_outlier"]).sum()),
                 retained_uncertain=int(annotated.get("jf_policy_retained_uncertain", empty).sum()),
                 quantity_rejected=int(annotated.get("jf_policy_quantity_rejected", empty).sum()),
                 max_deviation_rejected=int(annotated.get("jf_policy_max_deviation", empty).sum()))
    return stats


def fitting_statistics(annotated, *, cusip_col="CUSIP"):
    """Report hard/soft fitting coverage without treating abstentions as clean."""
    # Input: algorithm status=['ok','outlier','insufficient_history'],final status=['ok','outlier','retained_uncertain'],fit_eligible=[True,False,True],weight=[1,.2,1].
    # Output: frame gains _hard=[True,False,True],_soft=[True,True,True],_evaluated=[True,True,False],_weight=[1,.2,1],_solver=[False,False,False],_ambiguous=[False,False,False],_retained_uncertain=[False,False,True].
    # Trick: Policy retention changes fit selection without presenting an unsupported trade as algorithm-assessed evidence.
    # CORE LOGIC: STEP 1 — Separate assessed, retained, and softly weighted observations.
    status = annotated["jf_status"]
    algorithm = annotated.get("jf_algorithm_status", status)
    hard = annotated.get("jf_fit_eligible", status.eq("ok") & ~annotated["jf_is_outlier"])
    soft = status.isin(["ok", "outlier", "retained_uncertain"]) & annotated["jf_weight"].gt(0)
    frame = annotated.assign(_evaluated=algorithm.isin(["ok", "outlier", "provisional_jump"]),
                             _hard=hard, _soft=soft,
                             _weight=annotated["jf_weight"].where(soft, 0.0),
                             _solver=algorithm.eq("solver_not_converged"), _ambiguous=algorithm.eq("ambiguous_transition"),
                             _retained_uncertain=status.eq("retained_uncertain"))
    # Input: CUSIP=['A','A','A'],evaluated=[True,True,False],hard=[True,False,True],soft=[True,True,True],weight=[1,.2,1].
    # Output: A supplied=3,evaluated=2,hard_fit_rows=2,soft_fit_rows=3,soft_weight_sum=2.2,retained_uncertain=1,solver_failures=0,ambiguous_transitions=0.
    # Trick: Influence-weight sums are neither effective sample sizes nor inverse variances.
    # CORE LOGIC: STEP 2 — Aggregate the distinct algorithm and final fitting populations.
    table = frame.groupby(cusip_col, dropna=False, sort=False).agg(
        supplied=("jf_row_id", "size"), evaluated=("_evaluated", "sum"), hard_fit_rows=("_hard", "sum"),
        soft_fit_rows=("_soft", "sum"), soft_weight_sum=("_weight", "sum"), solver_failures=("_solver", "sum"),
        ambiguous_transitions=("_ambiguous", "sum"), retained_uncertain=("_retained_uncertain", "sum"))
    # Input: table index=['A'],supplied=[4],evaluated=[3],hard_fit_rows=[1],soft_fit_rows=[2],soft_weight_sum=[1.2],solver_failures=[0],ambiguous_transitions=[0].
    # Output: one row CUSIP='A' retains its statistics and adds coverage=.75,hard_retention=.25.
    # Trick: Denominators include every supplied row, including invalid and unsupported records.
    # CORE LOGIC: STEP 3 — Expose both evaluation and downstream retention coverage.
    table["coverage"] = table["evaluated"] / table["supplied"]
    table["hard_retention"] = table["hard_fit_rows"] / table["supplied"]
    return table.reset_index()


def method_comparison(data, config, *, cusip_col="CUSIP", time_col="time", spread_col="spread", quantity_col=None, timezone="UTC", session_schedule=None):
    """Evaluate all methods on an explicitly selected input population."""
    # SETUP LOGIC: Public engine imports avoid a dashboard/engine initialization cycle.
    from . import METHODS, filter_trades
    records = []
    # Input: data={CUSIP:['A'],time:['2026-10-01T10:00Z'],spread:[100]}, config=FilterConfig().
    # Output: each supported method (e.g. hampel) produces
    # each row has total=1,evaluated=0,outliers=0,flag_rate=NaN,invalid=0,
    # unsupported=1,provisional=0,regime_changes=0,fit_eligible=0,algorithm_outliers=0,
    # retained_uncertain=0,quantity_rejected=0,max_deviation_rejected=0.
    # Trick: Every method sees the same rows and hyperparameters; no result is chosen by its flag rate.
    # CORE LOGIC: STEP 1 — Recompute each algorithm on the same review population.
    for method in METHODS:
        result = filter_trades(data, replace(config, method=method), cusip_col=cusip_col,
                               time_col=time_col, spread_col=spread_col, quantity_col=quantity_col, timezone=timezone, session_schedule=session_schedule)
        records.append(dict(method=method, **evaluation_summary(result)))
    return pd.DataFrame(records)


def audit_tables(annotated):
    """Return daily support and exact reason counts for one review population."""
    # Input: jf_time=['2026-10-01T10:00Z','2026-10-01T11:00Z','2026-10-02T10:00Z']; statuses=['ok','outlier','invalid_input']; flags=[False,True,False].
    # Output: daily UTC rows=[('2026-10-01',trades=2,evaluated=2,flagged=1),('2026-10-02',trades=1,evaluated=0,flagged=0)].
    # Trick: Only observed UTC dates are listed; untimed records remain in the reason table without fabricated calendar bins.
    # CORE LOGIC: STEP 1 — Aggregate actual calendar support and flags on observed days.
    frame = annotated.assign(_day=annotated["jf_time"].dt.floor("D"),
                             _evaluated=annotated.get("jf_algorithm_status", annotated["jf_status"]).isin(["ok", "outlier", "provisional_jump"]))
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
    tables = {"daily": daily, "reasons": reasons}
    # Input: policy reasons=['algorithm_decision','small_suspicious','retained_uncertain'],flags=[False,True,False],fit_eligible=[True,False,True].
    # Output: reason rows in lexical order each have trades=1,flagged=[0,0,1],fit_eligible=[1,1,0].
    # Trick: Keep exact engine reason strings so policy decisions remain separate from statistical evidence.
    # CORE LOGIC: STEP 3 — Aggregate the optional policy decision audit.
    if "jf_policy_reason" in annotated:
        tables["policy_reasons"] = annotated.groupby("jf_policy_reason", dropna=False, sort=True).agg(
            trades=("jf_row_id", "size"), flagged=("jf_is_outlier", "sum"), fit_eligible=("jf_fit_eligible", "sum")).reset_index()
    return tables


class ReviewWorkspace:
    """Index a fixed source once and retain a bounded cache of applied reviews.

    ``selected`` evaluates one exact CUSIP; ``all`` explicitly evaluates every
    source row, including missing CUSIPs. Cached frames are immutable snapshots
    owned by the dashboard and retain global positional ``jf_row_id`` values.
    """

    def __init__(self, data, mapping, *, max_entries=16, max_bytes=64 * 1024 * 1024):
        # SETUP LOGIC: One source reference and one positional index avoid full-frame copies on every UI event.
        self.data, self.mapping = data, dict(mapping)
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cache, self.cache_bytes = OrderedDict(), 0
        # Input: CUSIP=['A','B','A'], source index=[7,7,2].
        # Output: positions={'A':array([0,2]),'B':array([1])}; bonds=['A','B'].
        # Trick: Positional indexing preserves duplicate source labels and leading-zero CUSIPs without repeatedly scanning all rows.
        # CORE LOGIC: STEP 1 — Index the supplied instruments in their original encounter order.
        self.positions = data.groupby(mapping["cusip_col"], sort=False, observed=True).indices
        self.bonds = list(self.positions)

    def source_bond(self, bond):
        # Input: source CUSIP=['A','B','A'],spread=[100,200,101],positions['A']=[0,2].
        # Output: CUSIP=['A','A'],spread=[100,101], with the original two index labels.
        # Trick: iloc uses the one-time index; its small selected frame cannot mutate the retained source through dashboard code.
        # CORE LOGIC: STEP 1 — Materialize only the requested instrument's source rows.
        return self.data.iloc[self.positions[bond]].copy()

    def review(self, config, *, scope="selected", bond=None, session_schedule=None):
        # CACHEING LOGIC: Workspace identity fixes source and mapping; settings and calendar identify remaining dependencies.
        from . import filter_trades, summarize
        calendar = sha256(session_schedule.to_csv(index=False).encode()).hexdigest() if session_schedule is not None else None
        key = (scope, bond if scope == "selected" else None, json.dumps(asdict(config), sort_keys=True, default=str), calendar)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        # VALIDATION LOGIC: An explicit population is required rather than silently interpreting an unknown scope.
        if scope not in {"selected", "all"}:
            raise ValueError("Review scope must be 'selected' or 'all'.")
        if scope == "selected" and bond not in self.positions:
            raise ValueError("Choose an available CUSIP before applying the selected-bond review.")
        # Input: source positions=[0,1,2],CUSIP=['A','B','A'],scope='selected',bond='A'; local filter jf_row_id=[0,1].
        # Output: result contains the two A rows, with jf_row_id=[0,2]; scope='all' instead retains jf_row_id=[0,1,2].
        # Trick: The engine numbers its supplied population; remapping restores source-wide positional identities without index-label joins.
        # CORE LOGIC: STEP 1 — Evaluate the explicitly chosen population and retain source row identities.
        population = self.source_bond(bond) if scope == "selected" else self.data
        result = filter_trades(population, config, **self.mapping, session_schedule=session_schedule)
        if scope == "selected":
            result["jf_row_id"] = self.positions[bond][result["jf_row_id"].to_numpy(dtype=int)]
        summary = summarize(result, cusip_col=self.mapping["cusip_col"])
        fitting = fitting_statistics(result, cusip_col=self.mapping["cusip_col"])
        # CACHEING LOGIC: Large batch frames remain available as the applied result but are not duplicated in the bounded cache.
        record = dict(result=result, summary=summary, fitting=fitting, scope=scope, bond=bond, key=key)
        size = int(result.memory_usage(index=True, deep=True).sum() + summary.memory_usage(index=True, deep=True).sum() + fitting.memory_usage(index=True, deep=True).sum())
        if size <= self.max_bytes and self.max_entries > 0:
            while self.cache and (len(self.cache) >= self.max_entries or self.cache_bytes + size > self.max_bytes):
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed["cache_bytes"]
            record["cache_bytes"] = size
            self.cache[key], self.cache_bytes = record, self.cache_bytes + size
        return record


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
    labels.update(retained_uncertain="Retained uncertain", algorithm_outliers="Algorithm flags", quantity_rejected="Small suspicious exclusions",
                  max_deviation_rejected="Trend-cap exclusions", jf_policy_reason="Policy decision", jf_algorithm_status="Algorithm status",
                  jf_algorithm_reason="Algorithm reason", jf_algorithm_is_outlier="Algorithm outlier", jf_quantity="Normalized notional",
                  jf_quantity_notional="Normalized notional", jf_quantity_valid="Valid Quantity", jf_support_trend="Independent support trend",
                  jf_support_reliable="Reliable support", jf_support_residual="Support-trend residual", jf_support_distance_bps="Support distance · bp")
    for name in ["coverage", "flagged_rate", "flag_rate", "hard_retention"]:
        if name in shown:
            shown[name] = shown[name].map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
    shown = shown.rename(columns=lambda name: labels.get(name, str(name).replace("_", " ").capitalize()))
    return (f'<p class="jf-help">Showing {len(shown):,} of {len(frame):,} rows.</p><div class="jf-table">' +
            shown.to_html(index=False, escape=True, border=0, na_rep="—", float_format=lambda x: f"{x:.5g}") + "</div>")


def _kpi_html(stats):
    # UI LOGIC: Words, exact counts and explicit denominators accompany status colors.
    rate = f'{stats["flag_rate"]:.1%}' if np.isfinite(stats["flag_rate"]) else "—"
    items = [("Supplied trades", f'{stats["total"]:,}'), ("Algorithm assessed", f'{stats["evaluated"]:,}'),
             ("Final outlier flags", f'{stats["outliers"]:,}'), ("Flags / algorithm assessed", rate),
             ("Insufficient support", f'{stats["unsupported"]:,}'), ("Invalid input", f'{stats["invalid"]:,}')]
    classes = ["", "", " jf-kpi-alert", " jf-kpi-rate", "", ""]
    return ('<div class="jf-kpis">' + ''.join(f'<div class="jf-kpi{style}"><strong>{value}</strong><span>{label}</span></div>' for (label, value), style in zip(items, classes)) + '</div>' +
            f'<p class="jf-help">Final fit eligible: {stats["fit_eligible"]:,} · explicitly retained uncertain: {stats["retained_uncertain"]:,} · original algorithm flags: {stats["algorithm_outliers"]:,} · small suspicious exclusions: {stats["quantity_rejected"]:,} · support-trend cap exclusions: {stats["max_deviation_rejected"]:,}. The assessed flag rate excludes unassessed policy decisions.</p>')


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


def _replace_outputs(output, *objects):
    """Replace a notebook section without capturing the kernel's display stream."""
    # UI LOGIC: Format native HTML, mathematics and widget references before publishing one complete section.
    # Input: prior outputs=[HTML('old')], objects=[HTML('<b>new</b>'), FigureWidget(model_id='abc')].
    # Output: exactly two display_data records: text/html='<b>new</b>' and widget-view model_id='abc'; msg_id=''.
    # Trick: Assigning the tuple replaces old records. Empty msg_id disables IOPub capture, so concurrent frontend views cannot echo displays into this section.
    from IPython.core.interactiveshell import InteractiveShell
    formatter = InteractiveShell.instance().display_formatter.format
    records = []
    for obj in objects:
        data, metadata = formatter(obj)
        records.append(dict(output_type="display_data", data=data, metadata=metadata))
    # UI LOGIC: Synchronize a single owned output snapshot; empty objects explicitly clears the section.
    with output.hold_sync():
        output.msg_id = ""
        output.outputs = tuple(records)


class FilterDashboard:
    """Live Jupyter widgets with an explicit applied result and bundled Plotly charts.

    Install ``jump-filter[notebook]`` for a complete Notebook 7/JupyterLab 4 setup,
    or ``jump-filter[dashboard]`` inside an existing Jupyter kernel. No Streamlit
    process is required. A running kernel handles filtering and control callbacks.
    ``result`` contains the applied population: the selected bond by default,
    or every supplied row after choosing All bonds (batch) and applying.
    """

    def __init__(self, data, *, config=None, cusip_col="CUSIP", time_col="time", spread_col="spread", quantity_col=None, timezone="UTC", unit="bp", session_schedule=None):
        # SETUP LOGIC: Copy caller data; keep optional notebook dependencies outside package imports.
        import ipywidgets as w
        from . import FilterConfig, METHODS
        from .plots import METHOD_LABELS
        self.w, self.data = w, data.copy()
        self.config = config or FilterConfig(time_basis="trading")
        self.mapping = dict(cusip_col=cusip_col, time_col=time_col, spread_col=spread_col, quantity_col=quantity_col, timezone=timezone)
        self.unit, self.result, self.applied_config = unit, None, None
        self.applied_scope, self._applied_record = "selected", None
        self._view_bond = None
        self.session_schedule = session_schedule.copy(deep=True) if session_schedule is not None else None
        self.applied_schedule = None
        self.comparison, self._applied_state, self.busy = None, None, False
        self._figure_widgets, self._comparison_widget = (), None
        self._displayed_request = None
        self._explanation_widgets = ()
        self._statistics_widgets = ()
        # VALIDATION LOGIC: Fail early for a bad mapping, before showing controls that cannot run.
        if not all(column in data for column in [cusip_col, time_col, spread_col]):
            raise ValueError("CUSIP, time and spread mappings must name existing columns.")
        if quantity_col is not None and quantity_col not in data:
            raise ValueError("Quantity mapping must name an existing column or be None.")
        # UI LOGIC: Exact values drive selection; text labels are presentation only.
        self.workspace = ReviewWorkspace(self.data, self.mapping)
        values = self.workspace.bonds
        initial_choices = values[:100] if len(values) > 250 else values
        self.cusip = w.Dropdown(description="Bond / CUSIP", options=[(str(value), value) for value in initial_choices])
        self.cusip_search = w.Text(description="Find CUSIP", placeholder="Type part of a CUSIP, then Enter", continuous_update=False)
        self.cusip_matches = w.HTML()
        self.scope = w.Dropdown(description="Review population", options=[("Selected bond · interactive", "selected"), ("All bonds · batch", "all")], value="selected")
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
        # UI LOGIC: Optional policy controls are independent of the selected statistical method.
        self.quantity_rule = w.Checkbox(value=self.config.quantity_rule, description="Enable Quantity-sensitive screening")
        self.max_deviation_rule = w.Checkbox(value=self.config.max_deviation_rule, description="Enable maximum support-trend deviation")
        self.quantity_col = w.Dropdown(description="Quantity column", options=[("Not mapped", None), *[(str(name), name) for name in data.columns]], value=quantity_col)
        quantity_options = [("Raw amount · 1MM = 1,000,000", 1.), ("Thousands · 1MM = 1,000", 1000.), ("Millions · 1MM = 1", 1000000.)]
        spread_options = [("Basis points · 1 bp = 1", 1.), ("Percentage points · 1 bp = 0.01", .01), ("Decimal · 1 bp = 0.0001", .0001)]
        if self.config.quantity_multiplier not in [value for _, value in quantity_options]:
            quantity_options.append((f"Custom multiplier · {self.config.quantity_multiplier:g}", float(self.config.quantity_multiplier)))
        if self.config.spread_units_per_bp not in [value for _, value in spread_options]:
            spread_options.append((f"Custom units per bp · {self.config.spread_units_per_bp:g}", float(self.config.spread_units_per_bp)))
        self.quantity_multiplier = w.Dropdown(description="Quantity units", options=quantity_options, value=float(self.config.quantity_multiplier))
        self.spread_units_per_bp = w.Dropdown(description="Spread numeric units", options=spread_options, value=float(self.config.spread_units_per_bp))
        self.policy_params, self._policy_sliders, policy_cards = {}, {}, []
        # UI LOGIC: Start the raw-amount slider at 1 with unit steps so the default 1MM lies exactly on its frontend grid.
        # Trick: Float traits preserve precise typed amounts even between slider ticks; Apply reads the exact input value.
        for name, label, lower, upper, step in [("quantity_threshold", "Small-trade threshold · notional amount", 1., 10000000., 1.), ("max_deviation_bps", "Maximum trend deviation · bp", .1, 100., .1)]:
            value = float(getattr(self.config, name))
            slider = w.FloatSlider(value=value, min=min(lower, value), max=max(upper, value), step=step, readout=False, continuous_update=False,
                                   description=label, tooltip=PARAMETER_HELP[name], style={"description_width": "0px"})
            number = w.BoundedFloatText(value=value, min=min(lower, value), max=max(upper, value), step=step,
                                       description=f"Exact value · {label}", tooltip="Exact value. " + PARAMETER_HELP[name], style={"description_width": "0px"})
            w.link((slider, "value"), (number, "value"))
            self.policy_params[name], self._policy_sliders[name] = number, slider
            values_row = w.HBox([slider, number], layout=w.Layout(width="100%")).add_class("jf-param-values")
            policy_cards.append(w.VBox([w.HTML(f'<div class="jf-param-title">{escape(label)}</div>'), values_row, w.HTML(f'<p class="jf-help">{escape(PARAMETER_HELP[name])}</p>')]).add_class("jf-param"))
        self._policy_controls = [self.quantity_rule, self.quantity_col, self.quantity_multiplier, self.max_deviation_rule, self.spread_units_per_bp, *self.policy_params.values()]
        self.horizon = w.Text(value=str(self.config.horizon), description="Reference horizon (e.g. 3D)")
        self.max_gap = w.Text(value=str(self.config.max_gap), description="Session break gap (e.g. 1D)")
        self.time_basis = w.Dropdown(options=[("Cumulative trading time", "trading"), ("Wall-clock time", "wall")], value=self.config.time_basis, description="Distance clock")
        self.session_timezone = w.Text(value=self.config.session_timezone, description="Market session timezone")
        self.session_open = w.Text(value=self.config.session_open, description="Weekday session opens")
        self.session_close = w.Text(value=self.config.session_close, description="Weekday session closes")
        self.holidays = w.Text(value=", ".join(self.config.holidays), description="Closed dates · YYYY-MM-DD, ...")
        self._calendar_controls = [self.time_basis, self.session_timezone, self.session_open, self.session_close, self.holidays]
        self._controls = [self.method, self.scope, self.horizon, self.max_gap, *self._calendar_controls, *self.params.values(), *self._policy_controls]
        for control in [self.cusip, self.cusip_search, self.scope, self.method, self.horizon, self.max_gap, self.export_path, *self._calendar_controls, self.quantity_col, self.quantity_multiplier, self.spread_units_per_bp]:
            control.add_class("jf-control")
            control.style.description_width = "initial"
        # UI LOGIC: Native checkboxes retain their horizontal label/input structure, separate from vertically labeled text controls.
        for control in [self.quantity_rule, self.max_deviation_rule]:
            control.add_class("jf-policy-toggle")
        # UI LOGIC: CUSIP selection changes the view only; parameter edits await explicit Apply.
        self.cusip.observe(self._focus_changed, names="value")
        self.cusip_search.observe(self._search_bonds, names="value")
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
        policy_grid = w.Box(policy_cards, layout=w.Layout(display="flex", flex_flow="row wrap")).add_class("jf-row")
        # UI LOGIC: Vertical policy controls use their natural heights; wrapping rows retain the shared horizontal flex sizing.
        # Trick: A 230px flex basis means height inside VBox, so override it only for this panel's direct control children.
        policies = w.VBox([self.quantity_rule, self._row(self.quantity_col, self.quantity_multiplier), self.max_deviation_rule,
                           self.spread_units_per_bp, policy_grid, w.HTML('<p class="jf-help">Both rules are optional. Small trades are excluded only with suspicious evidence. Known positive Quantity can retain weak-evidence records explicitly as retained uncertain; confirmed algorithm outliers remain excluded at every size. Independent support follows separate left/right trends where both sides have enough references; disagreement disables the hard cap. Only insufficient side counts allow a bracketed whole-neighborhood fallback. No Quantity or spread units are inferred from labels.</p>')]).add_class("jf-policy-controls")
        settings = w.Accordion(children=[w.VBox([parameter_grid, self._row(self.horizon, self.max_gap)]), calendar, policies], selected_index=None)
        settings.set_title(0, "Tune detection · relevant parameters and exact values")
        settings.set_title(1, "Trading calendar · sessions, closures and gaps")
        settings.set_title(2, "Optional screening · Quantity and support-trend distance")
        self.pending.add_class("jf-state-area")
        actions = self._row(self.apply_button, self.compare_button, self.pending).add_class("jf-actionbar")
        search = [self._row(self.cusip_search), self.cusip_matches] if len(values) > 250 else []
        selection = w.VBox([*search, self._row(self.cusip, self.method, self.scope), self.help,
                           w.HTML(f'<p class="jf-help">{len(self.data):,} source trades · {len(values):,} CUSIPs. Selected bond evaluates one instrument. All bonds explicitly runs the full population for portfolio statistics and exports.</p>')]).add_class("jf-selection")
        export_row = self._row(self.export_path, self.export_button).add_class("jf-export")
        self.widget = w.VBox([w.HTML("<style>" + STYLE + "</style>"), hero,
                              selection, actions, settings, self.status, self.kpis, self.tabs, note, export_row],
                              layout=w.Layout(width="100%", max_width="1500px")).add_class("jf-workbench")
        self._empty_outputs()
        self._help_changed()
        self._pending_changed()

    def _search_bonds(self, _=None):
        # UI LOGIC: Search labels locally; retaining the current choice prevents search text from running a filter.
        query, selected = self.cusip_search.value.casefold().strip(), self.cusip.value
        matches = [bond for bond in self.workspace.bonds if query in str(bond).casefold()]
        choices = matches[:100]
        if selected is not None and selected not in choices:
            choices.insert(0, selected)
        self.cusip.options = [(str(bond), bond) for bond in choices]
        self.cusip_matches.value = f'<p class="jf-help">{len(matches):,} matches · showing up to 100. Choose a CUSIP below; refine the search for more.</p>'

    def _row(self, *children):
        # UI LOGIC: Wrapping respects notebook pane width rather than browser viewport width.
        return self.w.Box(list(children), layout=self.w.Layout(display="flex", flex_flow="row wrap", width="100%")).add_class("jf-row")

    def _empty_outputs(self):
        # UI LOGIC: The first view explains the next action and gives each empty results tab a useful purpose.
        from IPython.display import HTML
        prompts = [(self.chart, "Ready to review", "Choose a bond and method, then Apply filter to see trades and outlier flags."),
                   (self.statistics, "Statistics follow your applied review", "Coverage, fitting retention and liquidity diagnostics will appear here."),
                   (self.method_output, "Compare detection methods", "Apply a filter, then Compare methods to inspect sensitivity on the selected bond.")]
        for output, title, message in prompts:
            _replace_outputs(output, HTML(f'<div class="jf-empty"><strong>{escape(title)}</strong>{escape(message)}</div>'))

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
                            quantity_rule=self.quantity_rule.value, quantity_multiplier=self.quantity_multiplier.value,
                            max_deviation_rule=self.max_deviation_rule.value, spread_units_per_bp=self.spread_units_per_bp.value,
                            **{name: control.value for name, control in self.params.items()},
                            **{name: control.value for name, control in self.policy_params.items()})

    def _help_changed(self, _=None):
        # UI LOGIC: Show and enable only parameters used by the currently selected method.
        card = METHOD_EXPLANATIONS[self.method.value]
        mode = card["mode"].partition(" · ")[0]
        self.help.value = (f'<div class="jf-method-context"><span class="jf-badge">{escape(mode)}</span>'
                           '<p class="jf-help">' + escape(METHOD_HELP[self.method.value]) + '</p></div>')
        if (self.quantity_rule.value or self.max_deviation_rule.value) and self.method.value == "causal_ewma":
            self.help.value += '<p class="jf-help">Optional screening uses a centered support trend and can use future trades; the final policy decisions are retrospective even though the original EWMA algorithm is causal.</p>'
        for control in [self.session_timezone, self.session_open, self.session_close, self.holidays]:
            control.disabled = self.busy or self.session_schedule is not None or self.time_basis.value == "wall"
        relevant = METHOD_EXPLANATIONS[self.method.value]["parameters"]
        for name, control in self.params.items():
            control.disabled = self.busy or name not in relevant
            self._param_sliders[name].disabled = self.busy or name not in relevant
            self._param_cards[name].layout.display = "" if name in relevant else "none"
        # UI LOGIC: Inactive policy values remain visible for auditing but cannot accidentally edit a running review.
        for control in [self.quantity_col, self.quantity_multiplier, self.policy_params["quantity_threshold"], self._policy_sliders["quantity_threshold"]]:
            control.disabled = self.busy or not self.quantity_rule.value
        for control in [self.policy_params["max_deviation_bps"], self._policy_sliders["max_deviation_bps"]]:
            control.disabled = self.busy or not self.max_deviation_rule.value
        self.spread_units_per_bp.disabled = self.busy or not (self.quantity_rule.value or self.max_deviation_rule.value)

    def _render_explanation(self):
        # UI LOGIC: Group native mathematics into progressive sections without removing any formula or example.
        from IPython.display import HTML, Math
        card = METHOD_EXPLANATIONS[self.method.value]
        applied = self.applied_config.method if self.applied_config else "none yet"
        groups = [("Detection method · equations and worked example", card["steps"]),
                  ("Shared preparation · timestamps, cohorts and support", COMMON_STEPS),
                  ("Trading clock · irregular events and session boundaries", CLOCK_STEPS),
                  ("Fitting policy · eligibility and suggested influence weights", FITTING_STEPS),
                  ("Optional screening · Quantity and reliable support trend", POLICY_STEPS)]
        sections = []
        for index, (_, steps) in enumerate(groups):
            section = self.w.Output(layout=self.w.Layout(width="100%")).add_class("jf-math-section")
            contents = []
            for title, prose, formula in steps:
                contents.append(HTML(f'<h4>{escape(title)}</h4><p>{escape(prose)}</p>'))
                contents.extend(Math(block) for block in math_display_blocks(formula))
            if index == 0:
                title, url = card["reference"]
                contents.append(HTML(f'<div class="jf-example"><h4>Worked numerical example</h4><p>{escape(card["example"])}</p></div>'
                                     f'<h4>Assumptions and limitations</h4><p>{escape(card["tradeoffs"])}</p>'
                                     f'<p class="jf-help">Reference: <a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(title)}</a></p>'))
            _replace_outputs(section, *contents)
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
        rows += [("quantity_rule", self.quantity_rule.value, "Enable the explicitly optional Quantity policy"),
                 ("quantity_col", self.quantity_col.value, "Mapped source Quantity column; required when Quantity screening is enabled"),
                 ("quantity_multiplier", self.quantity_multiplier.value, PARAMETER_HELP["quantity_multiplier"]),
                 ("quantity_threshold", self.policy_params["quantity_threshold"].value, PARAMETER_HELP["quantity_threshold"]),
                 ("max_deviation_rule", self.max_deviation_rule.value, "Enable reliable support-trend distance exclusion"),
                 ("max_deviation_bps", self.policy_params["max_deviation_bps"].value, PARAMETER_HELP["max_deviation_bps"]),
                 ("spread_units_per_bp", self.spread_units_per_bp.value, PARAMETER_HELP["spread_units_per_bp"])]
        contents = [HTML(_table_html(pd.DataFrame(rows, columns=["parameter", "pending value", "effect"])))]
        if self.session_schedule is not None:
            contents.append(HTML('<p class="jf-help">An authoritative session_schedule was supplied; trading mode uses only its intervals and overrides weekday calendar controls.</p>' + _table_html(self.session_schedule)))
        _replace_outputs(parameters, *contents)
        sections.append(parameters)
        accordion = self.w.Accordion(children=sections, selected_index=0)
        for index, (title, _) in enumerate(groups):
            accordion.set_title(index, title)
        accordion.set_title(len(groups), "Selected parameter values · complete audit")
        _replace_outputs(self.explanation,
                         HTML(f'<h3>{escape(card["name"])}</h3><p class="jf-help">{escape(card["mode"])} · Explanation: selected method {escape(self.method.value)} · Applied plots/exports: {escape(applied)}</p><p>{escape(card["summary"])}</p>'),
                         accordion)
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
        for control in [*self._controls, self.cusip, self.cusip_search, self.apply_button, self.compare_button]:
            control.disabled = busy
        self.export_button.disabled = busy or self.result is None
        self._help_changed()

    def _selected(self, frame):
        # Input: CUSIP=['A','B','A'], jf_row_id=[0,1,2], selected='A'.
        # Output: retained jf_row_id=[0,2], in source order.
        # Trick: Exact equality keeps string CUSIPs distinct from numeric IDs; original dataframe index labels are irrelevant.
        # CORE LOGIC: STEP 1 — Restrict display records to the exact selected instrument.
        if frame is self.result and self.applied_scope == "all":
            return frame.iloc[self.workspace.positions.get(self.cusip.value, np.array([], dtype=int))].copy()
        return frame.loc[frame[self.mapping["cusip_col"]].eq(self.cusip.value)].copy()

    def _render(self, record):
        # PLOTTING LOGIC: Native widget views use bundled Plotly assets, avoiding scripts inside sanitized HTML outputs.
        from .plots import trade_figure, diagnostic_figure, METHOD_LABELS
        result = record["result"]
        selected = result.iloc[self.workspace.positions.get(self.cusip.value, np.array([], dtype=int))].copy() if record["scope"] == "all" else result
        figure = trade_figure(selected, cusip_col=self.mapping["cusip_col"], spread_col=self.mapping["spread_col"],
                              unit=self.unit, max_gap=self.applied_config.max_gap if self.applied_config else "1D",
                              title=f'{self.cusip.value} · {METHOD_LABELS[self.applied_config.method] if self.applied_config else "Review"}')
        diagnostic = diagnostic_figure(selected, spread_col=self.mapping["spread_col"], unit=self.unit)
        return selected, _notebook_figure(figure), _notebook_figure(diagnostic), record["summary"], record["fitting"]

    def _publish(self, selected, figure, diagnostic, summary, fitting):
        # UI LOGIC: Replace output areas only after engine and plot construction both succeed.
        from IPython.display import HTML
        self.kpis.value = f'<p class="jf-help">Headline counts · selected bond {escape(str(self.cusip.value))}. Statistics and annotated exports · {"all bonds" if self.applied_scope == "all" else "selected bond"}.</p>' + _kpi_html(evaluation_summary(selected))
        contents = [HTML('<div class="jf-section-heading"><h4>Spread history &amp; review flags</h4><span class="jf-help">Actual trade timestamps · UTC</span></div>'),
                    figure, HTML('<p class="jf-help">Red crosses mark final outlier exclusions, including optional policy decisions. Gray open circles are unscored; amber open circles are explicitly retained uncertain and fit eligible; amber open diamonds mark unresolved transitions. Shading shows the method\'s statistical cutoff. Optional dashed limits show the separate support-trend distance cap. Closures, configured gap breaks and missing usable references interrupt both paths. Hover for exact evidence and reasons.</p>')]
        counts = figure.layout.meta or {}
        if counts.get("sampled"):
            contents.append(HTML(f'<p class="jf-help">Chart displays {counts["displayed_trades"]:,} of {counts["total_timed_trades"]:,} timed trades and {counts["displayed_outliers"]:,} of {counts["total_outliers"]:,} flags. Display sampling prioritizes review markers; filtering, statistics and annotated exports retain every reviewed trade.</p>'))
        _replace_outputs(self.chart, *contents)
        # UI LOGIC: Keep the liquidity chart immediately visible and progressively disclose the complete audit tables.
        audits = audit_tables(selected)
        support_columns = ["jf_time", "jf_status", "jf_n_reference", "jf_reference_span_minutes",
                           "jf_reference_density_per_hour", "jf_scale", "jf_gap_minutes", "jf_wall_gap_minutes", "jf_session_boundary"]
        overview_columns = [self.mapping["cusip_col"], "rows", "evaluated", "flagged", "fit_eligible", "flagged_rate", "coverage"]
        overview = summary[[name for name in overview_columns if name in summary]]
        full_summary = ('<details class="jf-note"><summary>Full bond statistics · all diagnostic columns</summary>' +
                        _table_html(summary) + '</details>')
        population = "All bonds" if self.applied_scope == "all" else "Selected bond"
        groups = [(f"{population} · evaluation and fitting coverage", '<h4>Bond overview · applied method</h4>' + _table_html(overview) + full_summary +
                   f'<h4>{population} · fitting coverage and retention</h4>' + _table_html(fitting) +
                   '<p class="jf-help">Hard fit follows final jf_fit_eligible. Optional Quantity screening can explicitly retain uncertain records; these remain unassessed in algorithm coverage. Soft fit includes positive suggested weights for ok, original outlier and retained-uncertain records. Optional policy exclusions have zero weight. Invalid, outside-session and solver-failure records stay excluded. Weight sum is not an effective sample size.</p>'),
                  ("Selected bond · daily support and decision reasons", '<h4>Observed daily support</h4>' + _table_html(audits["daily"]) +
                   '<h4>Flag and support reasons</h4>' + _table_html(audits["reasons"])),
                  ("Selected bond · liquidity and local uncertainty", _table_html(selected[[name for name in support_columns if name in selected]])),
                  ("Selected bond · complete trade audit", _table_html(selected))]
        if "policy_reasons" in audits:
            groups.insert(2, ("Selected bond · optional policy decisions", '<h4>Optional policy decisions</h4>' + _table_html(audits["policy_reasons"])))
        sections = []
        for _, contents in groups:
            section = self.w.Output(layout=self.w.Layout(width="100%"))
            _replace_outputs(section, HTML(contents))
            sections.append(section)
        accordion = self.w.Accordion(children=sections, selected_index=0)
        for index, (title, _) in enumerate(groups):
            accordion.set_title(index, title)
        _replace_outputs(self.statistics,
                         HTML('<div class="jf-section-heading"><h4>Liquidity &amp; local uncertainty</h4><span class="jf-help">Selected bond · applied method</span></div>'),
                         diagnostic, HTML('<p class="jf-help">Review support density and uncertainty before using flags for fitting. Expand a section below for exact counts and original-order trade diagnostics.</p>'),
                         accordion)
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
        """Apply settings to the explicit review population and retain a successful snapshot."""
        # UI LOGIC: Failed pending changes preserve the previous valid result and export settings.
        if self.busy:
            return self
        self._lock(True)
        self.status.value = '<div class="jf-status">Evaluating supplied trades…</div>'
        previous = self.applied_config, self.applied_scope, self.mapping, self.workspace
        try:
            config = self._configuration()
            schedule = self.session_schedule.copy(deep=True) if self.session_schedule is not None else None
            mapping = dict(self.mapping, quantity_col=self.quantity_col.value)
            workspace = self.workspace if mapping == self.mapping else ReviewWorkspace(self.data, mapping)
            record = workspace.review(config, scope=self.scope.value, bond=self.cusip.value, session_schedule=schedule)
            self.mapping, self.workspace = mapping, workspace
            self.applied_config, self.applied_scope = config, self.scope.value
            if record is not self._applied_record or self.cusip.value != self._view_bond:
                rendered = self._render(record)
                self._publish(*rendered)
            self.result, self._applied_state, self.comparison = record["result"], self._state(), None
            self._applied_record = record
            self._view_bond = self.cusip.value
            self.applied_schedule = schedule
            _replace_outputs(self.method_output)
            self._clear_comparison_widget()
            population = "All bonds (batch)" if self.applied_scope == "all" else f"Selected bond {self.cusip.value}"
            self.status.value = f'<div class="jf-status">{escape(population)} · applied {escape(config.method)} to {len(self.result):,} of {len(self.data):,} source rows. Exports contain this review population.</div>'
        except Exception as exc:
            self.applied_config, self.applied_scope, self.mapping, self.workspace = previous
            self.status.value = '<div class="jf-status jf-status-error">Could not apply settings: ' + escape(str(exc)) + '</div>'
        finally:
            self._lock(False)
            self._pending_changed()
        return self

    def _focus_changed(self, _=None):
        # UI LOGIC: First visits evaluate only this bond under the applied settings; revisits reuse bounded cached annotations.
        if self.result is None or self.busy:
            return
        self._lock(True)
        try:
            record = self._applied_record if self.applied_scope == "all" else self.workspace.review(self.applied_config, bond=self.cusip.value, session_schedule=self.applied_schedule)
            self._publish(*self._render(record))
            self.result, self._applied_record = record["result"], record
            self._view_bond = self.cusip.value
            population = "all bonds" if self.applied_scope == "all" else "selected bond"
            self.status.value = f'<div class="jf-status">Viewing {escape(str(self.cusip.value))} · applied {escape(self.applied_config.method)} · {population}: {len(self.result):,} of {len(self.data):,} source rows.</div>'
            self.comparison = None
            _replace_outputs(self.method_output)
            self._clear_comparison_widget()
        except Exception as exc:
            # UI LOGIC: A failed bond visit restores the published focus so later exports still match visible charts.
            if self._view_bond is not None:
                self.cusip.value = self._view_bond
            self.status.value = '<div class="jf-status jf-status-error">Could not review this bond: ' + escape(str(exc)) + '</div>'
        finally:
            self._lock(False)
            self._pending_changed()

    def _clear_comparison_widget(self):
        # UI LOGIC: Closed comparison models cannot retain stale applied settings in the notebook frontend.
        if self._comparison_widget is not None:
            self._comparison_widget.close()
            self._comparison_widget = None

    def compare(self, _=None):
        # UI LOGIC: Compare explicitly using the applied settings, even when new edits are pending.
        from IPython.display import HTML
        from .plots import comparison_figure
        if self.result is None or self.busy:
            self.status.value = '<div class="jf-status">Apply a filter before comparing methods.</div>'
            return
        self._lock(True)
        try:
            table = method_comparison(self.workspace.source_bond(self.cusip.value), self.applied_config, **self.mapping, session_schedule=self.applied_schedule)
            figure = _notebook_figure(comparison_figure(table))
            _replace_outputs(self.method_output,
                             HTML('<p class="jf-help">Same selected bond, same applied hyperparameters. Differences measure sensitivity, not accuracy. Centered and causal methods use different information sets.</p>'),
                             figure, HTML(_table_html(table)))
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
        from .plots import trade_figure
        if self.result is None:
            return None
        output = Path(self.export_path.value).expanduser() / datetime.now(timezone.utc).strftime("review_%Y%m%dT%H%M%S_%fZ")
        try:
            output.mkdir(parents=True, exist_ok=False)
            self.result.to_csv(output / "annotated_trades.csv", index=False)
            self._applied_record["summary"].to_csv(output / "bond_summary.csv", index=False)
            self._applied_record["fitting"].to_csv(output / "fitting_coverage.csv", index=False)
            for name, table in audit_tables(self._selected(self.result)).items():
                table.to_csv(output / f"selected_bond_{name}.csv", index=False)
            settings = dict(config=asdict(self.applied_config), mapping=self.mapping, unit=self.unit,
                            selected_cusip=str(self.cusip.value), rows=len(self.result), source_rows=len(self.data), review_scope=self.applied_scope, source="applied result",
                            calendar=self.result.attrs.get("jump_filter", {}).get("calendar"))
            if "jf_policy_reason" in self.result:
                settings["optional_policy_counts"] = self.result["jf_policy_reason"].value_counts(dropna=False).to_dict()
            if self.applied_schedule is not None:
                self.applied_schedule.to_csv(output / "session_schedule.csv", index=False)
            figure = trade_figure(self._selected(self.result), cusip_col=self.mapping["cusip_col"],
                                  spread_col=self.mapping["spread_col"], unit=self.unit, title=str(self.cusip.value),
                                  max_gap=self.applied_config.max_gap)
            settings["chart_display"] = figure.layout.meta
            (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
            figure.write_html(output / "selected_bond.html", include_plotlyjs=True)
            if self.comparison is not None:
                self.comparison.to_csv(output / "method_comparison.csv", index=False)
            self.status.value = '<div class="jf-status">Saved applied review: ' + escape(str(output.resolve())) + '</div>'
            return output
        except Exception as exc:
            self.status.value = '<div class="jf-status jf-status-error">Could not export: ' + escape(str(exc)) + '</div>'
            return None

    def _ipython_display_(self):
        # UI LOGIC: Show a single workbench view per notebook request; later cells can display it again.
        # Input: current request='cell-1',last displayed='cell-1' -> Output: no additional view.
        # Input: current request='cell-2',last displayed='cell-1' -> Output: one view,last displayed='cell-2'.
        # Trick: show_filter displays immediately and also returns self; IPython automatically displays a final expression. Without this guard, a bare or chained call creates duplicate views.
        from IPython import get_ipython
        from IPython.display import display
        kernel = getattr(get_ipython(), "kernel", None)
        parent = kernel.get_parent() if kernel is not None else {}
        request = parent.get("header", {}).get("msg_id")
        if request is not None and request == self._displayed_request:
            return
        display(self.widget)
        self._displayed_request = request


def show_filter(data, **kwargs):
    """Display the complete dashboard inside Jupyter; call .run() to apply settings.

    Return a FilterDashboard with native ipywidgets/Plotly views, including method
    explanations, mathematical formulas, statistics and applied-data exports.
    """
    # UI LOGIC: Return the workbench so callers can inspect its applied result and exports.
    panel = FilterDashboard(data, **kwargs)
    panel._ipython_display_()
    return panel
