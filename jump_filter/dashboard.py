"""Optional notebook workbench and reusable dashboard statistics."""
# SETUP LOGIC: The notebook dependency is imported only when constructing a workbench.
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: These descriptions explain the algorithm rather than infer a trade's economic cause.
METHOD_HELP = {
    "hampel": "A centered leave-one-out median and MAD identifies unusual local spread levels.",
    "local_linear": "A robust time trend separates isolated deviations from a smooth spread drift.",
    "jump_reversion": "A large move followed by a prompt return suggests an isolated anomalous trade.",
    "causal_ewma": "A past-only robust baseline supports chronological use; persistent moves update the regime.",
    "consensus": "Reversal confirmation combines unusual local level or trend residual with a return to the earlier neighborhood.",
}

# UI LOGIC: CSS is scoped to this workbench and never changes another notebook's controls.
STYLE = """
.jf-workbench{background:#f2f6fa;color:#19364b;font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:18px;border-radius:17px;border:1px solid #dce5ee;max-width:1500px;width:100%;box-sizing:border-box}
.jf-workbench *{box-sizing:border-box}.jf-workbench .widget-vbox{gap:12px;min-width:0}.jf-workbench .jf-hero{padding:25px;background:linear-gradient(115deg,#15354c,#096977);color:white;border-radius:12px}.jf-workbench .jf-hero h2{color:white;margin:4px 0 9px;font-size:27px}.jf-workbench .jf-hero p{margin:0;color:#dfedf3}.jf-workbench .jf-eyebrow{color:#bde3e6;font-size:11px;letter-spacing:.15em;font-weight:700;text-transform:uppercase}
.jf-workbench .jf-card{background:white;border:1px solid #dce5ee;border-radius:11px;padding:17px;min-width:0}.jf-workbench .jf-help{color:#526a7f;font-size:13px}.jf-workbench .jf-row{display:flex;flex-flow:row wrap;gap:13px;align-items:flex-end}.jf-workbench .jf-control{flex:1 1 230px;min-width:0;max-width:100%;height:auto}.jf-workbench .jf-control:not(.widget-checkbox){display:flex;flex-direction:column;align-items:stretch}.jf-workbench .jf-control .widget-label{width:100%!important;text-align:left;white-space:normal;overflow:visible;height:auto;font-size:12px;font-weight:650;color:#304b61;margin-bottom:4px}.jf-workbench input:not([type=checkbox]),.jf-workbench select{border:1px solid #bfd0de;border-radius:6px;color:#19364b;padding:5px;min-height:34px}.jf-workbench .widget-button{height:38px;border-radius:7px;font-weight:650;padding:8px 16px}.jf-workbench .widget-button.mod-primary{background:#087f8c;color:white;border-color:#087f8c}.jf-workbench .jf-status{padding:11px 14px;border:1px solid #cbdce6;border-left:4px solid #087f8c;background:#edf5f8;border-radius:7px}.jf-workbench .jf-pending{color:#855d17;background:#fff5df;padding:7px 11px;border-radius:7px}.jf-workbench .jf-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:11px}.jf-workbench .jf-kpi{background:#f5fafb;border:1px solid #dbe6ec;padding:13px;border-radius:8px}.jf-workbench .jf-kpi strong{display:block;font-size:24px;color:#086f7d}.jf-workbench .jf-kpi span{font-size:12px;color:#526a7f}.jf-workbench .jf-table{overflow:auto;max-height:430px;border:1px solid #dce5ee;border-radius:8px}.jf-workbench table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;font-variant-numeric:tabular-nums}.jf-workbench th,.jf-workbench td{white-space:nowrap;padding:8px 10px!important;border:0!important;border-bottom:1px solid #e7edf3!important;text-align:right}.jf-workbench thead th{position:sticky;top:0;background:#eaf1f6;z-index:1;color:#29465c}.jf-workbench tbody tr:nth-child(even){background:#f6f9fc}.jf-workbench td:first-child,.jf-workbench th:first-child{text-align:left}.jf-workbench .widget-tab-contents{padding:13px;border:1px solid #dce5ee;background:white}.jf-workbench .jf-param{flex:1 1 270px;min-width:0;border:1px solid #e0e8ef;border-radius:8px;padding:10px}.jf-workbench .jf-param-title{font-size:12px;font-weight:650;color:#304b61}.jf-workbench .jf-param .widget-hslider{width:100%;min-width:0}.jf-workbench .jf-param .widget-text{width:100%}.jf-workbench .widget-html-content{max-width:100%}
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


def method_comparison(data, config, *, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC"):
    """Evaluate all methods on an explicitly selected input population."""
    # SETUP LOGIC: Public engine imports avoid a dashboard/engine initialization cycle.
    from . import METHODS, filter_trades
    records = []
    # Input: data={CUSIP:['A'],time:['2026-10-01T10:00Z'],spread:[100]}, config=FilterConfig().
    # Output: methods=['hampel','local_linear','jump_reversion','causal_ewma','consensus'];
    # each row has total=1,evaluated=0,outliers=0,flag_rate=NaN,invalid=0,
    # unsupported=1,provisional=0,regime_changes=0.
    # Trick: Every method sees the same rows and hyperparameters; no result is chosen by its flag rate.
    # CORE LOGIC: STEP 1 — Recompute each algorithm on the same review population.
    for method in METHODS:
        result = filter_trades(data, replace(config, method=method), cusip_col=cusip_col,
                               time_col=time_col, spread_col=spread_col, timezone=timezone)
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
    # UI LOGIC: Complete rows remain in exports; notebook display alone is bounded.
    shown = frame.head(limit)
    return (f'<p class="jf-help">Showing {len(shown):,} of {len(frame):,} rows.</p><div class="jf-table">' +
            shown.to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.5g}") + "</div>")


def _kpi_html(stats):
    # UI LOGIC: Words, exact counts and explicit denominators accompany status colors.
    rate = f'{stats["flag_rate"]:.1%}' if np.isfinite(stats["flag_rate"]) else "—"
    items = [("Supplied trades", f'{stats["total"]:,}'), ("Evaluated", f'{stats["evaluated"]:,}'),
             ("Flagged outliers", f'{stats["outliers"]:,}'), ("Flag rate / evaluated", rate),
             ("Insufficient support", f'{stats["unsupported"]:,}'), ("Invalid input", f'{stats["invalid"]:,}')]
    return '<div class="jf-kpis">' + ''.join(f'<div class="jf-kpi"><strong>{value}</strong><span>{label}</span></div>' for label, value in items) + '</div>'


class FilterDashboard:
    """Notebook controls with an explicit, atomically published applied result."""

    def __init__(self, data, *, config=None, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC", unit="bp"):
        # SETUP LOGIC: Copy caller data; keep optional notebook dependencies outside package imports.
        import ipywidgets as w
        from . import FilterConfig, METHODS
        from .plots import METHOD_LABELS
        self.w, self.data = w, data.copy()
        self.config = config or FilterConfig()
        self.mapping = dict(cusip_col=cusip_col, time_col=time_col, spread_col=spread_col, timezone=timezone)
        self.unit, self.result, self.applied_config = unit, None, None
        self.comparison, self._applied_state, self.busy = None, None, False
        # VALIDATION LOGIC: Fail early for a bad mapping, before showing controls that cannot run.
        if not all(column in data for column in [cusip_col, time_col, spread_col]):
            raise ValueError("CUSIP, time and spread mappings must name existing columns.")
        # UI LOGIC: Exact values drive selection; text labels are presentation only.
        values = list(data[cusip_col].dropna().drop_duplicates())
        self.cusip = w.Dropdown(description="Bond / CUSIP", options=[(str(value), value) for value in values])
        self.method = w.Dropdown(description="Detection method", options=[(METHOD_LABELS[name], name) for name in METHODS], value=self.config.method)
        self.help = w.HTML()
        self.pending, self.status, self.kpis = w.HTML(), w.HTML(), w.HTML()
        self.chart, self.statistics, self.method_output = w.Output(), w.Output(), w.Output()
        self.apply_button = w.Button(description="Apply filter", button_style="primary", icon="check")
        self.compare_button = w.Button(description="Compare methods for this bond", icon="bar-chart")
        self.export_button = w.Button(description="Export applied review", icon="download", disabled=True)
        self.export_path = w.Text(value="reports/jump_filter", description="Export folder")
        self.params, param_cards = {}, []
        # UI LOGIC: Every numerical slider has a synchronized input box for exact values.
        specs = [("window", "Maximum reference timestamps", 3, 201, 2, "int"),
                 ("min_neighbors", "Minimum reference timestamps", 2, 60, 1, "int"),
                 ("threshold", "Robust score threshold", .5, 12., .1, "float"),
                 ("abs_floor", f"Minimum deviation · {unit}", .001, 50., .1, "float"),
                 ("reversion_tolerance", "Return tolerance · robust scale", .1, 6., .1, "float"),
                 ("alpha", "EWMA learning rate", .01, 1., .01, "float"),
                 ("persistence", "Persistence confirmation cohorts", 2, 10, 1, "int")]
        for name, label, lower, upper, step, kind in specs:
            value = getattr(self.config, name)
            slider_cls, input_cls = (w.IntSlider, w.BoundedIntText) if kind == "int" else (w.FloatSlider, w.BoundedFloatText)
            slider = slider_cls(value=value, min=min(lower, value), max=max(upper, value), step=step, readout=False, continuous_update=False)
            number = input_cls(value=value, min=min(lower, value), max=max(upper, value), step=step)
            w.link((slider, "value"), (number, "value"))
            self.params[name] = number
            param_cards.append(w.VBox([w.HTML(f'<div class="jf-param-title">{escape(label)}</div>'), slider, number]).add_class("jf-param"))
        self.horizon = w.Text(value=str(self.config.horizon), description="Reference horizon (e.g. 3D)")
        self.max_gap = w.Text(value=str(self.config.max_gap), description="Session break gap (e.g. 1D)")
        self._controls = [self.method, self.horizon, self.max_gap, *self.params.values()]
        for control in [self.cusip, self.method, self.horizon, self.max_gap, self.export_path]:
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
        self.tabs = w.Tab(children=[self.chart, self.statistics, self.method_output])
        for i, name in enumerate(["Trade review", "Statistical dashboard", "Method comparison"]):
            self.tabs.set_title(i, name)
        hero = w.HTML('<div class="jf-hero"><div class="jf-eyebrow">Bond trade quality</div><h2>Jump Filter</h2><p>Review unusual trade spreads with robust local evidence. Preserve genuine spread moves and inspect the reason for every flag.</p></div>')
        note = w.HTML('<p class="jf-help">CUSIP, time and spread identify statistical anomalies. They cannot establish retail origin, distress, markup or commission. Centered methods use future trades; Causal robust EWMA uses past trades only. Baselines are diagnostic estimates.</p>')
        settings = w.Accordion(children=[w.VBox([w.Box(param_cards, layout=w.Layout(display="flex", flex_flow="row wrap", gap="12px")), self._row(self.horizon, self.max_gap)])], selected_index=None)
        settings.set_title(0, "Hyperparameters · sliders and exact inputs")
        self.widget = w.VBox([w.HTML("<style>" + STYLE + "</style>"), hero,
                              self._row(self.cusip, self.method), self.help, settings, note,
                              self._row(self.apply_button, self.compare_button), self.pending, self.status, self.kpis, self.tabs,
                              self._row(self.export_path, self.export_button)]).add_class("jf-workbench")
        self._help_changed()
        self._pending_changed()

    def _row(self, *children):
        # UI LOGIC: Wrapping respects notebook pane width rather than browser viewport width.
        return self.w.Box(list(children), layout=self.w.Layout(display="flex", flex_flow="row wrap", width="100%", gap="13px")).add_class("jf-row")

    def _state(self):
        # UI LOGIC: Only algorithm settings determine pending state; view focus and export path do not.
        return tuple(control.value for control in self._controls)

    def _configuration(self):
        # CONFIGURATION LOGIC: Validation belongs to FilterConfig and the engine rather than the widgets.
        from . import FilterConfig
        return FilterConfig(method=self.method.value, horizon=self.horizon.value, max_gap=self.max_gap.value,
                            **{name: control.value for name, control in self.params.items()})

    def _help_changed(self, _=None):
        # UI LOGIC: Literal method keys have curated explanatory text.
        self.help.value = '<p class="jf-help">' + escape(METHOD_HELP[self.method.value]) + '</p>'

    def _pending_changed(self, _=None):
        # UI LOGIC: Reverting edits restores the applied badge without running algorithms.
        label = "Choose settings and Apply." if self.result is None else (
            "Pending settings · Apply to update plots and exports." if self._state() != self._applied_state else "Applied · plots and exports match these settings.")
        self.pending.value = '<div class="jf-pending">' + label + '</div>'

    def _lock(self, busy):
        # UI LOGIC: Keep state fixed during computation and allow export only after a successful Apply.
        self.busy = busy
        for control in [*self._controls, self.cusip, self.apply_button, self.compare_button]:
            control.disabled = busy
        self.export_button.disabled = busy or self.result is None

    def _selected(self, frame):
        # Input: CUSIP=['A','B','A'], jf_row_id=[0,1,2], selected='A'.
        # Output: retained jf_row_id=[0,2], in source order.
        # Trick: Exact equality keeps string CUSIPs distinct from numeric IDs; original dataframe index labels are irrelevant.
        # CORE LOGIC: STEP 1 — Restrict display records to the exact selected instrument.
        return frame.loc[frame[self.mapping["cusip_col"]].eq(self.cusip.value)].copy()

    def _render(self, result):
        # PLOTTING LOGIC: Build complete figures before publishing successful applied state.
        from .plots import trade_figure, diagnostic_figure, METHOD_LABELS
        from . import summarize
        selected = self._selected(result)
        figure = trade_figure(selected, cusip_col=self.mapping["cusip_col"], spread_col=self.mapping["spread_col"],
                              unit=self.unit, max_gap=self.applied_config.max_gap if self.applied_config else "1D",
                              title=f'{self.cusip.value} · {METHOD_LABELS[self.applied_config.method] if self.applied_config else "Review"}')
        diagnostic = diagnostic_figure(selected, spread_col=self.mapping["spread_col"], unit=self.unit)
        summary = summarize(result, cusip_col=self.mapping["cusip_col"])
        return selected, figure, diagnostic, summary

    def _publish(self, selected, figure, diagnostic, summary):
        # UI LOGIC: Replace output areas only after engine and plot construction both succeed.
        from IPython.display import HTML, display
        self.kpis.value = _kpi_html(evaluation_summary(selected))
        with self.chart:
            self.chart.clear_output(wait=True)
            display(HTML('<p class="jf-help">All timed finite trades for this bond are drawn. Red crosses are statistical outliers; amber markers are level changes or provisional jumps.</p>'))
            display(HTML(figure.to_html(full_html=False, include_plotlyjs="inline")))
        with self.statistics:
            self.statistics.clear_output(wait=True)
            display(HTML('<h4>All bonds · applied method</h4>' + _table_html(summary)))
            display(HTML(diagnostic.to_html(full_html=False, include_plotlyjs="inline")))
            audits = audit_tables(selected)
            display(HTML('<h4>Selected bond · observed daily support</h4>' + _table_html(audits["daily"])))
            display(HTML('<h4>Selected bond · flag and support reasons</h4>' + _table_html(audits["reasons"])))
            display(HTML('<h4>Selected bond · trade audit</h4>' + _table_html(selected)))

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
            result = filter_trades(self.data, config, **self.mapping)
            self.applied_config = config
            rendered = self._render(result)
            self._publish(*rendered)
            self.result, self._applied_state, self.comparison = result, self._state(), None
            self.method_output.clear_output()
            self.status.value = f'<div class="jf-status">Applied {escape(config.method)} to {len(result):,} supplied rows. Charts focus on {escape(str(self.cusip.value))}.</div>'
        except Exception as exc:
            self.applied_config = previous
            self.status.value = '<div class="jf-status">Could not apply settings: ' + escape(str(exc)) + '</div>'
        finally:
            self._lock(False)
            self._pending_changed()
        return self

    def _focus_changed(self, _=None):
        # UI LOGIC: Reuse applied annotations when browsing bonds; no algorithm is rerun.
        if self.result is None or self.busy:
            return
        self._publish(*self._render(self.result))
        self.comparison = None
        self.method_output.clear_output()

    def compare(self, _=None):
        # UI LOGIC: Compare explicitly using the applied settings, even when new edits are pending.
        from IPython.display import HTML, display
        from .plots import comparison_figure
        if self.result is None or self.busy:
            self.status.value = '<div class="jf-status">Apply a filter before comparing methods.</div>'
            return
        self._lock(True)
        try:
            table = method_comparison(self._selected(self.data), self.applied_config, **self.mapping)
            figure = comparison_figure(table)
            with self.method_output:
                self.method_output.clear_output(wait=True)
                display(HTML('<p class="jf-help">Same selected bond, same applied hyperparameters. Differences measure sensitivity, not accuracy. Centered and causal methods use different information sets.</p>'))
                display(HTML(figure.to_html(full_html=False, include_plotlyjs="inline")))
                display(HTML(_table_html(table)))
            self.comparison = table
            self.tabs.selected_index = 2
        except Exception as exc:
            self.status.value = '<div class="jf-status">Could not compare methods: ' + escape(str(exc)) + '</div>'
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
            for name, table in audit_tables(self._selected(self.result)).items():
                table.to_csv(output / f"selected_bond_{name}.csv", index=False)
            settings = dict(config=asdict(self.applied_config), mapping=self.mapping, unit=self.unit,
                            selected_cusip=str(self.cusip.value), rows=len(self.result), source="applied result")
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
            self.status.value = '<div class="jf-status">Could not export: ' + escape(str(exc)) + '</div>'
            return None

    def _ipython_display_(self):
        # UI LOGIC: Displaying a workbench does not rerun filtering or change caller data.
        from IPython.display import display
        display(self.widget)


def show_filter(data, **kwargs):
    """Construct and display a notebook dashboard; call .run() to apply settings."""
    # UI LOGIC: Return the workbench so callers can inspect its applied result and exports.
    from IPython.display import display
    panel = FilterDashboard(data, **kwargs)
    display(panel.widget)
    return panel
