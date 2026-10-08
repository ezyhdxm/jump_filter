"""Optional policy controls preserve applied mappings, coverage and fitting audits."""
# TEST LOGIC: Small synthetic records exercise UI state without private trade data or a browser process.
from pathlib import Path
import json
import pandas as pd
import pytest
from jump_filter import FilterConfig, filter_trades, select_fit_data
from jump_filter.dashboard import evaluation_summary, fitting_statistics, audit_tables


def _source():
    # TEST LOGIC: Two differently named Quantity columns distinguish pending mapping edits from applied decisions.
    return pd.DataFrame({"CUSIP": ["A"] * 9 + ["B"] * 9,
                         "time": list(pd.date_range("2026-10-01T14:00:00Z", periods=9, freq="min")) * 2,
                         "spread": [100., 100., 100., 100., 130., 100., 100., 100., 100.] * 2,
                         "old_size": [500000.] * 18, "new_size": [2000000.] * 18})


def test_policy_statistics_keep_algorithm_coverage_separate_from_final_retention():
    # TEST LOGIC: Quantity can retain unsupported edge trades but cannot upgrade them to assessed algorithm evidence.
    result = filter_trades(_source(), FilterConfig(method="local_piecewise", window=6, min_neighbors=6, quantity_rule=True), quantity_col="new_size")
    stats = evaluation_summary(result)
    assert stats["retained_uncertain"] > 0
    assert stats["evaluated"] == result["jf_algorithm_status"].isin(["ok", "outlier", "provisional_jump"]).sum()
    assert stats["fit_eligible"] == len(select_fit_data(result, policy="hard"))
    fitting = fitting_statistics(result)
    assert fitting["hard_fit_rows"].sum() == len(select_fit_data(result, policy="hard"))
    assert fitting["soft_fit_rows"].sum() == len(select_fit_data(result, policy="soft"))
    assert fitting["retained_uncertain"].sum() == stats["retained_uncertain"]
    assert "policy_reasons" in audit_tables(result)
    assert audit_tables(result)["daily"]["evaluated"].sum() == stats["evaluated"]


def test_notebook_optional_controls_apply_mapping_and_export_only_successful_snapshot(monkeypatch, tmp_path):
    # TEST LOGIC: Rich output is covered elsewhere; this test follows mappings through Apply, bond focus and export.
    pytest.importorskip("ipywidgets")
    from jump_filter.dashboard import FilterDashboard
    monkeypatch.setattr(FilterDashboard, "_render_explanation", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_empty_outputs", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_render", lambda panel, record: ())
    monkeypatch.setattr(FilterDashboard, "_publish", lambda *args: None)
    panel = FilterDashboard(_source(), config=FilterConfig(method="local_piecewise", window=6, min_neighbors=6), quantity_col="old_size")
    assert not panel.quantity_rule.value and not panel.max_deviation_rule.value
    assert panel.quantity_multiplier.value == 1
    assert panel.policy_params["quantity_threshold"].value == 1000000
    assert panel.policy_params["max_deviation_bps"].value == 10
    panel.run()
    initial = panel.result
    panel.quantity_rule.value = True
    assert not panel.spread_units_per_bp.disabled
    panel.quantity_col.value = "new_size"
    assert panel.result is initial and panel.mapping["quantity_col"] == "old_size"
    panel.policy_params["max_deviation_bps"].value = 12.3
    assert panel._policy_sliders["max_deviation_bps"].value == 12.3
    panel._policy_sliders["quantity_threshold"].value = 1500000
    assert panel.policy_params["quantity_threshold"].value == 1500000
    panel.run()
    assert panel.mapping["quantity_col"] == "new_size"
    assert panel.result["jf_policy_retained_uncertain"].any()
    panel.quantity_col.value = "old_size"
    panel.cusip.value = "B"
    assert panel.mapping["quantity_col"] == "new_size"
    assert panel.result["jf_quantity"].eq(2000000).all()
    assert "Pending settings" in panel.pending.value
    panel.export_path.value = str(tmp_path)
    output = panel.export()
    settings = json.loads((output / "settings.json").read_text())
    assert settings["mapping"]["quantity_col"] == "new_size"
    assert settings["config"]["quantity_threshold"] == 1500000
    assert (output / "selected_bond_policy_reasons.csv").exists()
    applied = panel.result
    panel.quantity_col.value = None
    panel.run()
    assert panel.result is applied and panel.mapping["quantity_col"] == "new_size"
    assert "quantity_col" in panel.status.value


def test_streamlit_optional_policy_controls_and_failed_mapping_preserve_applied_review():
    # TEST LOGIC: Both synchronized number controls and explicit unit conversion participate in the Apply snapshot.
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    project = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project / "jump_filter" / "app.py"), default_timeout=30).run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Quantity column · optional").set_value("Quantity").run()
    next(box for box in app.checkbox if box.label == "Enable Quantity-sensitive screening").check().run()
    assert not next(box for box in app.selectbox if box.label == "Spread numeric units").disabled
    next(box for box in app.checkbox if box.label == "Enable maximum support-trend deviation").check().run()
    app.number_input(key="jf_number_max_deviation_bps").set_value(12.).run()
    assert app.slider(key="jf_slider_max_deviation_bps").value == 12
    app.slider(key="jf_slider_quantity_threshold").set_value(1500000.).run()
    assert app.number_input(key="jf_number_quantity_threshold").value == 1500000
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    review = app.session_state["jf_review"]
    assert review["mapping"]["quantity_col"] == "Quantity"
    assert review["config"].quantity_rule and review["config"].max_deviation_rule
    assert review["config"].quantity_multiplier == review["config"].spread_units_per_bp == 1
    assert review["config"].quantity_threshold == 1500000
    assert "jf_policy_reason" in review["result"]
    next(box for box in app.selectbox if box.label == "Quantity column · optional").set_value(None).run()
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    assert app.session_state["jf_review"] is review
    assert any("quantity_col" in error.value for error in app.error)


def test_notebook_policy_slider_grids_and_accessible_inputs_preserve_exact_apply_values(monkeypatch):
    # TEST LOGIC: Native range controls snap to their step grid; default 1MM must lie on that grid without narrowing valid typed values.
    pytest.importorskip("ipywidgets")
    from jump_filter.dashboard import FilterDashboard
    monkeypatch.setattr(FilterDashboard, "_render_explanation", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_empty_outputs", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_render", lambda panel, record: ())
    monkeypatch.setattr(FilterDashboard, "_publish", lambda *args: None)
    panel = FilterDashboard(_source(), config=FilterConfig(method="local_piecewise", window=6, min_neighbors=6, quantity_rule=True, max_deviation_rule=True), quantity_col="new_size")
    amount, cap = panel.policy_params["quantity_threshold"], panel.policy_params["max_deviation_bps"]
    amount_slider, cap_slider = panel._policy_sliders["quantity_threshold"], panel._policy_sliders["max_deviation_bps"]
    assert amount.value == amount_slider.value == 1000000
    assert amount_slider.min == 1 and amount_slider.step == 1
    assert (amount_slider.value - amount_slider.min) / amount_slider.step == 999999
    assert (cap_slider.value - cap_slider.min) / cap_slider.step == pytest.approx(99)
    assert amount.description == "Exact value · Small-trade threshold · notional amount"
    assert cap.description == "Exact value · Maximum trend deviation · bp"
    assert amount_slider.description and cap_slider.description and amount.tooltip and cap.tooltip

    # TEST LOGIC: Exact fractional amounts retain FloatText precision, and Apply uses the exact box rather than a frontend thumb position.
    for value in [1., 12500.25, 1000000.]:
        amount.value = value
        assert amount_slider.value == value
        panel.run()
        assert panel.applied_config.quantity_threshold == value
        assert amount.value == value
    cap.value = 12.
    assert cap_slider.value == 12
    cap_slider.value = 13.
    assert cap.value == 13
    panel.run()
    assert panel.applied_config.max_deviation_bps == 13
