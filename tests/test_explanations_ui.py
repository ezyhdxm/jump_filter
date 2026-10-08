"""Method explanations remain independent of applied fitting snapshots."""

# TEST LOGIC: Optional dashboards skip cleanly for minimal numerical installs.
from pathlib import Path
import pytest
import pandas as pd
from jump_filter import METHODS, FilterConfig, filter_trades, select_fit_data
from jump_filter.explanations import METHOD_EXPLANATIONS, PARAMETER_HELP, COMMON_STEPS, CLOCK_STEPS, FITTING_STEPS, math_display_blocks
from jump_filter.dashboard import fitting_statistics

# TEST LOGIC: Locate the application from this test rather than the shell working directory.
PROJECT = Path(__file__).resolve().parents[1]


def test_method_cards_cover_implemented_methods_and_relevant_controls():
    # TEST LOGIC: Every selectable implementation has its own exact equations, example, and relevant controls.
    assert set(METHOD_EXPLANATIONS) == set(METHODS)
    config_fields = set(FilterConfig.__dataclass_fields__)
    for method, card in METHOD_EXPLANATIONS.items():
        assert card["steps"] and card["example"] and card["tradeoffs"] and card["reference"]
        assert set(card["parameters"]) <= config_fields & set(PARAMETER_HELP)
        assert all(formula and prose for _, prose, formula in card["steps"])
    assert "threshold" not in METHOD_EXPLANATIONS["rolling_iqr"]["parameters"]
    assert "alpha" not in METHOD_EXPLANATIONS["consensus"]["parameters"]
    assert "trend_penalty" in METHOD_EXPLANATIONS["robust_trend"]["parameters"]


def test_math_pane_layout_preserves_equations_and_splits_independent_clauses():
    # TEST LOGIC: Layout-only changes retain mathematical tokens, source formulas, and case environments.
    import re
    steps = COMMON_STEPS + CLOCK_STEPS + FITTING_STEPS + tuple(step for card in METHOD_EXPLANATIONS.values() for step in card["steps"])
    def normalized(value):
        # TEST LOGIC: Remove only display alignment and spacing, comparing every original operator/name/number.
        value = value.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "").replace(r"\\&\qquad ", "").replace("&", "")
        return re.sub(r"\s+", "", value.replace(r"\quad", "").replace(r"\ ", ""))
    for _, _, formula in steps:
        blocks = math_display_blocks(formula)
        assert normalized("".join(blocks)) == normalized(formula)
    iqr_formula = METHOD_EXPLANATIONS["rolling_iqr"]["steps"][0][2]
    assert len(math_display_blocks(iqr_formula)) == 6
    trend_formula = METHOD_EXPLANATIONS["robust_trend"]["steps"][0][2]
    assert any(r"\begin{aligned}" in block and r"\\&\qquad +\lambda" in block for block in math_display_blocks(trend_formula))


def test_fitting_dashboard_counts_agree_with_public_fit_selection():
    # TEST LOGIC: Coverage exposes provisional and solver failures instead of calling them clean.
    frame = pd.DataFrame({"CUSIP": ["A"] * 6, "jf_row_id": range(6),
                          "jf_status": ["ok", "outlier", "insufficient_history", "provisional_jump", "solver_not_converged", "ambiguous_transition"],
                          "jf_is_outlier": [False, True, False, True, False, False], "jf_weight": [1., .2, 0., .1, 0., 0.],
                          "jf_fit_eligible": [True, False, False, False, False, False]})
    row = fitting_statistics(frame).iloc[0]
    assert row["coverage"] == pytest.approx(3 / 6)
    assert row["hard_retention"] == pytest.approx(1 / 6)
    assert row["hard_fit_rows"] == len(select_fit_data(frame, policy="hard")) == 1
    assert row["soft_fit_rows"] == len(select_fit_data(frame, policy="soft")) == 2
    assert row["soft_weight_sum"] == pytest.approx(1.2)
    assert row["solver_failures"] == 1
    assert row["ambiguous_transitions"] == 1


def test_iqr_card_numeric_example_matches_actual_fence():
    # TEST LOGIC: The documented six-reference numerical example uses the same raw units and linear quantiles.
    frame = pd.DataFrame({"CUSIP": ["A"] * 7, "time": pd.date_range("2026-10-01", periods=7, freq="min", tz="UTC"),
                          "spread": [99., 100., 100., 105., 101., 101., 102.]})
    cfg = FilterConfig(method="rolling_iqr", window=6, min_neighbors=6, iqr_multiplier=3.)
    row = filter_trades(frame, cfg).iloc[3]
    assert row["jf_baseline"] == 100.5
    assert row["jf_threshold"] == 3.5
    assert bool(row["jf_is_outlier"])
    frame.loc[3, "spread"] = 104.
    assert not filter_trades(frame, cfg).iloc[3]["jf_is_outlier"]


def test_browser_math_updates_before_apply_and_preserves_applied_review():
    # TEST LOGIC: Selection renders equations before any filter run and disables algorithm-irrelevant inputs.
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(PROJECT / "jump_filter/app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.get("latex")) >= 5
    assert app.slider(key="jf_slider_alpha").disabled
    method = next(box for box in app.selectbox if box.label == "Method")
    method.set_value("rolling_iqr").run()
    assert not app.exception
    assert app.slider(key="jf_slider_threshold").disabled
    assert not app.slider(key="jf_slider_iqr_multiplier").disabled
    assert any("selected method: rolling_iqr" in item.value for item in app.caption)
    assert "jf_review" not in app.session_state

    # TEST LOGIC: Explanations follow pending settings; charts/export snapshots still retain their applied method.
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    review = app.session_state["jf_review"]
    next(box for box in app.selectbox if box.label == "Method").set_value("robust_trend").run()
    assert not app.exception
    assert app.session_state["jf_review"] is review
    assert review["config"].method == "rolling_iqr"
    assert any("selected method: robust_trend" in item.value and "exports: rolling_iqr" in item.value for item in app.caption)
    assert not app.slider(key="jf_slider_trend_penalty").disabled
    assert app.slider(key="jf_slider_iqr_multiplier").disabled


def test_notebook_pending_help_and_calendar_preserve_applied_snapshot(monkeypatch):
    # TEST LOGIC: Suppress rich frontend output while observing notebook control-to-explanation bindings.
    pytest.importorskip("ipywidgets")
    pytest.importorskip("plotly")
    from jump_filter.dashboard import FilterDashboard
    explained = []
    monkeypatch.setattr(FilterDashboard, "_render_explanation", lambda panel: explained.append(panel.method.value))
    monkeypatch.setattr(FilterDashboard, "_publish", lambda *args: None)
    frame = pd.DataFrame({"CUSIP": ["A"] * 9,
                          "time": pd.date_range("2026-10-01T14:00:00Z", periods=9, freq="min"),
                          "spread": [100., 100., 100., 100., 130., 100., 100., 100., 100.]})
    panel = FilterDashboard(frame, config=FilterConfig(window=6, min_neighbors=6))
    assert explained[-1] == "consensus"
    assert panel._param_sliders["alpha"].disabled
    panel.run()
    assert panel.result is not None
    applied = panel.result
    panel.method.value = "local_piecewise"
    panel.time_basis.value = "trading"
    panel.holidays.value = "2026-10-12, 2026-11-26"
    assert explained[-1] == "local_piecewise"
    assert panel.result is applied and panel.applied_config.method == "consensus"
    assert panel._configuration().holidays == ("2026-10-12", "2026-11-26")
    assert panel._configuration().time_basis == "trading"
    assert panel._param_sliders["iqr_multiplier"].disabled
    assert not panel._param_sliders["reversion_tolerance"].disabled


def test_browser_authoritative_schedule_is_applied_and_audited(monkeypatch):
    # TEST LOGIC: Inject local CSV bytes at the upload boundary; no network or private trade data are used.
    st = pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    payload = b"open,close\n2026-09-14T08:00:00-04:00,2026-09-14T13:00:00-04:00\n"
    class Upload:
        # TEST LOGIC: Match the read-only getvalue interface of Streamlit's uploaded file.
        def getvalue(self):
            return payload
    original = st.file_uploader
    monkeypatch.setattr(st, "file_uploader", lambda label, *args, **kwargs: Upload() if label.startswith("Session schedule") else original(label, *args, **kwargs))
    app = AppTest.from_file(str(PROJECT / "jump_filter/app.py"), default_timeout=30).run()
    next(control for control in app.checkbox if control.label == "Use authoritative session schedule CSV").set_value(True).run()
    next(button for button in app.button if button.label == "Apply filter").click().run()
    assert not app.exception
    review = app.session_state["jf_review"]
    assert review["session_schedule"].iloc[0]["close"] == "2026-09-14T13:00:00-04:00"
    assert review["signature"]["schedule_fingerprint"]
    calendar = review["result"].attrs["jump_filter"]["calendar"]
    assert calendar["calendar_kind"] == "custom_schedule"
    assert calendar["session_count"] == 1
    assert calendar["schedule_fingerprint"]
    assert any(button.label == "Applied session schedule CSV" for button in app.get("download_button"))
