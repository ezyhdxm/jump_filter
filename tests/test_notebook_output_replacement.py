"""Notebook sections replace their state without relying on frontend output capture."""

# TEST LOGIC: Numerical-only installations do not require the native notebook stack.
import pandas as pd
import pytest
pytest.importorskip("anywidget")
pytest.importorskip("plotly.graph_objects")
w = pytest.importorskip("ipywidgets")
from IPython.display import HTML
from jump_filter import FilterConfig
from jump_filter.dashboard import FilterDashboard, _replace_outputs

# TEST LOGIC: These MIME records are also consumed by Notebook 7 and JupyterLab 4.
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"


def _widget_ids(output):
    # TEST LOGIC: Inspect published model references rather than Python figure trace counts.
    return [item["data"][WIDGET_VIEW]["model_id"] for item in output.outputs
            if WIDGET_VIEW in item.get("data", {})]


def _html(output):
    # TEST LOGIC: Keep individual HTML records separate so repeated headings cannot pass unnoticed.
    return [item["data"]["text/html"] for item in output.outputs
            if "text/html" in item.get("data", {})]


def _assert_published_once(panel):
    # TEST LOGIC: The same model may have multiple notebook views; its serialized section still contains one chart.
    assert sum(value.count("Spread history &amp; review flags") for value in _html(panel.chart)) == 1
    assert _widget_ids(panel.chart) == [panel._figure_widgets[0].model_id]
    assert sum(value.count("Liquidity &amp; local uncertainty") for value in _html(panel.statistics)) == 1
    assert _widget_ids(panel.statistics) == [panel._figure_widgets[1].model_id, panel._statistics_widgets[-1].model_id]
    assert _widget_ids(panel.explanation) == [panel._explanation_widgets[-1].model_id]
    outputs = [panel.chart, panel.statistics, panel.method_output, panel.explanation]
    outputs += [widget for widget in (*panel._statistics_widgets, *panel._explanation_widgets)
                if isinstance(widget, w.Output)]
    assert all(output.msg_id == "" for output in outputs)
    assert all(item["output_type"] == "display_data" for output in outputs for item in output.outputs)
    assert any("text/latex" in item.get("data", {}) for output in outputs for item in output.outputs)


def test_atomic_replacement_removes_duplicate_records_and_capture_id():
    # TEST LOGIC: Reproduce accumulated frontend state and an active capture hook without requiring a browser.
    output = w.Output()
    duplicate = {"output_type": "display_data", "data": {"text/html": "<h4>Old heading</h4>"}, "metadata": {}}
    output.outputs, output.msg_id = (duplicate,) * 3, "stale-callback-message"
    try:
        _replace_outputs(output, HTML("<h4>Current heading</h4>"))
        assert output.msg_id == ""
        assert len(output.outputs) == 1
        assert _html(output) == ["<h4>Current heading</h4>"]
        _replace_outputs(output)
        assert output.outputs == ()
    finally:
        output.close()


def test_dashboard_callbacks_publish_once_without_display_capture(monkeypatch):
    # TEST LOGIC: Emitting rich displays inside callbacks would reintroduce frontend-dependent capture.
    def unexpected_display(*args, **kwargs):
        # TEST LOGIC: show_filter intentionally displays the root UI; this test directly constructs its model instead.
        pytest.fail("Dashboard callbacks must replace output state without emitting rich display messages.")
    monkeypatch.setattr("IPython.display.display", unexpected_display)
    source = pd.DataFrame({"CUSIP": ["A", "B"] * 9,
                           "time": pd.date_range("2026-10-01T14:00:00Z", periods=18, freq="min"),
                           "spread": [100., 200.] * 5 + [150., 200.] + [100., 200.] * 3})
    panel = FilterDashboard(source, config=FilterConfig(method="hampel", window=6, min_neighbors=2))
    try:
        assert all(len(output.outputs) == 1 for output in (panel.chart, panel.statistics, panel.method_output))
        panel.run()
        assert panel.result is not None, panel.status.value
        _assert_published_once(panel)
        previous_ids = {figure.model_id for figure in panel._figure_widgets}
        for bond in ("B", "A", "B"):
            panel.cusip.value = bond
            _assert_published_once(panel)
            assert not previous_ids.intersection(_widget_ids(panel.chart) + _widget_ids(panel.statistics))
            previous_ids = {figure.model_id for figure in panel._figure_widgets}
        panel.method.value = "rolling_iqr"
        panel.run()
        assert panel.applied_config.method == "rolling_iqr", panel.status.value
        _assert_published_once(panel)
        panel.compare()
        assert panel.comparison is not None, panel.status.value
        assert _widget_ids(panel.method_output) == [panel._comparison_widget.model_id]
        assert len(panel.method_output.outputs) == 3
        _assert_published_once(panel)
        panel.cusip.value = "A"
        assert panel.method_output.outputs == ()
        _assert_published_once(panel)
    finally:
        # TEST LOGIC: Release every live plot and section model constructed by the callback sequence.
        panel._clear_comparison_widget()
        for widget in (*panel._figure_widgets, *panel._statistics_widgets, *panel._explanation_widgets,
                       panel.chart, panel.statistics, panel.method_output, panel.explanation, panel.widget):
            widget.close()
