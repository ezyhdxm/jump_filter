"""Native notebook figures use strict JSON without changing source diagnostics."""

# TEST LOGIC: Notebook-only dependencies skip cleanly for a numerical engine installation.
import json
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("anywidget")
go = pytest.importorskip("plotly.graph_objects")
from ipywidgets.widgets.widget import _remove_buffers
from jump_filter import FilterConfig, filter_trades
from jump_filter.dashboard import _notebook_figure
from jump_filter.plots import trade_figure


def test_native_figure_comm_encodes_missing_hover_values_as_null():
    # TEST LOGIC: Reproduce the object-array NaN/Infinity values rejected by strict JSON widget transports.
    source = go.Figure(go.Scatter(x=[1, 2], y=[100., np.nan],
                                 customdata=np.array([[np.nan, 1], [np.inf, 2]], dtype=object)))
    widget = _notebook_figure(source)
    try:
        state, _, _ = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
        assert widget.data[0].customdata == ([None, 1], [None, 2])
        assert np.isnan(source.data[0].y[1])
        assert np.isnan(source.data[0].customdata[0, 0])
        assert np.isposinf(source.data[0].customdata[1, 0])
    finally:
        widget.close()


def test_unscored_bond_chart_remains_json_compliant_and_preserves_nan_annotations():
    # TEST LOGIC: Unsupported real chart rows expose missing scores/gaps without manufacturing zero-valued evidence.
    frame = pd.DataFrame({"CUSIP": ["A"] * 4,
                          "time": pd.date_range("2026-10-01T14:00:00Z", periods=4, freq="min"),
                          "spread": [100., 101., 100., 100.]})
    annotated = filter_trades(frame, FilterConfig(method="hampel", window=6, min_neighbors=6))
    widget = _notebook_figure(trade_figure(annotated))
    try:
        state, _, _ = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
        unscored = next(trace for trace in widget.data if trace.name == "Unscored")
        assert len(unscored.x) == 4
        assert all(row[4] is None for row in unscored.customdata)
        assert annotated["jf_score"].isna().all()
        assert annotated["jf_status"].eq("insufficient_history").all()
    finally:
        widget.close()
