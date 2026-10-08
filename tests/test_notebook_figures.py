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
from jump_filter.plots import comparison_figure, diagnostic_figure, trade_figure


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
        assert all(isinstance(value, str) for value in unscored.x)
        assert pd.to_datetime(unscored.x, utc=True).tolist() == frame["time"].tolist()
        assert all(row[4] is None for row in unscored.customdata)
        assert annotated["jf_score"].isna().all()
        assert annotated["jf_status"].eq("insufficient_history").all()
    finally:
        widget.close()


def test_native_datetime_array_preserves_missing_coordinate_gaps_and_binary_nan_values():
    # TEST LOGIC: pandas 3 can expose nanosecond datetime arrays; native .tolist() would emit oversized integer coordinates.
    times = np.array(["2026-10-01T14:00:00", "NaT", "2026-10-02T14:00:00"], dtype="datetime64[ns]")
    values = np.array([100., np.nan, 101.])
    source = go.Figure(go.Scatter(x=times, y=values, connectgaps=False))
    widget = _notebook_figure(source)
    try:
        assert list(widget.data[0].x) == ["2026-10-01T14:00:00.000000000", None, "2026-10-02T14:00:00.000000000"]
        assert isinstance(widget.data[0].y, np.ndarray)
        np.testing.assert_equal(widget.data[0].y, values)
        assert np.isnat(source.data[0].x[1])
        assert not widget.data[0].connectgaps
        state, paths, buffers = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
        assert paths and buffers
    finally:
        widget.close()


def _typed_row_deltas(matrix):
    # TEST LOGIC: Plotly's frontend reports typed matrix rows as descriptors; its Python decoder leaves these as dictionaries.
    # Input: float64 matrix [[10., 11.], [20., 21.]].
    # Output: two row descriptors, each dtype='float64', shape=[2], and a buffer containing the corresponding two values.
    return [dict(dtype=str(row.dtype), shape=[len(row)], value=memoryview(np.ascontiguousarray(row)))
            for row in matrix]


def test_native_numeric_matrix_accepts_frontend_trace_deltas_without_type_conflict():
    # TEST LOGIC: The JSON roundtrip changed the original customdata into a bdata dictionary and raised AssertionError here.
    matrix = np.array([[10., 11.], [20., 21.]])
    source = go.Figure(go.Bar(x=[1, 2], y=[3., 4.], customdata=matrix))
    # TEST LOGIC: The legacy JSON representation becomes typed frontend rows, whose returned descriptors trigger the observed assertion.
    legacy = go.FigureWidget(json.loads(source.to_json(remove_uids=False)))
    try:
        if isinstance(legacy.data[0].customdata, dict):
            legacy_message = dict(trace_deltas=[dict(uid=legacy.data[0].uid, customdata=_typed_row_deltas(matrix))],
                                  trace_edit_id=legacy._last_trace_edit_id)
            with pytest.raises(AssertionError):
                legacy._js2py_traceDeltas = legacy_message
    finally:
        legacy.close()
    widget = _notebook_figure(source)
    try:
        trace = widget.data[0]
        assert isinstance(trace.customdata, np.ndarray)
        assert trace.customdata.shape == (2, 2)
        # TEST LOGIC: Native matrix transport sends ordinary nested lists, so frontend deltas preserve that same representation.
        message = dict(trace_deltas=[dict(uid=trace.uid, customdata=matrix.tolist())],
                       trace_edit_id=widget._last_trace_edit_id)
        widget._js2py_traceDeltas = message
        assert widget._js2py_traceDeltas is None
        np.testing.assert_equal(trace.customdata, matrix)
        np.testing.assert_equal(source.data[0].customdata, matrix)
        state, _, _ = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
    finally:
        widget.close()


def test_diagnostic_histogram_accepts_frontend_matrix_deltas_and_keeps_binary_coordinates():
    # TEST LOGIC: Applying the filter/CUSIP callback renders these real histogram matrices even when Statistics is hidden.
    annotated = pd.DataFrame(dict(jf_residual=[0., 1., 20.], jf_is_outlier=[False, False, True],
                                  jf_status=["ok", "ok", "outlier"]))
    source = diagnostic_figure(annotated)
    widget = _notebook_figure(source)
    try:
        trace = widget.data[0]
        assert isinstance(trace.x, np.ndarray)
        assert isinstance(trace.customdata, np.ndarray)
        matrix = np.asarray(source.data[0].customdata)
        widget._js2py_traceDeltas = dict(trace_deltas=[dict(uid=trace.uid, customdata=matrix.tolist())],
                                        trace_edit_id=widget._last_trace_edit_id)
        np.testing.assert_equal(trace.customdata, matrix)
        assert sum(widget.data[0].y) + sum(widget.data[1].y) == 3
        state, paths, buffers = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
        assert paths and buffers
    finally:
        widget.close()


def test_trade_chart_keeps_source_outliers_and_trading_session_breaks():
    # TEST LOGIC: A repeated dataframe index must not move a flagged trade or join two inactive sessions.
    times = list(pd.date_range("2026-10-01T14:00Z", periods=7, freq="min"))
    times += list(pd.date_range("2026-10-02T14:00Z", periods=7, freq="min"))
    frame = pd.DataFrame({"CUSIP": ["A"] * 14, "time": times,
                          "spread": [100.] * 5 + [160.] + [100.] * 8}, index=[4] * 14)
    annotated = filter_trades(frame, FilterConfig(method="hampel", window=6, min_neighbors=4, time_basis="trading"))
    original = annotated.copy(deep=True)
    figure = trade_figure(annotated)
    flagged = next(trace for trace in figure.data if trace.name == "Outlier")
    baseline = next(trace for trace in figure.data if trace.name == "Baseline")
    residual = next(trace for trace in figure.data if trace.name == "Residual")
    assert pd.to_datetime(flagged.x, utc=True).tolist() == [times[5]]
    assert list(flagged.y) == [160.]
    assert flagged.customdata[0][0] == "5"
    assert list(baseline.x) == times[:7] + [None] + times[7:]
    assert residual.marker.symbol[5] == "x"
    np.testing.assert_allclose(residual.y, annotated["jf_residual"], equal_nan=True)
    pd.testing.assert_frame_equal(annotated, original)


def test_comparison_preserves_zero_counts_and_explicit_evaluation_denominator():
    # TEST LOGIC: An unsupported bond still displays every requested method and its actual supplied denominator.
    table = pd.DataFrame({"method": ["hampel", "causal_ewma"], "total": [2, 2],
                          "evaluated": [0, 0], "outliers": [0, 0], "flag_rate": [np.nan, np.nan]})
    figure = comparison_figure(table)
    widget = _notebook_figure(figure)
    try:
        bar = figure.data[0]
        assert list(bar.x) == [0, 0]
        assert len(bar.y) == len(table)
        assert list(bar.text) == [0, 0]
        np.testing.assert_allclose(bar.customdata, [[np.nan, 0., 2.], [np.nan, 0., 2.]], equal_nan=True)
        assert "Evaluation coverage" in bar.hovertemplate
        state, _, _ = _remove_buffers(widget.get_state())
        json.dumps(state, allow_nan=False)
        assert table["flag_rate"].isna().all()
    finally:
        widget.close()
