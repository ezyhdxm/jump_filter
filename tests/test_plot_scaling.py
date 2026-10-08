"""Bounded drawing keeps actual source positions and full diagnostic totals."""

# TEST LOGIC: Optional chart dependencies do not burden a numerical-only installation.
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("plotly")
from jump_filter import FilterConfig, filter_trades
from jump_filter.plots import _display_positions, _extrema_positions, _residual_histograms, diagnostic_figure, trade_figure


def _two_session_review():
    # TEST LOGIC: Duplicate labels and a closure shorter than max_gap exercise positional drawing and preserved session breaks.
    times = list(pd.date_range("2026-10-01T14:00Z", periods=10, freq="min"))
    times += list(pd.date_range("2026-10-02T14:00Z", periods=10, freq="min"))
    frame = pd.DataFrame({"CUSIP": ["A"] * 20, "time": times,
                          "spread": [100.] * 5 + [160.] + [100.] * 9 + [160.] + [100.] * 4}, index=[7] * 20)
    return filter_trades(frame, FilterConfig(method="hampel", window=6, min_neighbors=4,
                                             time_basis="trading"), backend="python")


def test_bounded_trade_chart_preserves_flags_positions_and_skipped_closures():
    # TEST LOGIC: Sparse display evidence must not alter engine annotations or connect across omitted session boundary records.
    annotated = _two_session_review()
    original = annotated.copy(deep=True)
    figure = trade_figure(annotated, max_points=8)
    flags = next(trace for trace in figure.data if trace.name == "Outlier")
    baseline = next(trace for trace in figure.data if trace.name == "Baseline")
    assert [int(row[0]) for row in flags.customdata] == [5, 15]
    assert list(flags.y) == [160., 160.]
    assert list(baseline.x).count(None) == 1
    assert figure.layout.meta["displayed_trades"] <= 8
    assert figure.layout.meta["total_timed_trades"] == 20
    assert figure.layout.meta["displayed_outliers"] == figure.layout.meta["total_outliers"] == 2
    assert figure.layout.meta["sampled"]
    pd.testing.assert_frame_equal(annotated, original)


def test_marker_overflow_is_bounded_and_disclosed_with_source_endpoints():
    # TEST LOGIC: Even an all-flagged pathological population cannot overflow the browser drawing budget.
    annotated = _two_session_review()
    annotated["jf_is_outlier"], annotated["jf_status"] = True, "outlier"
    figure = trade_figure(annotated, max_points=6)
    flags = next(trace for trace in figure.data if trace.name == "Outlier")
    positions = [int(row[0]) for row in flags.customdata]
    assert len(positions) <= 6
    assert positions[0] == 0 and positions[-1] == 19
    assert figure.layout.meta["total_outliers"] == 20
    assert figure.layout.meta["displayed_outliers"] == len(positions)
    assert figure.layout.meta["sampled"]
    assert np.array_equal(_extrema_positions(np.array([0, 9, 1, 8, 2, 7]), 4), [0, 1, 2, 5])


def test_histograms_aggregate_all_finite_residuals_and_keep_extreme_tails():
    # TEST LOGIC: Fixed-size payloads retain complete counts rather than trimming or sampling diagnostic distributions.
    values = np.array([0., 0., 1., 1_000_000., np.nan, .4])
    flags = np.array([False, True, False, True, False, False])
    edges, unflagged, flagged = _residual_histograms(values, flags, bins=2)
    np.testing.assert_equal(edges, [0., 500_000., 1_000_000.])
    np.testing.assert_equal(unflagged, [3, 0])
    np.testing.assert_equal(flagged, [1, 1])
    figure = diagnostic_figure(pd.DataFrame({"jf_residual": values, "jf_is_outlier": flags,
                                            "jf_status": ["ok", "outlier", "ok", "outlier", "invalid_input", "ok"]}))
    assert len(figure.data[0].x) == len(figure.data[1].x) == 45
    assert sum(figure.data[0].y) + sum(figure.data[1].y) == 5


def test_nearly_full_interior_marker_budget_preserves_unflagged_period_endpoints():
    # TEST LOGIC: Three interior markers numerically fit a four-point budget, but their union with both period endpoints does not.
    rows = pd.DataFrame({"jf_is_outlier": [False, True, True, True, False, False],
                         "jf_regime_change": [False] * 6, "jf_status": ["ok", "outlier", "outlier", "outlier", "ok", "ok"]})
    positions = _display_positions(rows, np.array([0., 9., 20., 8., 2., 7.]), max_points=4)
    np.testing.assert_equal(positions, [0, 1, 3, 5])
