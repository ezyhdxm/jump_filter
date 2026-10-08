"""Cutoff polygons respect closures and unsupported references after display sampling."""

# TEST LOGIC: Optional chart dependencies skip cleanly for numerical-only installations.
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("plotly")
from jump_filter.plots import AMBER, RED, _band_polygons, diagnostic_figure, trade_figure


def _review(times):
    # TEST LOGIC: A deterministic supported review isolates rendering from algorithm-dependent boundary support.
    count = len(times)
    return pd.DataFrame({"CUSIP": ["A"] * count, "spread": [100.] * count,
                         "jf_time": pd.to_datetime(times, utc=True), "jf_row_id": np.arange(count),
                         "jf_baseline": [100.] * count, "jf_threshold": [1.] * count,
                         "jf_scale": [1.] * count, "jf_residual": [0.] * count,
                         "jf_score": [0.] * count, "jf_n_reference": [6] * count,
                         "jf_regime_change": [False] * count, "jf_is_outlier": [False] * count,
                         "jf_status": ["ok"] * count, "jf_reason": ["within_threshold"] * count,
                         "jf_session_boundary": [False] * count})


def _segments(trace):
    # TEST LOGIC: Decode only the rendered null-separated polygons for independent containment assertions.
    segments, current = [], []
    for stamp, value in zip(trace.x, trace.y):
        if stamp is None:
            segments.append(current)
            current = []
        else:
            current.append((stamp, value))
    assert not current
    return segments


def test_band_polygons_close_each_supported_run_independently():
    # TEST LOGIC: Different band widths and centres on each side of a gap must not produce a connecting fill.
    times = np.array([pd.Timestamp("2026-10-01T14:00Z"), pd.Timestamp("2026-10-01T14:01Z"),
                      None, pd.Timestamp("2026-10-02T14:00Z")], dtype=object)
    x, y = _band_polygons(times, np.array([101., 102., np.nan, 103.]), np.array([99., 100., np.nan, 101.]))
    assert x == [times[0], times[1], times[1], times[0], times[0], None,
                 times[3], times[3], times[3], None]
    assert y == [101., 102., 100., 99., 101., None, 103., 101., 103., None]


def test_cutoff_band_and_baseline_break_at_actual_session_closures():
    # TEST LOGIC: The closure is shorter than max_gap and must still split shading into independently closed polygons.
    times = ["2026-10-01T14:00Z", "2026-10-01T14:01Z", "2026-10-02T14:00Z", "2026-10-02T14:01Z"]
    reviewed = _review(times)
    reviewed.loc[2, "jf_session_boundary"] = True
    original = reviewed.copy(deep=True)
    figure = trade_figure(reviewed, max_gap="1D")
    band = next(trace for trace in figure.data if trace.name == "Cutoff band")
    baseline = next(trace for trace in figure.data if trace.name == "Baseline")
    segments = _segments(band)
    assert band.fill == "toself" and not band.connectgaps
    assert len(segments) == 2
    assert all(segment[0] == segment[-1] for segment in segments)
    assert max(stamp for stamp, _ in segments[0]) < min(stamp for stamp, _ in segments[1])
    assert list(baseline.x).count(None) == 1
    pd.testing.assert_frame_equal(reviewed, original)


def test_unsupported_reference_hole_survives_when_its_trade_is_not_drawn():
    # TEST LOGIC: The four-point budget omits row2; both lines and fill must remember its unsupported-reference hole.
    reviewed = _review(pd.date_range("2026-10-01T14:00Z", periods=6, freq="min"))
    reviewed.loc[2, ["jf_baseline", "jf_threshold"]] = np.nan
    reviewed.loc[2, "jf_status"] = "insufficient_history"
    original = reviewed.copy(deep=True)
    figure = trade_figure(reviewed, max_points=4)
    band = next(trace for trace in figure.data if trace.name == "Cutoff band")
    baseline = next(trace for trace in figure.data if trace.name == "Baseline")
    segments = _segments(band)
    assert len(segments) == 2
    assert set(stamp for stamp in baseline.x if stamp is not None) == set(reviewed.loc[[0, 1, 4, 5], "jf_time"])
    assert list(baseline.x).count(None) == 1
    assert max(stamp for stamp, _ in segments[0]) == reviewed.loc[1, "jf_time"]
    assert min(stamp for stamp, _ in segments[1]) == reviewed.loc[4, "jf_time"]
    assert figure.layout.meta["total_timed_trades"] == 6
    assert figure.layout.meta["displayed_trades"] == 4
    pd.testing.assert_frame_equal(reviewed, original)


def test_all_unsupported_references_keep_trade_markers_without_a_fabricated_band():
    # TEST LOGIC: Missing diagnostics remain visible as unscored trades and never create a zero-valued shaded area.
    reviewed = _review(pd.date_range("2026-10-01T14:00Z", periods=4, freq="min"))
    reviewed[["jf_baseline", "jf_threshold"]] = np.nan
    reviewed["jf_status"] = "insufficient_history"
    figure = trade_figure(reviewed)
    band = next(trace for trace in figure.data if trace.name == "Cutoff band")
    unscored = next(trace for trace in figure.data if trace.name == "Unscored")
    assert len(band.x) == len(band.y) == 0
    assert len(unscored.x) == 4
    assert list(unscored.y) == [100.] * 4


def test_optional_policy_markers_keep_uncertain_and_excluded_trades_distinct():
    # TEST LOGIC: Quantity-aware retention is a separate decision from clean support; exclusion remains a red marker even inside a method's statistical band.
    reviewed = _review(pd.date_range("2026-10-01T14:00Z", periods=3, freq="min"))
    reviewed["jf_status"] = ["ok", "retained_uncertain", "policy_outlier"]
    reviewed["jf_is_outlier"] = [False, False, True]
    reviewed["jf_quantity_notional"] = [250_000., 2_000_000., 250_000.]
    reviewed["jf_support_trend"] = [100.] * 3
    reviewed["jf_support_distance_bps"] = [0., .5, .75]
    reviewed["jf_support_reliable"] = [True] * 3
    reviewed.attrs["jump_filter"] = {"config": {"quantity_rule": True, "max_deviation_rule": False}}
    figure = trade_figure(reviewed)
    retained = next(trace for trace in figure.data if trace.name == "Retained · uncertain")
    excluded = next(trace for trace in figure.data if trace.name == "Outlier")
    residuals = next(trace for trace in figure.data if trace.name == "Residual")
    assert pd.to_datetime(retained.x, utc=True).tolist() == [reviewed.loc[1, "jf_time"]]
    assert retained.marker.symbol == "circle-open" and retained.marker.color == AMBER
    assert pd.to_datetime(excluded.x, utc=True).tolist() == [reviewed.loc[2, "jf_time"]]
    assert excluded.marker.symbol == "x" and excluded.marker.color == RED
    assert "Quantity (notional)" in retained.hovertemplate and "Distance to support" in retained.hovertemplate
    assert retained.customdata[0][11:15].tolist() == [2_000_000., 100., .5, "True"]
    assert list(residuals.marker.symbol) == ["circle", "circle-open", "x"]
    diagnostics = diagnostic_figure(reviewed)
    coverage = diagnostics.data[2]
    colors = dict(zip(coverage.y, coverage.marker.color))
    assert colors["policy outlier"] == RED
    assert colors["retained uncertain"] == AMBER


def test_policy_support_holes_do_not_inherit_the_finite_statistical_band():
    # TEST LOGIC: Unreliable row2 is omitted by sampling while the selected method remains fully supported; only the policy paths must break.
    reviewed = _review(pd.date_range("2026-10-01T14:00Z", periods=6, freq="min"))
    reviewed["jf_support_trend"] = [105.] * 6
    reviewed["jf_support_reliable"] = [True, True, False, True, True, True]
    reviewed.attrs["jump_filter"] = {"config": {"max_deviation_rule": True, "max_deviation_bps": 10., "spread_units_per_bp": 1.}}
    original = reviewed.copy(deep=True)
    figure = trade_figure(reviewed, max_points=4)
    support = next(trace for trace in figure.data if trace.name == "Support trend")
    baseline = next(trace for trace in figure.data if trace.name == "Baseline")
    limits = [trace for trace in figure.data if trace.name == "10 bp limit"]
    band = next(trace for trace in figure.data if trace.name == "Cutoff band")
    assert list(support.x).count(None) == 1
    assert list(baseline.x).count(None) == 0
    assert len(_segments(band)) == 1
    assert len(limits) == 2 and limits[0].showlegend and not limits[1].showlegend
    assert list(limits[0].x).count(None) == list(limits[1].x).count(None) == 1
    np.testing.assert_equal(limits[0].y, [115., 115., np.nan, 115., 115.])
    np.testing.assert_equal(limits[1].y, [95., 95., np.nan, 95., 95.])
    assert not support.connectgaps and all(not trace.connectgaps for trace in limits)
    pd.testing.assert_frame_equal(reviewed, original)


def test_policy_limit_uses_configured_bp_conversion_and_avoids_duplicate_baselines():
    # TEST LOGIC: A percentage-point spread needs an explicit .01-per-bp conversion even when its chart label is descriptive.
    reviewed = _review(pd.date_range("2026-10-01T14:00Z", periods=3, freq="min"))
    reviewed["jf_baseline"] = 1.1
    reviewed["jf_threshold"] = .01
    reviewed["jf_support_trend"] = 1.1
    reviewed["jf_support_reliable"] = True
    reviewed.attrs["jump_filter"] = {"config": {"max_deviation_rule": True, "max_deviation_bps": 7.5, "spread_units_per_bp": .01}}
    figure = trade_figure(reviewed, unit="percentage points")
    limits = [trace for trace in figure.data if trace.name == "7.5 bp limit"]
    assert not any(trace.name == "Support trend" for trace in figure.data)
    assert len(limits) == 2
    np.testing.assert_allclose(limits[0].y, [1.175] * 3)
    np.testing.assert_allclose(limits[1].y, [1.025] * 3)
