"""Synthetic examples match their configured trading sessions and economic truth labels."""

# SETUP LOGIC: public data generator and independent calendar verification.
import numpy as np
import pandas as pd
from jump_filter import FilterConfig, make_demo
from jump_filter.clock import trading_clock
from jump_filter.demo import DEMO_SCENARIOS, _demo_times


# TEST LOGIC: independent reproduction keeps the advertised number of bonds and rows stable.
def test_demo_scenario_count_and_seed_reproduction():
    frame = make_demo(42)
    pd.testing.assert_frame_equal(frame, make_demo(42))
    assert frame["CUSIP"].nunique() == 11
    assert len(frame) == 1760
    assert set(frame["scenario"].unique()) == set(DEMO_SCENARIOS)
    assert frame.groupby("CUSIP").size().eq(160).all()
    assert frame.attrs["simulation"]["holidays"] == []


# TEST LOGIC: all synthetic trades lie inside the declared weekday opening interval.
def test_all_demo_timestamps_are_irregular_and_in_declared_sessions():
    frame = make_demo()
    local = frame["time"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    assert local.dt.weekday.lt(5).all()
    assert minute.ge(8 * 60).all()
    assert minute.lt(18 * 60 + 30).all()
    _, active, metadata = trading_clock(frame["time"], FilterConfig(time_basis="trading"))
    assert active.all()
    assert metadata["calendar_kind"] == "configured_weekdays"
    assert frame["time"].dt.tz is not None


# TEST LOGIC: inverse mapping retains active time intervals even across skipped closures.
def test_demo_elapsed_coordinates_match_independent_trading_clock():
    frame = make_demo()
    clock, _, _ = trading_clock(frame["time"], FilterConfig(time_basis="trading"))
    frame["check_clock"] = clock
    for _, part in frame.groupby("CUSIP"):
        expected = np.diff(part["elapsed_trading_minutes"].to_numpy()) * 60 * 1_000_000_000
        np.testing.assert_array_equal(np.diff(part["check_clock"].to_numpy()), expected)
        assert part["time"].is_monotonic_increasing
        assert part["interarrival_trading_minutes"].nunique() > 1


# TEST LOGIC: session endpoints move to the following opening instead of yielding closed timestamps.
def test_inverse_clock_exact_session_boundaries():
    times = _demo_times([0, 629, 630, 4 * 630, 5 * 630])
    local = times.tz_convert("America/New_York")
    assert list(local.strftime("%Y-%m-%d %H:%M")) == [
        "2026-09-14 08:00", "2026-09-14 18:29", "2026-09-15 08:00",
        "2026-09-18 08:00", "2026-09-21 08:00",
    ]


# TEST LOGIC: true asymmetric turning points and permanent level drops remain legitimate.
def test_turning_path_and_combined_drop_are_not_bad_trade_truth():
    frame = make_demo()
    turning = frame.loc[frame["scenario"].eq("turning_point")].reset_index(drop=True)
    dropped = frame.loc[frame["scenario"].eq("turning_with_drop")].reset_index(drop=True)
    np.testing.assert_allclose(turning.loc[[79, 80, 81], "true_mid"], [243.7, 244.0, 243.35])
    np.testing.assert_allclose(dropped.loc[[79, 80, 81], "true_mid"], [258.7, 247.0, 246.35])
    assert not turning.loc[80, "true_outlier"]
    assert not dropped.loc[80, "true_outlier"]
    assert turning["true_turning_point"].sum() == 1
    assert turning["true_regime_change"].sum() == 0
    assert dropped["true_turning_point"].sum() == 1
    assert dropped["true_regime_change"].sum() == 1


# TEST LOGIC: liquidity shift changes both observed scatter and open-session arrival density.
def test_liquidity_shift_changes_noise_and_trade_density():
    frame = make_demo()
    part = frame.loc[frame["scenario"].eq("liquidity_shift")].reset_index(drop=True)
    np.testing.assert_array_equal(part.iloc[:80]["observation_noise_sigma"], 0.4)
    np.testing.assert_array_equal(part.iloc[80:]["observation_noise_sigma"], 2.8)
    assert part.iloc[:80]["interarrival_trading_minutes"].between(2, 7).all()
    assert part.iloc[80:]["interarrival_trading_minutes"].between(25, 94).all()
    assert not part.loc[80, "true_outlier"]
    assert part["true_mid"].eq(250).all()


# TEST LOGIC: sparse support stress contains actual long open-session no-trade gaps.
def test_sparse_demo_retains_real_active_session_gaps():
    frame = make_demo()
    part = frame.loc[frame["scenario"].eq("sparse_gaps")].reset_index(drop=True)
    assert part.loc[::30, "interarrival_trading_minutes"].gt(31.5 * 60).all()
    assert part.loc[1:29, "interarrival_trading_minutes"].lt(18).all()
