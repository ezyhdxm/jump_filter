"""Behavioral checks for row integrity and economically distinct price paths."""

# SETUP LOGIC: Import the public contract and testing utilities.
import numpy as np
import pandas as pd
import pytest

from jump_filter import FilterConfig, METHODS, filter_trades, make_demo, select_fit_data, summarize


# TEST LOGIC: Build small labeled frames without sharing mutable inputs.
def trades(values, *, cusip="A", frequency="h"):
    return pd.DataFrame(
        {
            "CUSIP": cusip,
            "time": pd.date_range("2025-01-01", periods=len(values), freq=frequency),
            "spread": np.asarray(values, dtype=float),
            "source_row": np.arange(len(values)),
        }
    )


# TEST LOGIC: Every public method returns the auditable fields promised by the API.
@pytest.mark.parametrize("method", METHODS)
def test_public_methods_preserve_input_and_duplicate_index(method):
    original = trades(np.full(48, 100.0))
    original.loc[17, "spread"] = 130.0
    original = original.sample(frac=1, random_state=4)
    original.index = np.repeat(np.arange(24), 2)
    snapshot = original.copy(deep=True)
    result = filter_trades(original, FilterConfig(method=method))
    pd.testing.assert_frame_equal(original, snapshot)
    pd.testing.assert_frame_equal(result[snapshot.columns], snapshot)
    required = {
        "jf_is_outlier", "jf_score", "jf_baseline", "jf_scale", "jf_status",
        "jf_regime_change", "jf_weight", "jf_n_reference", "jf_time", "jf_row_id",
        "jf_fit_eligible",
    }
    assert required <= set(result.columns)
    assert result["jf_row_id"].is_unique
    assert result["jf_is_outlier"].dtype == bool
    assert result["jf_weight"].between(0, 1).all()


# TEST LOGIC: Changing another bond cannot affect this bond's estimates or decisions.
@pytest.mark.parametrize("method", METHODS)
def test_cusip_isolation(method):
    first = trades(np.full(48, 100.0))
    first.loc[20, "spread"] = 130.0
    second = trades(np.linspace(1000, 1200, 48), cusip="B")
    isolated = filter_trades(first, FilterConfig(method=method))
    combined = filter_trades(pd.concat([second, first]), FilterConfig(method=method))
    selected = combined.loc[combined["CUSIP"].eq("A")]
    comparison = ["jf_is_outlier", "jf_baseline", "jf_scale", "jf_n_reference"]
    pd.testing.assert_frame_equal(
        isolated[comparison].reset_index(drop=True),
        selected[comparison].reset_index(drop=True),
    )


# TEST LOGIC: Original row order is only presentation; time ordering drives calculations.
@pytest.mark.parametrize("method", METHODS)
def test_unsorted_input_is_equivalent(method):
    ordered = trades(100 + np.sin(np.arange(90) / 13))
    ordered.loc[43, "spread"] += 25.0
    shuffled = ordered.sample(frac=1, random_state=19)
    direct = filter_trades(ordered, FilterConfig(method=method))
    mixed = filter_trades(shuffled, FilterConfig(method=method))
    columns = ["jf_is_outlier", "jf_baseline", "jf_scale", "jf_n_reference"]
    pd.testing.assert_frame_equal(
        direct.set_index("source_row")[columns].sort_index(),
        mixed.set_index("source_row")[columns].sort_index(),
    )


# TEST LOGIC: Simultaneous trades exclude the whole cohort from their reference sample.
@pytest.mark.parametrize("method", METHODS)
def test_equal_timestamp_order_invariance(method):
    frame = trades(np.full(60, 100.0))
    extra = frame.iloc[[30]].copy()
    extra["spread"] = 125.0
    extra["source_row"] = 60
    frame.loc[30, "spread"] = 130.0
    frame = pd.concat([frame, extra], ignore_index=True)
    direct = filter_trades(frame, FilterConfig(method=method))
    mixed = filter_trades(frame.sample(frac=1, random_state=3), FilterConfig(method=method))
    columns = ["jf_is_outlier", "jf_baseline", "jf_scale", "jf_n_reference"]
    pd.testing.assert_frame_equal(
        direct.set_index("source_row")[columns].sort_index(),
        mixed.set_index("source_row")[columns].sort_index(),
    )
    cohort = direct.loc[direct["source_row"].isin([30, 60])]
    tolerance = 0.02 if method == "robust_trend" else 1e-10
    np.testing.assert_allclose(cohort["jf_baseline"], 100.0, atol=tolerance)
    assert cohort["jf_is_outlier"].all()


# TEST LOGIC: Invalid identifiers, times, and numerical values remain audit-visible.
def test_invalid_records_are_retained_but_have_zero_weight():
    frame = trades(np.full(40, 100.0))
    frame["time"] = frame["time"].astype(object)
    frame.loc[3, "CUSIP"] = None
    frame.loc[7, "time"] = "not a timestamp"
    frame.loc[9, "spread"] = np.nan
    frame.loc[11, "spread"] = np.inf
    result = filter_trades(frame)
    invalid = result.loc[[3, 7, 9, 11]]
    assert len(result) == len(frame)
    assert invalid["jf_weight"].eq(0.0).all()
    assert not invalid["jf_is_outlier"].any()
    assert invalid["jf_status"].eq("invalid_input").all()


# TEST LOGIC: Thin support must not turn a single observation into a confident outlier.
@pytest.mark.parametrize("method", METHODS)
def test_short_history_abstains(method):
    frame = trades([100.0, 130.0, 100.0])
    result = filter_trades(frame, FilterConfig(method=method, min_neighbors=6))
    assert not result["jf_is_outlier"].any()
    assert result["jf_n_reference"].max() < 6
    assert result["jf_weight"].eq(0).all()


# TEST LOGIC: A zero median deviation still identifies a material isolated deviation.
@pytest.mark.parametrize("method", METHODS)
def test_flat_series_and_isolated_spike(method):
    frame = trades(np.full(90, 100.0))
    frame.loc[45, "spread"] = 130.0
    result = filter_trades(frame, FilterConfig(method=method))
    assert result.loc[45, "jf_is_outlier"]
    tolerance = 0.02 if method == "robust_trend" else 1e-10
    assert result.loc[45, "jf_baseline"] == pytest.approx(100.0, abs=tolerance)
    assert np.isfinite(result.loc[45, "jf_score"])
    assert result["jf_is_outlier"].sum() == 1


# TEST LOGIC: A long market gap separates support; distant history cannot flag a new level.
@pytest.mark.parametrize("method", METHODS)
def test_long_gap_resets_support(method):
    before = trades(np.full(40, 100.0))
    after = trades(np.full(3, 130.0))
    after["time"] += pd.Timedelta("10D")
    frame = pd.concat([before, after], ignore_index=True)
    result = filter_trades(frame, FilterConfig(method=method, max_gap="1D"))
    assert not result.iloc[-3:]["jf_is_outlier"].any()
    assert result.iloc[-3:]["jf_n_reference"].max() < 6


# TEST LOGIC: A lasting repricing is retained while isolated trade premiums are flagged.
@pytest.mark.parametrize("method", METHODS)
def test_persistent_level_shift_is_not_wholesale_deleted(method):
    values = np.r_[np.full(45, 100.0), np.full(45, 120.0)]
    frame = trades(values)
    frame.loc[20, "spread"] += 30.0
    frame.loc[70, "spread"] += 30.0
    result = filter_trades(frame, FilterConfig(method=method))
    assert result.loc[[20, 70], "jf_is_outlier"].all()
    assert not result.loc[50:65, "jf_is_outlier"].any()
    assert result.loc[45:65, "jf_is_outlier"].sum() <= 3


# TEST LOGIC: A causal decision depends only on earlier timestamp cohorts and this trade.
def test_causal_future_perturbation_and_appending_are_prefix_invariant():
    frame = trades(100 + 0.03 * np.arange(90))
    duplicate = frame.iloc[[44]].copy()
    duplicate["source_row"] = 90
    duplicate["spread"] += 20.0
    frame = pd.concat([frame, duplicate], ignore_index=True)
    prefix = frame.loc[frame["time"].le(pd.Timestamp("2025-01-02 20:00"))]
    perturbed = frame.copy()
    perturbed.loc[perturbed["time"].gt(prefix["time"].max()), "spread"] += 1000.0
    config = FilterConfig(method="causal_ewma")
    direct = filter_trades(frame, config).set_index("source_row")
    changed = filter_trades(perturbed, config).set_index("source_row")
    truncated = filter_trades(prefix, config).set_index("source_row")
    columns = ["jf_is_outlier", "jf_baseline", "jf_scale", "jf_status", "jf_regime_change"]
    pd.testing.assert_frame_equal(direct.loc[truncated.index, columns], truncated[columns])
    pd.testing.assert_frame_equal(direct.loc[truncated.index, columns], changed.loc[truncated.index, columns])


# TEST LOGIC: Scale the dimensional floor; dimensionless controls retain their meaning.
@pytest.mark.parametrize("method", METHODS)
def test_scale_equivariance(method):
    frame = trades(100 + np.sin(np.arange(90) / 9))
    frame.loc[40, "spread"] += 25.0
    original = filter_trades(frame, FilterConfig(method=method, abs_floor=1.0))
    scaled_frame = frame.copy()
    scaled_frame["spread"] *= 0.01
    scaled = filter_trades(
        scaled_frame,
        FilterConfig(method=method, abs_floor=0.01, reversion_tolerance=2.0),
    )
    pd.testing.assert_series_equal(original["jf_is_outlier"], scaled["jf_is_outlier"])
    np.testing.assert_allclose(original["jf_score"], scaled["jf_score"], equal_nan=True, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(original["jf_baseline"] * 0.01, scaled["jf_baseline"], equal_nan=True)


# TEST LOGIC: Custom names and an explicit timezone work without renaming the caller's data.
def test_column_mapping_and_timezone():
    frame = trades(np.full(48, 100.0)).rename(
        columns={"CUSIP": "bond", "time": "observed_at", "spread": "oas"}
    )
    frame.loc[22, "oas"] = 130.0
    result = filter_trades(
        frame, cusip_col="bond", time_col="observed_at", spread_col="oas",
        timezone="America/New_York",
    )
    assert result.loc[22, "jf_is_outlier"]
    assert result.loc[0, "jf_time"] == pd.Timestamp("2025-01-01 05:00:00", tz="UTC")
    pd.testing.assert_frame_equal(result[frame.columns], frame)


# TEST LOGIC: The bundled demonstration and aggregate view use the public engine contract.
def test_demo_and_summary_are_usable():
    frame = make_demo()
    assert {"CUSIP", "time", "spread"} <= set(frame.columns)
    assert frame["CUSIP"].nunique() >= 2
    result = filter_trades(frame)
    report = summarize(result)
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert not report.empty


# TEST LOGIC: Input validation should fail before partially processing an invalid request.
def test_missing_required_columns_raise_actionable_error():
    with pytest.raises((ValueError, KeyError), match="spread"):
        filter_trades(pd.DataFrame({"CUSIP": ["A"], "time": ["2025-01-01"]}))


# TEST LOGIC: Empty inputs are accepted and preserve the output schema.
def test_empty_frame_preserves_schema():
    frame = trades([])
    result = filter_trades(frame)
    assert result.empty
    assert {"jf_is_outlier", "jf_status", "jf_weight"} <= set(result.columns)


# TEST LOGIC: Repeating a clean print cannot make one timestamp dominate its neighbors.
@pytest.mark.parametrize("method", METHODS)
def test_duplicating_a_timestamp_does_not_increase_reference_support(method):
    frame = trades(100 + np.sin(np.arange(70) / 15))
    repeated = pd.concat([frame, pd.concat([frame.iloc[[30]]] * 40)], ignore_index=True)
    direct = filter_trades(frame, FilterConfig(method=method))
    enlarged = filter_trades(repeated, FilterConfig(method=method)).iloc[: len(frame)]
    columns = ["jf_is_outlier", "jf_baseline", "jf_scale", "jf_n_reference"]
    pd.testing.assert_frame_equal(direct[columns], enlarged[columns])


# TEST LOGIC: Ambiguous/nonexistent local times and ambiguous numeric epochs must abstain.
def test_dst_and_numeric_epoch_inputs_are_invalid():
    frame = pd.DataFrame(
        {"CUSIP": ["A", "A", "A"],
         "time": ["2025-11-02 01:30:00", "2025-03-09 02:30:00", 1735689600],
         "spread": [100.0, 100.0, 100.0]}
    )
    result = filter_trades(frame, timezone="America/New_York")
    assert result["jf_status"].eq("invalid_input").all()
    assert result["jf_weight"].eq(0).all()
    assert result["jf_time"].isna().all()


# TEST LOGIC: Reject illegal numerical controls rather than silently changing their meaning.
@pytest.mark.parametrize(
    "kwargs", [{"threshold": 0}, {"abs_floor": -1}, {"alpha": 1.1},
               {"persistence": 1}, {"min_neighbors": 100}, {"window": True},
               {"horizon": "0D"}, {"max_gap": "-1D"}, {"threshold": np.nan},
               {"horizon": "NaT"}, {"max_gap": "NaT"}],
)
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        FilterConfig(**kwargs)


# TEST LOGIC: Every flag must agree with the numerical band displayed in plots and exports.
@pytest.mark.parametrize("method", METHODS)
def test_flags_are_outside_the_displayed_band_and_have_reduced_weight(method):
    result = filter_trades(make_demo(), FilterConfig(method=method))
    flagged = result.loc[result["jf_is_outlier"]]
    assert len(flagged) > 0
    assert flagged["jf_residual"].abs().gt(flagged["jf_threshold"]).all()
    assert flagged["jf_weight"].gt(0).all()
    assert flagged["jf_weight"].lt(1).all()
    np.testing.assert_allclose(
        flagged["jf_score"], flagged["jf_residual"].abs() / flagged["jf_scale"],
    )
    unsupported = result.loc[result["jf_status"].eq("insufficient_history")]
    assert unsupported["jf_weight"].eq(0).all()


# TEST LOGIC: Two-sided trend projection should retain smooth drift and identify extra premiums.
@pytest.mark.parametrize("method", ["local_linear", "consensus"])
def test_smooth_trend_with_spikes_regression(method):
    frame = trades(100 + 0.25 * np.arange(120) + 0.05 * np.sin(np.arange(120)))
    frame.loc[30, "spread"] += 8.0
    frame.loc[95, "spread"] -= 9.0
    result = filter_trades(frame, FilterConfig(method=method))
    assert result.loc[[30, 95], "jf_is_outlier"].all()
    assert result["jf_is_outlier"].sum() == 2
    assert not result.loc[40:80, "jf_regime_change"].any()


# TEST LOGIC: Historical fitting uses future observations; causal EWMA remains an explicit option.
@pytest.mark.parametrize("method", ["hampel", "rolling_iqr", "local_linear", "local_piecewise", "jump_reversion", "multiscale", "robust_trend", "consensus"])
def test_retrospective_methods_can_use_later_support(method):
    frame = trades(np.full(70, 100.0))
    frame.loc[4, "spread"] = 125.0
    full = filter_trades(frame, FilterConfig(method=method))
    prefix = filter_trades(frame.iloc[:5], FilterConfig(method=method))
    assert full.loc[4, "jf_is_outlier"]
    assert full.loc[4, "jf_fit_eligible"] == False
    assert prefix["jf_fit_eligible"].eq(False).all()
    assert not prefix["jf_is_outlier"].any()


# TEST LOGIC: Invalid values abstain consistently for each new algorithm, not only the default.
@pytest.mark.parametrize("method", METHODS)
def test_each_method_marks_invalid_and_unsupported_rows_ineligible(method):
    frame = trades(np.full(50, 100.0))
    frame.loc[5, "spread"] = np.nan
    frame.loc[7, "CUSIP"] = None
    result = filter_trades(frame, FilterConfig(method=method))
    invalid = result.loc[[5, 7]]
    assert invalid["jf_status"].eq("invalid_input").all()
    assert invalid["jf_fit_eligible"].eq(False).all()
    assert invalid["jf_weight"].eq(0).all()
    expected = result["jf_status"].eq("ok") & ~result["jf_is_outlier"]
    pd.testing.assert_series_equal(result["jf_fit_eligible"], expected, check_names=False)


# TEST LOGIC: Short calendar gaps keep legitimate support; a max-gap reset is strictly larger.
@pytest.mark.parametrize("method", METHODS)
def test_short_gaps_do_not_reset_reference_support(method):
    frame = trades(np.full(70, 100.0))
    frame.loc[35:, "time"] += pd.Timedelta("2h")
    frame.loc[36, "spread"] = 130.0
    result = filter_trades(frame, FilterConfig(method=method, max_gap="1D"))
    assert result.loc[36, "jf_is_outlier"]
    assert result.loc[36, "jf_n_reference"] >= 6
    assert result.loc[34, "jf_fit_eligible"]


# TEST LOGIC: Genuine steep trends should remain eligible under explicit trend-aware methods.
@pytest.mark.parametrize("method", ["local_linear", "local_piecewise", "robust_trend", "consensus"])
def test_steep_trend_with_isolated_premiums(method):
    frame = trades(100 + 0.75 * np.arange(120))
    frame.loc[35, "spread"] += 10.0
    frame.loc[85, "spread"] -= 10.0
    result = filter_trades(frame, FilterConfig(method=method))
    assert result.loc[[35, 85], "jf_is_outlier"].all()
    assert result.loc[45:70, "jf_fit_eligible"].all()


# TEST LOGIC: Dimensionless method controls retain decisions and numerical scale after unit changes.
@pytest.mark.parametrize("method", ["rolling_iqr", "multiscale", "robust_trend"])
def test_new_methods_scale_equivariance_with_nondefault_controls(method):
    frame = trades(100 + 0.1 * np.arange(80) + 0.1 * np.sin(np.arange(80)))
    frame.loc[42, "spread"] += 15.0
    controls = {"iqr_multiplier": 2.0, "multiscale_votes": 3, "trend_penalty": 12.0, "huber_delta": 2.0}
    original = filter_trades(frame, FilterConfig(method=method, abs_floor=1.0, **controls))
    transformed = frame.copy()
    transformed["spread"] = transformed["spread"] * 10.0 + 500.0
    scaled = filter_trades(transformed, FilterConfig(method=method, abs_floor=10.0, **controls))
    pd.testing.assert_series_equal(original["jf_is_outlier"], scaled["jf_is_outlier"])
    pd.testing.assert_series_equal(original["jf_fit_eligible"], scaled["jf_fit_eligible"])
    np.testing.assert_allclose(scaled["jf_baseline"], original["jf_baseline"] * 10 + 500, equal_nan=True, atol=0.01)
    np.testing.assert_allclose(scaled["jf_score"], original["jf_score"], equal_nan=True, rtol=0.01, atol=0.01)


# TEST LOGIC: Wider IQR fences must lower rejection on an unchanged reference population.
def test_iqr_multiplier_controls_fence_width():
    frame = trades(np.tile([98.0, 99.0, 100.0, 101.0, 102.0], 20))
    frame.loc[50, "spread"] = 107.0
    narrow = filter_trades(frame, FilterConfig(method="rolling_iqr", iqr_multiplier=1.0))
    wide = filter_trades(frame, FilterConfig(method="rolling_iqr", iqr_multiplier=4.0))
    assert narrow.loc[50, "jf_is_outlier"]
    assert not wide.loc[50, "jf_is_outlier"]
    assert wide.loc[50, "jf_threshold"] > narrow.loc[50, "jf_threshold"]


# TEST LOGIC: Voting at more horizons must demand at least as much confirmation.
def test_multiscale_vote_requirement_reduces_or_preserves_flags():
    frame = trades(100 + 0.2 * np.sin(np.arange(120) / 7))
    frame.loc[30:35, "spread"] += 10.0
    frame.loc[80, "spread"] -= 20.0
    permissive = filter_trades(frame, FilterConfig(method="multiscale", multiscale_votes=1))
    strict = filter_trades(frame, FilterConfig(method="multiscale", multiscale_votes=3))
    assert strict.loc[80, "jf_is_outlier"]
    assert not (strict["jf_is_outlier"] & ~permissive["jf_is_outlier"]).any()


# TEST LOGIC: New controls fail clearly on meaningless or numerically unsafe settings.
@pytest.mark.parametrize("kwargs", [
    {"iqr_multiplier": 0}, {"iqr_multiplier": np.nan}, {"trend_penalty": 0},
    {"trend_penalty": np.inf}, {"huber_delta": -1}, {"max_iter": 0},
    {"max_iter": True}, {"tolerance": 0}, {"tolerance": np.nan},
    {"multiscale_votes": 0}, {"multiscale_votes": 4}, {"multiscale_votes": True},
])
def test_invalid_new_configuration_controls_raise(kwargs):
    with pytest.raises(ValueError):
        FilterConfig(**kwargs)


# TEST LOGIC: One ADMM iteration cannot masquerade as a fitted converged historical model.
def test_robust_trend_nonconvergence_abstains_instead_of_flagging():
    values = 100 + 0.4 * np.sin(np.arange(90) / 4)
    frame = trades(values)
    frame.loc[30, "spread"] += 25.0
    frame.loc[60:, "spread"] += 12.0
    result = filter_trades(frame, FilterConfig(method="robust_trend", max_iter=1, tolerance=1e-12))
    failure = result["jf_status"].eq("solver_not_converged")
    assert failure.any()
    assert result.loc[failure, "jf_solver_converged"].eq(False).all()
    assert result.loc[failure, "jf_solver_iterations"].eq(1).all()
    assert result.loc[failure, "jf_baseline"].isna().all()
    assert not result.loc[failure, "jf_is_outlier"].any()
    assert result.loc[failure, "jf_weight"].eq(0).all()
    assert not result.loc[failure, "jf_fit_eligible"].any()


# TEST LOGIC: Normal solver settings must return usable, fully audited trend decisions.
def test_robust_trend_default_convergence_produces_eligible_clean_rows():
    frame = trades(np.r_[np.full(45, 100.0), np.full(45, 115.0)])
    frame.loc[20, "spread"] = 130.0
    result = filter_trades(frame, FilterConfig(method="robust_trend"))
    assert not result["jf_status"].eq("solver_not_converged").any()
    assert result["jf_solver_converged"].dropna().eq(True).all()
    assert result.loc[20, "jf_is_outlier"]
    assert result["jf_fit_eligible"].sum() >= 80


# TEST LOGIC: Hard fitting candidates contain only accepted rows and preserve the caller's frame.
@pytest.mark.parametrize("method", METHODS)
def test_hard_fit_selection_is_explicit_and_preserves_input(method):
    frame = trades(np.full(70, 100.0))
    frame.loc[30, "spread"] = 130.0
    result = filter_trades(frame, FilterConfig(method=method))
    snapshot = result.copy(deep=True)
    selected = select_fit_data(result)
    pd.testing.assert_frame_equal(result, snapshot)
    assert selected["jf_fit_eligible"].all()
    assert selected["jf_status"].eq("ok").all()
    assert not selected["jf_is_outlier"].any()
    assert selected["jf_fit_weight"].eq(1).all()
    assert 30 not in selected.index


# TEST LOGIC: Soft fitting permits bounded influence only for supported decisions.
def test_soft_fit_selection_excludes_abstention_and_causal_provisional_by_default():
    result = pd.DataFrame({
        "jf_status": ["ok", "outlier", "provisional_jump", "invalid_input", "solver_not_converged"],
        "jf_is_outlier": [False, True, True, False, False],
        "jf_weight": [1.0, 0.15, 0.2, 0.0, 0.0],
        "jf_fit_eligible": [True, False, False, False, False],
    })
    selected = select_fit_data(result, policy="soft")
    assert selected.index.tolist() == [0, 1]
    assert selected["jf_fit_weight"].tolist() == [1.0, 0.15]
    explicit = select_fit_data(result, policy="soft", include_provisional=True)
    assert explicit.index.tolist() == [0, 1, 2]


# TEST LOGIC: Bad fitting policies and non-annotated data fail before selecting rows.
@pytest.mark.parametrize("kwargs", [{"policy": "unknown"}, {"policy": "soft"}])
def test_fit_selection_requires_valid_policy_and_engine_annotations(kwargs):
    with pytest.raises(ValueError):
        select_fit_data(trades([100, 101]), **kwargs)


# TEST LOGIC: Reject calendar strings that only fail later when a session clock is constructed.
@pytest.mark.parametrize("holiday", ["2026-01-01T08:00", "NaT", "20260101", 20260101])
def test_holiday_controls_require_complete_local_dates(holiday):
    with pytest.raises(ValueError):
        FilterConfig(holidays=(holiday,))


# TEST LOGIC: Test session timestamps are explicitly localized before being converted to UTC.
def session_trades(first_date="2025-01-03", next_date="2025-01-06"):
    before = pd.date_range(f"{first_date} 17:00", periods=8, freq="10min", tz="America/New_York")
    after = pd.date_range(f"{next_date} 08:00", periods=8, freq="10min", tz="America/New_York")
    frame = trades(np.full(16, 100.0))
    frame["time"] = before.append(after).tz_convert("UTC")
    frame.loc[8, "spread"] = 125.0
    return frame


# TEST LOGIC: Closed nights/weekends add no artificial trading-clock age or reset.
def test_trading_clock_preserves_support_over_weekend_but_wall_clock_resets():
    frame = session_trades()
    controls = {"method": "causal_ewma", "max_gap": "90min", "horizon": "1D"}
    trading = filter_trades(frame, FilterConfig(time_basis="trading", **controls))
    wall = filter_trades(frame, FilterConfig(time_basis="wall", **controls))
    assert trading.loc[8, "jf_gap_minutes"] == pytest.approx(20.0)
    assert trading.loc[8, "jf_wall_gap_minutes"] == pytest.approx(3710.0)
    assert trading.loc[8, "jf_session_boundary"]
    assert not trading.loc[1, "jf_session_boundary"]
    assert wall.loc[8, "jf_gap_minutes"] == pytest.approx(3710.0)
    assert trading.loc[8, "jf_n_reference"] == 8
    assert trading.loc[8, "jf_is_outlier"]
    assert wall.loc[8, "jf_n_reference"] == 0
    assert wall.loc[8, "jf_status"] == "insufficient_history"
    assert not wall.loc[8, "jf_is_outlier"]
    assert trading.loc[8, "jf_reference_span_minutes"] == pytest.approx(70.0)
    assert trading.loc[8, "jf_reference_density_per_hour"] == pytest.approx(8 / (70 / 60))


# TEST LOGIC: A closure is removed only when the caller explicitly supplies its date.
def test_supplied_holiday_is_compressed_and_no_holiday_calendar_is_guessed():
    frame = session_trades(next_date="2025-01-07")
    controls = {"method": "causal_ewma", "time_basis": "trading", "max_gap": "90min"}
    supplied = filter_trades(frame, FilterConfig(holidays=("2025-01-06",), **controls))
    unsupplied = filter_trades(frame, FilterConfig(**controls))
    assert supplied.loc[8, "jf_gap_minutes"] == pytest.approx(20.0)
    assert unsupplied.loc[8, "jf_gap_minutes"] == pytest.approx(650.0)
    assert supplied.loc[8, "jf_is_outlier"]
    assert unsupplied.loc[8, "jf_n_reference"] == 0


# TEST LOGIC: Genuine inactivity while the bond market is open remains a support-breaking gap.
def test_trading_clock_does_not_compress_an_illiquid_intraday_gap():
    before = pd.date_range("2025-01-06 13:00", periods=8, freq="10min", tz="America/New_York")
    after = pd.date_range("2025-01-06 15:30", periods=3, freq="10min", tz="America/New_York")
    frame = trades(np.r_[np.full(8, 100.0), np.full(3, 125.0)])
    frame["time"] = before.append(after).tz_convert("UTC")
    result = filter_trades(frame, FilterConfig(method="causal_ewma", time_basis="trading", max_gap="45min"))
    assert result.loc[8, "jf_gap_minutes"] == pytest.approx(80.0)
    assert result.loc[8, "jf_n_reference"] == 0
    assert not result.iloc[8:]["jf_is_outlier"].any()
    assert not result.iloc[8:]["jf_fit_eligible"].any()


# TEST LOGIC: Mixing bond cohorts must not change a bond's compressed time or density audit.
def test_calendar_density_diagnostics_are_cusip_isolated():
    first = session_trades()
    second = trades(np.full(40, 1000.0), cusip="B", frequency="min")
    second["time"] = pd.date_range("2025-01-06 09:00", periods=40, freq="min", tz="America/New_York").tz_convert("UTC")
    config = FilterConfig(method="hampel", time_basis="trading")
    direct = filter_trades(first, config)
    combined = filter_trades(pd.concat([second, first]), config)
    selected = combined.loc[combined["CUSIP"].eq("A")]
    columns = ["jf_clock_time", "jf_gap_minutes", "jf_reference_span_minutes", "jf_reference_density_per_hour"]
    pd.testing.assert_frame_equal(direct[columns].reset_index(drop=True), selected[columns].reset_index(drop=True))


# TEST LOGIC: A downward market turn is not enough evidence to delete genuine clean trade levels.
@pytest.mark.parametrize("drop", [0.0, 20.0])
def test_piecewise_rule_abstains_at_uncertain_turns_without_labeling_market_move_bad(drop):
    index = np.arange(120)
    fair = 100 + 0.75 * np.minimum(index, 60) - 0.5 * np.maximum(index - 60, 0)
    fair[60:] -= drop
    frame = trades(fair)
    frame.loc[25, "spread"] += 15.0
    frame.loc[95, "spread"] -= 15.0
    result = filter_trades(frame, FilterConfig(method="local_piecewise"))
    assert result.loc[[25, 95], "jf_is_outlier"].all()
    assert not result.loc[52:68, "jf_is_outlier"].any()
    if drop:
        ambiguous = result.loc[52:68, "jf_status"].eq("ambiguous_transition")
        assert ambiguous.any()
        assert not result.loc[52:68].loc[ambiguous, "jf_fit_eligible"].any()
        assert result.loc[52:68].loc[ambiguous, "jf_weight"].eq(0).all()


# TEST LOGIC: Explicit sessions can encode early closes without an inferred exchange calendar.
def test_supplied_session_schedule_compresses_a_declared_early_close():
    frame = session_trades()
    frame.loc[:7, "time"] -= pd.Timedelta("6h")
    schedule = pd.DataFrame({
        "open": pd.to_datetime(["2025-01-03 08:00", "2025-01-06 08:00"]).tz_localize("America/New_York").tz_convert("UTC"),
        "close": pd.to_datetime(["2025-01-03 13:00", "2025-01-06 18:30"]).tz_localize("America/New_York").tz_convert("UTC"),
    })
    config = FilterConfig(method="causal_ewma", time_basis="trading", max_gap="90min")
    supplied = filter_trades(frame, config, session_schedule=schedule)
    standard = filter_trades(frame, config)
    assert supplied.loc[8, "jf_gap_minutes"] == pytest.approx(50.0)
    assert supplied.loc[8, "jf_is_outlier"]
    assert supplied.loc[8, "jf_n_reference"] == 8
    assert standard.loc[8, "jf_gap_minutes"] == pytest.approx(380.0)
    assert standard.loc[8, "jf_n_reference"] == 0


# TEST LOGIC: A supplied schedule's timezone is required rather than silently guessed.
def test_naive_session_schedule_is_rejected():
    schedule = pd.DataFrame({"open": [pd.Timestamp("2025-01-03 08:00")], "close": [pd.Timestamp("2025-01-03 18:30")]})
    with pytest.raises(ValueError):
        filter_trades(session_trades(), FilterConfig(time_basis="trading"), session_schedule=schedule)


# TEST LOGIC: Weekend records have no reference age under a weekday-only open-session clock.
def test_closed_session_print_is_auditable_and_not_fit_eligible():
    frame = session_trades()
    closed = frame.iloc[[0]].copy()
    closed["time"] = pd.Timestamp("2025-01-04 10:00", tz="America/New_York").tz_convert("UTC")
    closed["spread"] = 135.0
    frame = pd.concat([frame, closed], ignore_index=True)
    result = filter_trades(frame, FilterConfig(method="hampel", time_basis="trading"))
    assert not result.iloc[-1]["jf_fit_eligible"]
    assert not result.iloc[-1]["jf_is_outlier"]
    assert result.iloc[-1]["jf_weight"] == 0
    assert pd.isna(result.iloc[-1]["jf_clock_time"])


# TEST LOGIC: The declared clock and session bounds require valid, meaningful controls.
@pytest.mark.parametrize("kwargs", [
    {"time_basis": "row"}, {"session_open": "18:30", "session_close": "08:00"},
    {"session_open": "08:00-04:00"}, {"holidays": ("not-a-date",)},
])
def test_invalid_calendar_controls_raise(kwargs):
    with pytest.raises(ValueError):
        FilterConfig(**kwargs)


# TEST LOGIC: A short trading-clock age after a closure does not turn genuine opening repricing bad.
@pytest.mark.parametrize("method", [method for method in METHODS if method != "causal_ewma"])
def test_offline_methods_retain_or_abstain_at_clean_new_session_level_change(method):
    from benchmarks.compare import DEFAULT_SEEDS, make_case
    frame = make_case("calendar_sessions", DEFAULT_SEEDS[0], n=240)
    result = filter_trades(frame[["CUSIP", "time", "spread"]], FilterConfig(method=method, time_basis="trading"))
    clean = ~frame.loc[40:59, "true_contamination"]
    assert not result.loc[40:59].loc[clean, "jf_is_outlier"].any()
    assert not result.loc[40, "jf_is_outlier"]
    assert result.loc[40, "jf_session_boundary"]
    assert result.loc[40, "jf_wall_gap_minutes"] > 60 * 60
    assert result.loc[40, "jf_gap_minutes"] < 60
