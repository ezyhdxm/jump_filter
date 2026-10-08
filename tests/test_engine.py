"""Behavioral checks for row integrity and economically distinct price paths."""

# SETUP LOGIC: Import the public contract and testing utilities.
import numpy as np
import pandas as pd
import pytest

from jump_filter import FilterConfig, METHODS, filter_trades, make_demo, summarize


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
    np.testing.assert_allclose(cohort["jf_baseline"], 100.0)
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
    assert result.loc[45, "jf_baseline"] == pytest.approx(100.0)
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
    np.testing.assert_allclose(original["jf_score"], scaled["jf_score"], equal_nan=True, rtol=1e-8)
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
