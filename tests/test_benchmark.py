"""Keep construction truth, clean-target errors, and abstention metrics independent."""

# SETUP LOGIC: Import only the evaluation contract and deterministic numerical tools.
import numpy as np
import pandas as pd
import pytest

from benchmarks.compare import (
    DEFAULT_SEEDS, LEGACY_SEEDS, SCENARIOS, contamination_positions,
    downstream_fit, evaluate, make_case, run_comparison,
)


# TEST LOGIC: Complete hand-built audit frames; engine truth is never involved in these tests.
def audit_case(truth, spread, predicted, eligible, status, weight, baseline):
    length = len(truth)
    return pd.DataFrame({
        "true_contamination": truth, "true_fair_spread": np.full(length, 100.0),
        "spread": spread, "jf_is_outlier": predicted, "jf_fit_eligible": eligible,
        "jf_status": status, "jf_weight": weight, "jf_baseline": baseline,
        "jf_regime_change": np.zeros(length, dtype=bool),
        "true_valid_wide_side": np.zeros(length, dtype=bool),
        "evaluation_target": np.ones(length, dtype=bool),
        "downstream_fitted": np.full(length, 100.0),
        "unfiltered_fitted": np.full(length, 102.0),
        "jf_solver_converged": [None] * length, "jf_solver_iterations": np.zeros(length, dtype=int),
    })


# TEST LOGIC: Missed contamination counts even if excluded by an unsupported decision.
def test_metrics_count_abstention_and_retained_contamination():
    result = audit_case(
        [False, True, True, False, False], [100, 120, 120, 102, 105],
        [False, True, False, True, False], [True, False, True, False, False],
        ["ok", "outlier", "ok", "outlier", "insufficient_history"],
        [1, 0.1, 1, 0.5, 0], [100, 100, 100, 100, np.nan],
    )
    metrics = evaluate(result)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 2)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["scored_fraction"] == 0.8
    assert metrics["retained_fraction"] == 0.4
    assert metrics["excluded_fraction"] == 0.6
    assert metrics["clean_retained_fraction"] == pytest.approx(1 / 3)
    assert metrics["retained_rmse"] == pytest.approx(np.sqrt(200))
    assert metrics["retained_bias"] == 10.0
    assert metrics["reference_clean_coverage"] == pytest.approx(2 / 3)
    assert metrics["downstream_clean_rmse"] == 0.0
    assert metrics["unfiltered_clean_fit_rmse"] == 2.0


# TEST LOGIC: Positive weight cannot disguise an explicit unsupported/nonconverged status.
@pytest.mark.parametrize("unsupported", ["insufficient_history", "solver_not_converged", "invalid_input"])
def test_unsupported_status_excludes_fit_even_with_positive_weight(unsupported):
    result = audit_case(
        [False, True], [100, 120], [False, False], [True, True],
        ["ok", unsupported], [1, 1], [100, 110],
    )
    metrics = evaluate(result)
    assert metrics["retained_fraction"] == 0.5
    assert metrics["retained_rmse"] == 0.0
    assert metrics["fn"] == 1


# TEST LOGIC: Failed convex fits show both support loss and numerical convergence diagnostics.
def test_metric_solver_failure_remains_visible():
    result = audit_case(
        [False, False], [100, 100], [False, False], [True, False],
        ["ok", "solver_not_converged"], [1, 0], [100, np.nan],
    )
    result["jf_solver_converged"] = [True, False]
    result["jf_solver_iterations"] = [130, 500]
    metrics = evaluate(result)
    assert metrics["solver_converged_fraction"] == 0.5
    assert metrics["solver_failure_fraction"] == 0.5
    assert metrics["solver_iterations_max"] == 500


# TEST LOGIC: Evaluate rejected clean rows too, so aggressive deletion cannot hide fit error.
def test_clean_reference_error_includes_rejected_rows_and_reports_missing_estimates():
    result = audit_case(
        [False, False, False], [100, 102, 104], [False, True, False],
        [True, False, False], ["ok", "outlier", "insufficient_history"],
        [1, 0.1, 0], [100, 103, np.nan],
    )
    result["downstream_fitted"] = [100, 108, np.nan]
    metrics = evaluate(result)
    assert metrics["retained_rmse"] == 0.0
    assert metrics["reference_clean_rmse"] == pytest.approx(np.sqrt(9 / 2))
    assert metrics["reference_clean_coverage"] == pytest.approx(2 / 3)
    assert metrics["downstream_clean_rmse"] == pytest.approx(np.sqrt(64 / 2))
    assert metrics["downstream_clean_coverage"] == pytest.approx(2 / 3)


# TEST LOGIC: A clean market with no flags has undefined precision/recall, not fake perfection.
def test_clean_unflagged_case_uses_undefined_rates():
    result = audit_case(
        [False, False], [100, 101], [False, False], [True, True],
        ["ok", "ok"], [1, 1], [100, 100],
    )
    metrics = evaluate(result)
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["recall"])
    assert np.isnan(metrics["f1"])
    assert metrics["false_positive_rate"] == 0.0


# TEST LOGIC: Original regime cases preserve their uncontaminated jump-neighborhood guarantee.
def test_legacy_regime_construction_truth_and_seed_reproducibility():
    frame = make_case("regime", 104729, n=120)
    pd.testing.assert_frame_equal(frame, make_case("regime", 104729, n=120))
    assert frame.loc[59, "true_fair_spread"] == 100.0
    assert frame.loc[60, "true_fair_spread"] == 116.0
    assert not frame.loc[52:68, "true_contamination"].any()
    assert frame["true_contamination"].sum() == 4
    assert not frame["spread"].equals(make_case("regime", 130363, n=120)["spread"])


# TEST LOGIC: Additional benchmark seeds are disjoint from the old development trials.
def test_new_stress_seeds_are_held_out_from_legacy_comparison():
    assert not set(DEFAULT_SEEDS).intersection(LEGACY_SEEDS)
    assert len(set(DEFAULT_SEEDS)) == 3


# TEST LOGIC: Every construction is deterministic and isolates its economic truth fields.
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stress_case_is_reproducible_and_finite(scenario):
    frame = make_case(scenario, DEFAULT_SEEDS[0], n=240)
    repeated = make_case(scenario, DEFAULT_SEEDS[0], n=240)
    pd.testing.assert_frame_equal(frame, repeated)
    assert np.isfinite(frame["spread"]).all()
    assert frame["time"].is_monotonic_increasing
    assert not frame["time"].duplicated().any()
    assert not (frame["true_contamination"] & frame["true_valid_wide_side"]).any()


# TEST LOGIC: Forty percent means within each block, not across the entire history.
def test_clustered_stress_keeps_clean_prints_inside_each_bad_episode():
    n = 400
    positions = contamination_positions("clustered_40pct", n, np.random.default_rng(9))
    for anchor in (100, 200, 300):
        assert np.sum((positions >= anchor) & (positions < anchor + 40)) == 16
    assert len(positions) == 48
    assert np.any(np.diff(positions) > 1)


# TEST LOGIC: Real jumps and errors near their onset must not share a blanket label.
def test_jump_adjacent_stress_contains_transition_errors_and_clean_new_regime():
    frame = make_case("jump_adjacent", DEFAULT_SEEDS[0], n=240)
    assert frame.loc[118:121, "true_contamination"].all()
    assert frame.loc[119, "true_fair_spread"] == 100.0
    assert frame.loc[120, "true_fair_spread"] == 116.0
    assert not frame.loc[130:145, "true_contamination"].all()


# TEST LOGIC: All legitimate bid/ask-side prints remain clean, including rare wider ones.
def test_bimodal_case_does_not_label_market_side_as_contamination():
    frame = make_case("bidask_bimodal", DEFAULT_SEEDS[0], n=240)
    assert not frame["true_contamination"].any()
    assert frame["true_valid_wide_side"].sum() == 8
    assert (frame["spread"] < 99).any() and (frame["spread"] > 101).any()


# TEST LOGIC: The same fixed downstream fit estimates rejected targets without using labels.
def test_fixed_downstream_fit_recovers_linear_path_after_rejecting_target():
    frame = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC"),
        "spread": 100 + 0.5 * np.arange(20),
    })
    retained = np.ones(20, dtype=bool)
    retained[9] = False
    frame.loc[9, "spread"] = 999
    fit = downstream_fit(frame, retained)
    np.testing.assert_allclose(fit, 100 + 0.5 * np.arange(20), atol=1e-10)


# TEST LOGIC: A support-breaking gap must remain separate for every supported pandas timestamp resolution.
@pytest.mark.parametrize("timestamp_unit", ["s", "ms", "us", "ns"])
def test_fixed_downstream_fit_reports_short_postgap_segment_as_missing(timestamp_unit):
    frame = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=9, freq="h", tz="UTC"),
        "spread": np.r_[np.full(6, 100.0), np.full(3, 120.0)],
    })
    frame.loc[6:, "time"] += pd.Timedelta("10D")
    frame["time"] = frame["time"].dt.as_unit(timestamp_unit)
    fit = downstream_fit(frame, np.ones(9, dtype=bool))
    np.testing.assert_allclose(fit[:6], 100.0)
    assert np.isnan(fit[6:]).all()


# TEST LOGIC: Benchmark method subsets remain explicit and do not change construction truth.
def test_subset_comparison_records_fit_errors_and_method_configuration():
    result = run_comparison((DEFAULT_SEEDS[0],), 120, ("hampel",), ("clean",))
    assert len(result) == 1
    assert result.loc[0, "method"] == "hampel"
    assert result.loc[0, "downstream_clean_coverage"] > 0.9
    assert result.loc[0, "evaluation_target_n"] == 120


# TEST LOGIC: Misspelled subsets should fail before expensive trials start.
@pytest.mark.parametrize("methods,scenarios", [(("wrong",), ("clean",)), (("hampel",), ("wrong",))])
def test_invalid_benchmark_subsets_raise(methods, scenarios):
    with pytest.raises(ValueError):
        run_comparison((DEFAULT_SEEDS[0],), 120, methods, scenarios)


# TEST LOGIC: A slope reversal and an abrupt drop retain distinct efficient-market truth.
def test_turning_stresses_preserve_continuity_or_exact_level_drop():
    continuous = make_case("turning_continuous", DEFAULT_SEEDS[0], n=240)
    abrupt = make_case("turning_drop", DEFAULT_SEEDS[0], n=240)
    assert continuous.loc[120, "true_fair_spread"] - continuous.loc[119, "true_fair_spread"] == pytest.approx(0.3)
    assert continuous.loc[121, "true_fair_spread"] - continuous.loc[120, "true_fair_spread"] == pytest.approx(-0.5)
    assert abrupt.loc[120, "true_fair_spread"] - abrupt.loc[119, "true_fair_spread"] == pytest.approx(-19.7)
    assert abrupt.loc[121, "true_fair_spread"] - abrupt.loc[120, "true_fair_spread"] == pytest.approx(-0.25)
    assert abrupt.loc[118:121, "true_contamination"].all()


# TEST LOGIC: Business-session construction stays inside explicitly stated New York hours.
def test_calendar_stress_has_irregular_business_hours_and_weekend_closures():
    frame = make_case("calendar_sessions", DEFAULT_SEEDS[0], n=240)
    local = frame["time"].dt.tz_convert("America/New_York")
    assert local.dt.dayofweek.lt(5).all()
    minute = local.dt.hour * 60 + local.dt.minute
    assert minute.ge(8 * 60).all() and minute.lt(18 * 60 + 30).all()
    wall_gap = frame["time"].diff().dt.total_seconds().div(3600)
    clock_gap = frame["evaluation_hours"].diff()
    assert wall_gap.max() > 60.0
    assert clock_gap.max() < 10.5
    assert wall_gap.gt(12).any()
    assert frame["time"].diff().nunique() > 10


# TEST LOGIC: Changes in cadence and variance must remain different from contamination labels.
def test_liquidity_stress_changes_density_without_labeling_whole_regimes_bad():
    frame = make_case("liquidity_shift", DEFAULT_SEEDS[0], n=300)
    gap = frame["time"].diff().dt.total_seconds().div(3600)
    assert gap.iloc[1:100].median() == pytest.approx(2.0)
    assert gap.iloc[110:190].median() == pytest.approx(1 / 60)
    assert gap.iloc[210:].median() == pytest.approx(8.0)
    clean = ~frame["true_contamination"]
    error = frame["spread"] - frame["true_fair_spread"]
    assert error.loc[200:299].loc[clean.loc[200:299]].std() > 5 * error.loc[100:199].loc[clean.loc[100:199]].std()


# TEST LOGIC: Compressed closures cannot imply that efficient value freezes overnight.
def test_calendar_stress_contains_clean_monday_opening_repricing():
    frame = make_case("calendar_sessions", DEFAULT_SEEDS[0], n=240)
    local = frame["time"].dt.tz_convert("America/New_York")
    assert local.iloc[40] == pd.Timestamp("2025-01-06 08:00", tz="America/New_York")
    assert frame.loc[39, "true_fair_spread"] == 100.0
    assert frame.loc[40, "true_fair_spread"] == 120.0
    assert not frame.loc[40, "true_contamination"]
    assert frame.loc[70, "true_contamination"]
