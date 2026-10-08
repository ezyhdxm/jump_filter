"""Optional decision rules preserve evidence, strict boundaries and source integrity."""

# SETUP LOGIC: small fixtures exercise the public API without private trade data.
import json
import numpy as np
import pandas as pd
import pytest
from jump_filter import FilterConfig, filter_trades, select_fit_data, summarize
from jump_filter.policies import _quantity_values


def source(size=51, level=100.0):
    # TEST LOGIC: one quantity column has an arbitrary name and raw notional units.
    return pd.DataFrame({"bond": "A", "stamp": pd.date_range("2026-09-14", periods=size, freq="h"),
                         "oas": np.full(size, level, dtype=float), "ticket_notional": np.full(size, 500000.0)})


def apply(frame, config, **options):
    # TEST LOGIC: exercise all column mappings; the original index is never an identity key.
    return filter_trades(frame, config, cusip_col="bond", time_col="stamp", spread_col="oas",
                         quantity_col="ticket_notional", **options)


@pytest.mark.parametrize("quantity,expected", [(999999.0, "policy_outlier"),
                                               (1000000.0, "retained_uncertain"),
                                               (2000000.0, "retained_uncertain")])
def test_small_suspicious_edge_and_exact_million_boundary(quantity, expected):
    # TEST LOGIC: the edge lacks one side's confirmation, while its independent trend is supported.
    frame = source()
    frame.loc[1, ["oas", "ticket_notional"]] = [130, quantity]
    result = apply(frame, FilterConfig(quantity_rule=True), backend="python")
    row = result.iloc[1]
    assert row.jf_algorithm_status == "insufficient_history"
    assert row.jf_status == expected
    assert row.jf_policy_suspicious
    assert row.jf_support_reliable
    assert row.jf_fit_eligible == (expected == "retained_uncertain")
    assert row.jf_algorithm_fit_eligible == False


def test_strong_outliers_of_any_size_keep_the_algorithm_decision():
    # TEST LOGIC: the quantity rule never protects a confirmed central outlier.
    frame = source()
    frame.loc[25, ["oas", "ticket_notional"]] = [130, 5000000]
    result = apply(frame, FilterConfig(quantity_rule=True))
    row = result.iloc[25]
    assert row.jf_algorithm_status == row.jf_status == "outlier"
    assert row.jf_is_outlier and not row.jf_fit_eligible
    assert row.jf_policy_reason == "no_override"
    assert not row.jf_policy_retained_uncertain


def test_weak_bridge_artifact_cannot_reject_a_neighboring_clean_trade():
    # TEST LOGIC: the edge outlier biases a weak consensus bridge at row2; independent support clears row2.
    frame = source()
    frame.loc[1, "oas"] = 130
    result = apply(frame, FilterConfig(quantity_rule=True))
    suspicious, neighbor = result.iloc[1], result.iloc[2]
    assert suspicious.jf_status == "policy_outlier"
    assert neighbor.jf_algorithm_status == "insufficient_history"
    assert abs(neighbor.jf_residual) > neighbor.jf_threshold
    assert neighbor.jf_support_reliable
    assert not neighbor.jf_policy_suspicious
    assert neighbor.jf_status == "retained_uncertain"
    assert neighbor.jf_policy_suspicious_source == "two_sided_support_trend"


@pytest.mark.parametrize("quantity", [None, np.nan, np.inf, -1, 0, "invalid"])
def test_unknown_quantity_cannot_trigger_weak_retention_or_rejection(quantity):
    # TEST LOGIC: size is not guessed from missing/nonpositive/nonfinite records.
    frame = source()
    frame["ticket_notional"] = frame["ticket_notional"].astype(object)
    frame.loc[1, ["oas", "ticket_notional"]] = [130, quantity]
    row = apply(frame, FilterConfig(quantity_rule=True)).iloc[1]
    assert pd.isna(row.jf_quantity) and not row.jf_quantity_valid
    assert row.jf_status == "insufficient_history" and not row.jf_fit_eligible
    assert row.jf_policy_reason == "quantity_unknown_no_override"


@pytest.mark.parametrize("multiplier,raw", [(1, 1000000), (1000, 1000), (1000000, 1)])
def test_explicit_quantity_units_have_identical_decisions(multiplier, raw):
    # TEST LOGIC: amount/thousands/millions use explicit conversion, with an exact boundary.
    frame = source()
    frame.loc[1, ["oas", "ticket_notional"]] = [130, raw]
    row = apply(frame, FilterConfig(quantity_rule=True, quantity_multiplier=multiplier)).iloc[1]
    assert row.jf_quantity == 1000000
    assert not row.jf_policy_small_quantity
    assert row.jf_status == "retained_uncertain"


def test_small_without_suspicious_evidence_is_retained_with_uncertainty():
    # TEST LOGIC: a sparse two-trade bond has no statistical reference; positive size alone is not suspicious.
    result = apply(source(2), FilterConfig(quantity_rule=True))
    assert result.jf_algorithm_status.eq("insufficient_history").all()
    assert not result.jf_policy_suspicious.any()
    assert result.jf_status.eq("retained_uncertain").all()
    assert result.jf_fit_eligible.all()
    assert result.jf_algorithm_fit_eligible.eq(False).all()
    assert result.jf_support_reliable.eq(False).all()


@pytest.mark.parametrize("quantity,expected", [(500000, "policy_outlier"), (1000000, "retained_uncertain")])
def test_history_supported_causal_provisional_jump_can_use_explicit_fallback(quantity, expected):
    # TEST LOGIC: the final print has no future bracket but a supported causal history.
    frame = source(31)
    frame.loc[30, ["oas", "ticket_notional"]] = [130, quantity]
    result = apply(frame, FilterConfig(method="causal_ewma", quantity_rule=True))
    row = result.iloc[-1]
    assert row.jf_algorithm_status == "provisional_jump"
    assert not row.jf_support_reliable
    assert row.jf_policy_suspicious_source == "causal_history"
    assert row.jf_status == expected
    assert result.attrs["jump_filter"]["future_observations"]
    assert not result.attrs["jump_filter"]["algorithm_future_observations"]
    assert result.attrs["jump_filter"]["support_method"] == "two_sided_piecewise_with_bracketed_linear_fallback"


@pytest.mark.parametrize("quantity", [None, 500000, 2000000])
def test_absolute_cap_takes_precedence_over_size_and_uncertain_retention(quantity):
    # TEST LOGIC: an edge consensus abstention still has a bracketed, reliable independent trend.
    frame = source()
    frame["ticket_notional"] = frame.ticket_notional.astype(object)
    frame.loc[1, ["oas", "ticket_notional"]] = [130, quantity]
    result = apply(frame, FilterConfig(quantity_rule=True, max_deviation_rule=True))
    row = result.iloc[1]
    assert row.jf_algorithm_status == "insufficient_history"
    assert row.jf_policy_max_deviation
    assert row.jf_status == "policy_outlier" and not row.jf_fit_eligible
    assert row.jf_policy_reason == "support_trend_max_deviation_exceeded"


def test_maximum_distance_is_strict_and_can_override_accepted_algorithm_rows():
    # TEST LOGIC: a deliberately wide statistical band isolates the user-selected absolute cap.
    frame = source(71, 0)
    frame.loc[20, "oas"], frame.loc[50, "oas"] = 10, 10.001
    config = FilterConfig(method="hampel", abs_floor=100, max_deviation_rule=True)
    result = apply(frame, config)
    assert result.loc[[20, 50], "jf_algorithm_status"].eq("ok").all()
    assert result.iloc[20].jf_support_distance_bps == 10
    assert result.iloc[20].jf_status == "ok"
    assert result.iloc[50].jf_status == "policy_outlier"
    assert not result.iloc[50].jf_fit_eligible
    assert result.iloc[50].jf_weight == 0
    assert 50 not in select_fit_data(result, policy="soft").index


@pytest.mark.parametrize("sign", [-1, 1])
def test_cap_symmetric_boundary_and_just_above_survive_fit_roundoff(sign):
    # TEST LOGIC: nonzero intercepts exercise numerical errors on both sides of the boundary.
    frame = source(91, 100)
    frame.loc[20, "oas"] = 100 + sign * 10
    frame.loc[60, "oas"] = 100 + sign * (10 + 1e-10)
    result = apply(frame, FilterConfig(method="local_linear", abs_floor=100, max_deviation_rule=True))
    assert not result.iloc[20].jf_policy_max_deviation
    assert result.iloc[20].jf_status == "ok"
    assert result.iloc[60].jf_policy_max_deviation
    assert result.iloc[60].jf_status == "policy_outlier"
    assert result.iloc[60].jf_support_distance_tolerance_bps < 1e-10


@pytest.mark.parametrize("units_per_bp", [1, .01, .0001])
def test_spread_unit_conversion_preserves_cap_decisions(units_per_bp):
    # TEST LOGIC: basis points, percentage points and decimal spreads use identical economic distances.
    frame = source(71, 0)
    frame.loc[20, "oas"], frame.loc[50, "oas"] = 10 * units_per_bp, 10.01 * units_per_bp
    config = FilterConfig(method="local_linear", abs_floor=100 * units_per_bp,
                          max_deviation_rule=True, spread_units_per_bp=units_per_bp)
    result = apply(frame, config)
    assert result.iloc[20].jf_status == "ok"
    assert result.iloc[50].jf_status == "policy_outlier"
    assert result.iloc[50].jf_support_distance_bps == pytest.approx(10.01)


def test_quantity_rule_does_not_replace_accepted_rows_by_a_new_method():
    # TEST LOGIC: a wide Hampel band explicitly accepts 12bp; quantity alone respects that decision.
    frame = source(71, 0)
    frame.loc[30, "oas"] = 12
    result = apply(frame, FilterConfig(method="hampel", abs_floor=100, quantity_rule=True))
    assert result.iloc[30].jf_algorithm_status == result.iloc[30].jf_status == "ok"
    assert result.iloc[30].jf_fit_eligible
    assert result.iloc[30].jf_policy_reason == "no_override"


def test_support_trend_cannot_extrapolate_or_cross_cusips_and_gap_segments():
    # TEST LOGIC: even an extreme endpoint with many future prints has no past bracket.
    frame = source(51, 0)
    frame.loc[0, "oas"] = 1000
    first = apply(frame, FilterConfig(method="local_linear", max_deviation_rule=True))
    assert not first.iloc[0].jf_support_reliable
    assert first.iloc[0].jf_support_reason == "no_past_and_future_bracket"
    assert not first.iloc[0].jf_policy_max_deviation
    second = frame.copy()
    second["bond"], second["oas"] = "B", 10000
    combined = apply(pd.concat([second, frame]), FilterConfig(method="local_linear", max_deviation_rule=True))
    fields = [name for name in first.columns if name.startswith("jf_") and name != "jf_row_id"]
    pd.testing.assert_frame_equal(first[fields].reset_index(drop=True),
                                  combined.loc[combined.bond.eq("A"), fields].reset_index(drop=True))
    gapped = source(6)
    gapped.loc[3:, "stamp"] += pd.Timedelta("20D")
    result = apply(gapped, FilterConfig(max_gap="2h", max_deviation_rule=True))
    assert not result.jf_support_reliable.any()


def test_legitimate_persistent_level_change_is_not_a_reliable_cap_reference():
    # TEST LOGIC: a straight line through a genuine step is uncertain, even with a 15bp residual.
    frame = source(71)
    frame.loc[35:, "oas"] = 130
    result = apply(frame, FilterConfig(method="local_linear", max_deviation_rule=True))
    transition = result.loc[32:37]
    assert transition.jf_support_reason.eq("protected_transition").all()
    assert not transition.jf_support_reliable.any()
    assert not transition.jf_policy_max_deviation.any()


def test_shape_aware_support_preserves_clean_corner_and_detects_injected_peak_trade():
    # TEST LOGIC: the user's steep up-then-down shape must not be flattened into a false 10bp violation.
    frame = source(91)
    frame["oas"] = 100 - 4 * np.abs(np.arange(91) - 45)
    frame["ticket_notional"] = 2000000
    config = FilterConfig(method="local_piecewise", max_deviation_rule=True, quantity_rule=True)
    clean = apply(frame, config)
    assert not clean.jf_policy_max_deviation.any()
    assert clean.iloc[45].jf_support_trend == pytest.approx(100)
    assert clean.iloc[45].jf_support_reliable
    assert clean.iloc[45].jf_support_method == "local_piecewise"
    frame.loc[45, "oas"] += 15
    injected = apply(frame, config)
    assert injected.iloc[45].jf_support_trend == pytest.approx(100)
    assert injected.iloc[45].jf_policy_max_deviation
    assert injected.iloc[45].jf_status == "policy_outlier"
    assert not injected.jf_policy_max_deviation.drop(index=45).any()


@pytest.mark.parametrize("method", ["local_linear", "consensus", "hampel"])
def test_support_shape_is_independent_of_the_selected_algorithm(method):
    # TEST LOGIC: optional cap evidence always preserves a genuine turning point even if an algorithm flattens it.
    frame = source(91)
    frame["oas"] = (100 - 4 * np.abs(np.arange(91) - 45)).astype(float)
    config = FilterConfig(method=method, max_deviation_rule=True)
    clean = apply(frame, config)
    assert not clean.jf_policy_max_deviation.any()
    assert clean.iloc[45].jf_support_trend == pytest.approx(100)
    assert clean.iloc[45].jf_support_method == "local_piecewise"


def test_retention_never_certifies_invalid_outside_session_or_failed_solver_rows():
    # TEST LOGIC: high size does not override operational validity or optimizer failure.
    frame = source()
    frame["ticket_notional"] = 2000000
    frame.loc[0, "stamp"] = pd.NaT
    frame.loc[1, "stamp"] = pd.Timestamp("2026-09-19T14:00Z").tz_localize(None)
    result = apply(frame, FilterConfig(quantity_rule=True, max_deviation_rule=True, time_basis="trading"))
    assert result.iloc[0].jf_status == "invalid_input"
    assert result.iloc[1].jf_status == "outside_session"
    assert not result.iloc[:2].jf_fit_eligible.any()
    failed = source(90)
    failed["oas"] = 100 + .4 * np.sin(np.arange(90) / 4)
    failed.loc[30, "oas"] += 25
    failed.loc[60:, "oas"] += 12
    config = FilterConfig(method="robust_trend", max_iter=1, tolerance=1e-12,
                          quantity_rule=True, max_deviation_rule=True)
    result = apply(failed, config)
    failure = result.jf_algorithm_status.eq("solver_not_converged")
    assert failure.any()
    assert result.loc[failure, "jf_status"].eq("solver_not_converged").all()
    assert not result.loc[failure, "jf_fit_eligible"].any()


def test_policy_audit_survives_duplicate_indexes_and_does_not_mutate_source():
    # TEST LOGIC: original rows/order/dtypes remain byte-for-byte pandas-equivalent.
    frame = source().sample(frac=1, random_state=11)
    frame.index = np.arange(len(frame)) // 2
    original = frame.copy(deep=True)
    result = apply(frame, FilterConfig(quantity_rule=True))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.jf_row_id.is_unique
    assert result.attrs["jump_filter"]["quantity_col"] == "ticket_notional"


def test_disabled_optional_rules_preserve_original_fields_and_decisions():
    # TEST LOGIC: merely selecting a quantity mapping adds no policy columns or new classifications.
    frame = source()
    frame.loc[25, "oas"] = 130
    baseline = filter_trades(frame, cusip_col="bond", time_col="stamp", spread_col="oas")
    result = apply(frame, FilterConfig())
    pd.testing.assert_frame_equal(baseline, result)
    assert not any(name.startswith("jf_support_") or name.startswith("jf_algorithm_") for name in result.columns)


def test_summary_keeps_algorithm_coverage_separate_from_policy_fit_rows():
    # TEST LOGIC: retained uncertainty increases fitting rows while zero evaluated support stays zero.
    result = apply(source(2), FilterConfig(quantity_rule=True))
    row = summarize(result, cusip_col="bond").iloc[0]
    assert row.evaluated == 0 and row.coverage == 0
    assert row.fit_eligible == row.retained_uncertain == 2
    assert pd.isna(row.flagged_rate)
    assert len(select_fit_data(result)) == len(select_fit_data(result, policy="soft")) == 2


@pytest.mark.parametrize("method", ["hampel", "local_linear", "consensus", "local_piecewise"])
def test_enabled_policies_keep_compiled_and_python_backend_parity(method):
    # TEST LOGIC: independent references and final rules use the same evidence on both backends.
    pytest.importorskip("numba")
    frame = source(90)
    frame.loc[[1, 25, 60], "oas"] = [130, 115, 80]
    config = FilterConfig(method=method, quantity_rule=True, max_deviation_rule=True)
    compiled, reference = apply(frame, config, backend="numba"), apply(frame, config, backend="python")
    for name in compiled.columns:
        if pd.api.types.is_float_dtype(compiled[name]):
            np.testing.assert_allclose(compiled[name], reference[name], rtol=2e-9, atol=2e-8, equal_nan=True)
        else:
            pd.testing.assert_series_equal(compiled[name], reference[name])


@pytest.mark.parametrize("changes", [{"quantity_rule": 1}, {"max_deviation_rule": "yes"},
                                      {"require_cap_support": 1}, {"require_cap_support": "yes"},
                                      {"quantity_threshold": 0}, {"quantity_multiplier": -1},
                                      {"quantity_multiplier": np.inf}, {"max_deviation_bps": True},
                                      {"max_deviation_bps": np.nan}, {"spread_units_per_bp": 0}])
def test_invalid_policy_configuration_raises(changes):
    # TEST LOGIC: units/threshold/toggle errors fail before reading or scoring any trade.
    with pytest.raises(ValueError):
        FilterConfig(**changes)


def test_quantity_mapping_validation_is_explicit():
    # TEST LOGIC: enabled size policies cannot silently substitute another source column.
    frame = source()
    config = FilterConfig(quantity_rule=True)
    with pytest.raises(ValueError, match="quantity_col"):
        filter_trades(frame, config, cusip_col="bond", time_col="stamp", spread_col="oas")
    with pytest.raises(ValueError, match="quantity_col"):
        filter_trades(frame, config, cusip_col="bond", time_col="stamp", spread_col="oas", quantity_col="oas")


def test_cli_maps_arbitrary_quantity_column(tmp_path, monkeypatch):
    # TEST LOGIC: public file adapter reproduces an exact-1MM uncertain retention.
    from jump_filter.cli import main
    frame = source()
    frame.loc[1, ["oas", "ticket_notional"]] = [130, 1000000]
    input_path, output_path, config_path = tmp_path / "input.csv", tmp_path / "output.csv", tmp_path / "config.json"
    frame.to_csv(input_path, index=False)
    config_path.write_text(json.dumps({"quantity_rule": True}))
    monkeypatch.setattr("sys.argv", ["jump-filter", str(input_path), str(output_path), "--config", str(config_path),
                                    "--cusip-col", "bond", "--time-col", "stamp", "--spread-col", "oas",
                                    "--quantity-col", "ticket_notional", "--backend", "python"])
    main()
    result = pd.read_csv(output_path)
    assert result.iloc[1].jf_status == "retained_uncertain"
    assert result.iloc[1].jf_quantity == 1000000


def test_quantity_conversion_reports_overflow_as_unknown():
    # TEST LOGIC: conversion overflow is masked and cannot become a large-trade exemption.
    frame = pd.DataFrame({"size": [1e308, 1, np.nan]})
    values, valid = _quantity_values(frame, "size", 1000000)
    np.testing.assert_allclose(values, [np.nan, 1000000, np.nan], equal_nan=True)
    np.testing.assert_array_equal(valid, [False, True, False])


def test_cap_audit_distinguishes_disabled_from_unassessed():
    # TEST LOGIC: quantity-only screening must never imply the 10bp check ran.
    result = apply(source(5), FilterConfig(quantity_rule=True), backend="python")
    assert not result.jf_policy_cap_assessed.any()
    assert result.jf_policy_cap_status.eq("disabled").all()
    assert result.jf_policy_retained_uncertain.all()


def test_sparse_retained_points_do_not_claim_a_verified_cap():
    # TEST LOGIC: five events cannot provide the required six leave-target-out references.
    frame = source(5)
    frame.loc[1, "oas"] = 140
    result = apply(frame, FilterConfig(quantity_rule=True, max_deviation_rule=True), backend="python")
    assert result.jf_fit_eligible.all()
    assert result.jf_policy_retained_uncertain.all()
    assert result.jf_support_trend.isna().all()
    assert not result.jf_policy_cap_assessed.any()
    assert result.jf_policy_cap_status.eq("no_reliable_support").all()


@pytest.mark.parametrize("require_cap_support", [False, True])
def test_cap_audit_reports_strict_boundary_and_rejection(require_cap_support):
    # TEST LOGIC: a wide statistical threshold isolates exact10bp and strictly larger10.001bp.
    frame = source(71, 0)
    frame.loc[20, "oas"], frame.loc[50, "oas"] = 10, 10.001
    result = apply(frame, FilterConfig(method="hampel", abs_floor=100, max_deviation_rule=True,
                                      require_cap_support=require_cap_support), backend="python")
    assert result.loc[[20, 50], "jf_policy_cap_assessed"].all()
    assert result.iloc[20].jf_policy_cap_status == "within_limit"
    assert result.iloc[50].jf_policy_cap_status == "exceeded"
    assert result.iloc[20].jf_fit_eligible and not result.iloc[50].jf_fit_eligible


def test_invalid_input_is_not_an_unverified_actionable_cap_point():
    # TEST LOGIC: operationally invalid rows remain distinguishable from a valid unsupported trade.
    frame = source()
    frame.loc[0, "stamp"] = pd.NaT
    result = apply(frame, FilterConfig(quantity_rule=True, max_deviation_rule=True), backend="python")
    assert result.iloc[0].jf_policy_cap_status == "not_actionable"
    assert not result.iloc[0].jf_policy_cap_assessed
    assert not result.iloc[0].jf_fit_eligible


def test_protected_jump_diagnostic_distance_is_not_an_assessed_cap():
    # TEST LOGIC: two genuine regimes100/130 have a misleading midpoint115; each real print remains valid evidence.
    frame = source(71)
    frame.loc[35:, "oas"] = 130
    result = apply(frame, FilterConfig(method="local_piecewise", quantity_rule=True,
                                      max_deviation_rule=True), backend="python")
    transition = result.loc[32:37]
    assert (transition.jf_support_distance_bps > 10).all()
    assert transition.jf_policy_retained_uncertain.all()
    assert transition.jf_fit_eligible.all()
    assert transition.jf_support_reason.eq("protected_transition").all()
    assert transition.jf_policy_cap_status.eq("no_reliable_support").all()
    assert not transition.jf_policy_cap_assessed.any()


@pytest.mark.parametrize("method", ["hampel", "rolling_iqr", "local_linear", "jump_reversion",
                                    "multiscale", "local_piecewise", "robust_trend", "consensus", "causal_ewma"])
def test_reliable_cap_has_no_fit_eligible_violations_across_methods(method):
    # TEST LOGIC: irregular hours, liquidity variation, a long inactivity gap and isolated spikes share one invariant.
    frame = source(91)
    spacing = np.resize(np.array([7, 13, 41, 61, 11], dtype=np.int64), len(frame))
    frame["stamp"] = pd.Timestamp("2026-09-14T09:00") + pd.to_timedelta(np.cumsum(spacing), unit="m")
    frame.loc[45:, "stamp"] += pd.Timedelta("4D")
    frame["oas"] = 100 + .1 * np.arange(len(frame))
    frame.loc[[1, 20, 65, 89], "oas"] += np.array([21, -25, 30, -19])
    frame["ticket_notional"] = 2000000
    result = apply(frame, FilterConfig(method=method, quantity_rule=True, max_deviation_rule=True), backend="python")
    verified = result.jf_policy_cap_assessed
    exceeds = result.jf_support_distance_bps > 10 + result.jf_support_distance_tolerance_bps
    assert verified.any() and result.jf_policy_max_deviation.any()
    assert not (result.jf_fit_eligible & verified & exceeds).any()
    assert result.loc[verified & exceeds, "jf_policy_cap_status"].eq("exceeded").all()
    assert result.loc[verified & ~exceeds, "jf_policy_cap_status"].eq("within_limit").all()
    assert not result.loc[[0, 44, 45, 90], "jf_policy_cap_assessed"].any()


def test_strict_cap_excludes_unverified_sparse_trades_without_claiming_outliers():
    # TEST LOGIC: insufficient evidence fails strict fitting certification but does not establish an anomaly.
    frame = source(5)
    frame.loc[1, "oas"] = 140
    frame["ticket_notional"] = 2000000
    result = apply(frame, FilterConfig(quantity_rule=True, max_deviation_rule=True,
                                      require_cap_support=True), backend="python")
    assert result.jf_status.eq("unverified_support").all()
    assert result.jf_algorithm_status.eq("insufficient_history").all()
    assert not result.jf_is_outlier.any() and not result.jf_fit_eligible.any()
    assert result.jf_weight.eq(0).all()
    assert result.jf_policy_support_unverified.all()
    assert not result.jf_policy_retained_uncertain.any()
    assert result.jf_policy_reason.eq("support_trend_unverified_for_fitting").all()
    assert select_fit_data(result).empty
    assert select_fit_data(result, policy="soft", include_provisional=True).empty


def test_strict_cap_preserves_original_outlier_evidence_and_blocks_soft_rescue():
    # TEST LOGIC: an endpoint130 can be a one-sided algorithm outlier while lacking strict two-sided support.
    frame = source()
    frame.loc[0, "oas"] = 130
    result = apply(frame, FilterConfig(method="local_linear", max_deviation_rule=True,
                                      require_cap_support=True), backend="python")
    row = result.iloc[0]
    assert row.jf_algorithm_status == row.jf_status == "outlier"
    assert row.jf_algorithm_is_outlier and row.jf_is_outlier
    assert row.jf_algorithm_weight > 0 and row.jf_weight == 0
    assert not row.jf_fit_eligible and row.jf_policy_support_unverified
    assert 0 not in select_fit_data(result, policy="soft").index


def test_strict_support_gate_has_no_effect_while_cap_is_disabled():
    # TEST LOGIC: the strict sub-control belongs to the cap; toggling it alone must preserve quantity decisions.
    frame = source(5)
    baseline = apply(frame, FilterConfig(quantity_rule=True), backend="python")
    strict_off = apply(frame, FilterConfig(quantity_rule=True, require_cap_support=True), backend="python")
    pd.testing.assert_frame_equal(baseline, strict_off)
    assert strict_off.jf_fit_eligible.all()
    assert not strict_off.jf_policy_support_unverified.any()


@pytest.mark.parametrize("method", ["consensus", "local_piecewise"])
def test_strict_cap_keeps_turning_point_and_abstains_across_genuine_step(method):
    # TEST LOGIC: a continuous100bp peak is certified; the conflicting100/130 regimes are unverified rather than bad prints.
    frame = source(91)
    frame["oas"] = (100 - 4 * np.abs(np.arange(91) - 45)).astype(float)
    frame["ticket_notional"] = 2000000
    config = FilterConfig(method=method, quantity_rule=True, max_deviation_rule=True, require_cap_support=True)
    corner = apply(frame, config, backend="python")
    assert corner.iloc[45].jf_policy_cap_status == "within_limit"
    assert corner.iloc[45].jf_fit_eligible
    assert not corner.jf_policy_max_deviation.any()
    step = source(71)
    step.loc[35:, "oas"] = 130
    transition = apply(step, config, backend="python").loc[32:37]
    assert transition.jf_policy_support_unverified.all()
    assert transition.jf_status.eq("unverified_support").all()
    assert not transition.jf_is_outlier.any()
    assert not transition.jf_fit_eligible.any() and transition.jf_weight.eq(0).all()


@pytest.mark.parametrize("method", ["hampel", "rolling_iqr", "local_linear", "jump_reversion",
                                    "multiscale", "local_piecewise", "robust_trend", "consensus", "causal_ewma"])
def test_strict_fit_rows_are_all_certified_inside_cap_across_methods(method):
    # TEST LOGIC: a 30bp spike, endpoints and a four-day gap challenge strict certification for every method.
    frame = source(71)
    frame["ticket_notional"] = 2000000
    frame.loc[25, "oas"] = 130
    frame.loc[35:, "stamp"] += pd.Timedelta("4D")
    result = apply(frame, FilterConfig(method=method, quantity_rule=True, max_deviation_rule=True,
                                      require_cap_support=True), backend="python")
    eligible = result.jf_fit_eligible
    assert eligible.any()
    assert result.loc[eligible, "jf_policy_cap_assessed"].all()
    assert result.loc[eligible, "jf_policy_cap_status"].eq("within_limit").all()
    distance = result.loc[eligible, "jf_support_distance_bps"]
    tolerance = result.loc[eligible, "jf_support_distance_tolerance_bps"]
    assert (distance <= 10 + tolerance).all()
    soft = select_fit_data(result, policy="soft", include_provisional=True)
    assert soft.jf_policy_cap_assessed.all()
    assert soft.jf_policy_cap_status.eq("within_limit").all()
    assert not result.loc[[0, 34, 35, 70], "jf_fit_eligible"].any()


def test_strict_cap_uses_active_time_across_weekend_and_excludes_closed_events():
    # TEST LOGIC: Friday18:25NY to Monday08:00NY spans5 active minutes; Saturday10:00NY is outside the calendar.
    friday = pd.date_range("2026-09-18T17:00", periods=18, freq="5min", tz="America/New_York")
    monday = pd.date_range("2026-09-21T08:00", periods=18, freq="5min", tz="America/New_York")
    frame = source(37)
    frame["stamp"] = pd.DatetimeIndex([*friday, *monday, pd.Timestamp("2026-09-19T10:00", tz="America/New_York")])
    frame["ticket_notional"] = 2000000
    frame.loc[[8, 26], "oas"] = [125, 75]
    config = FilterConfig(time_basis="trading", horizon="3h", max_gap="30min", quantity_rule=True,
                          max_deviation_rule=True, require_cap_support=True)
    result = apply(frame, config, backend="python")
    assert result.loc[[17, 18], "jf_policy_cap_assessed"].all()
    assert result.loc[[17, 18], "jf_fit_eligible"].all()
    assert result.iloc[18].jf_session_boundary
    assert result.loc[[8, 26], "jf_policy_max_deviation"].all()
    assert result.iloc[36].jf_status == "outside_session"
    assert result.iloc[36].jf_policy_cap_status == "not_actionable"
    assert not result.iloc[36].jf_fit_eligible
