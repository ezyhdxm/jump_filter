"""Independent mathematical checks for retrospective quantiles and Huber TV."""

# SETUP LOGIC: tiny configurations isolate solver math from engine orchestration.
from types import SimpleNamespace
import numpy as np
import pytest
from jump_filter.offline import (NORMAL_IQR, _difference_adjoint, _factor_tv_system,
                                 _huber_tv, _local_residual_scale, _solve_tv_system,
                                 iqr_point, robust_trend_segment)


# TEST LOGIC: construct controls in input spread units with deterministic solver limits.
def controls(**changes):
    values = dict(iqr_multiplier=3.0, abs_floor=1.0, threshold=4.5, window=31,
                  horizon="3D", trend_penalty=8.0, huber_delta=2.5,
                  max_iter=2000, tolerance=1e-5)
    values.update(changes)
    return SimpleNamespace(**values)


# TEST LOGIC: the raw fence equals Tukey's asymmetric quartile notation exactly.
def test_iqr_fences_and_strict_boundary():
    refs = np.array([98, 99, 100, 101, 102])
    center, scale, score, cutoff, flag = iqr_point(refs, 110, controls())
    assert center == 100
    assert cutoff == 7
    assert center - cutoff == 99 - 3 * 2
    assert center + cutoff == 101 + 3 * 2
    assert scale == pytest.approx(2 / NORMAL_IQR)
    assert score == pytest.approx(10 / scale)
    assert flag
    assert not iqr_point(refs, 107, controls())[-1]
    assert not iqr_point(refs, 93, controls())[-1]


# TEST LOGIC: constant references use a dimensional floor without division by zero.
def test_iqr_floor_and_threshold_independence():
    first = iqr_point(np.ones(20) * 100, 101, controls(threshold=1.5))
    second = iqr_point(np.ones(20) * 100, 101, controls(threshold=9))
    assert first == second
    assert first[1] > 0
    assert first[3] == 1
    assert not first[-1]
    assert iqr_point(np.ones(20) * 100, 101.00001, controls())[-1]


# TEST LOGIC: solve a dense reference system only in the test, independent of Thomas recurrences.
@pytest.mark.parametrize("size", [1, 2, 3, 17, 63])
def test_factored_solve_matches_dense_matrix(size):
    identity = np.eye(size)
    difference = np.diff(identity, axis=0)
    matrix = identity + difference.T @ difference
    rhs = np.random.default_rng(size).normal(size=size)
    actual = _solve_tv_system(_factor_tv_system(size), rhs)
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), atol=1e-12)


# TEST LOGIC: verify the boundary signs using the definition of the adjoint.
def test_difference_adjoint_inner_product():
    baseline = np.array([2.0, -1.0, 4.0, 5.0])
    vector = np.array([-2.0, 3.0, 0.5])
    assert np.dot(np.diff(baseline), vector) == pytest.approx(np.dot(baseline, _difference_adjoint(vector, 4)))
    np.testing.assert_array_equal(_difference_adjoint(np.array([]), 1), [0])


# TEST LOGIC: an analytical Huber-location solution certifies the large-TV optimum.
def test_huber_tv_has_known_constant_solution_and_objective():
    baseline, diagnostics = _huber_tv(np.array([0.0, 10.0, 0.0]), 8.0, 2.5, 2000, 1e-9)
    np.testing.assert_allclose(baseline, [1.25, 1.25, 1.25], atol=2e-7)
    np.testing.assert_allclose(diagnostics["sparse_component"], [0, 6.25, 0], atol=2e-7)
    assert diagnostics["objective"] == pytest.approx(20.3125, abs=2e-7)
    assert diagnostics["converged"]
    assert diagnostics["primal_residual"] <= diagnostics["primal_tolerance"]
    assert diagnostics["dual_residual"] <= diagnostics["dual_tolerance"]


# TEST LOGIC: a small TV penalty admits a real level excursion instead of forcing sparsity.
def test_small_penalty_known_piecewise_solution():
    baseline, diagnostics = _huber_tv(np.array([0.0, 10.0, 0.0]), 0.2, 2.5, 2000, 1e-9)
    np.testing.assert_allclose(baseline, [0.2, 9.6, 0.2], atol=2e-7)
    np.testing.assert_allclose(diagnostics["sparse_component"], 0, atol=2e-7)
    assert diagnostics["converged"]


# TEST LOGIC: Huber saturation limits self-influence even when an outlier grows 100-fold.
def test_large_execution_excursion_has_bounded_baseline_influence():
    small, first = _huber_tv(np.array([0.0, 10.0, 0.0]), 8.0, 2.5, 2000, 1e-8)
    large, second = _huber_tv(np.array([0.0, 1000.0, 0.0]), 8.0, 2.5, 2000, 1e-8)
    assert first["converged"] and second["converged"]
    np.testing.assert_allclose(small, large, atol=3e-5)


# TEST LOGIC: stopping before convergence is observable, never silently called optimal.
def test_iteration_limit_is_auditable():
    _, diagnostics = _huber_tv(np.array([0.0, 10.0, 0.0]), 8.0, 2.5, 1, 1e-8)
    assert diagnostics["iterations"] == 1
    assert not diagnostics["converged"]
    assert diagnostics["primal_residual"] > diagnostics["primal_tolerance"]


# TEST LOGIC: deterministic all-clean data require zero sparse components and zero objective.
def test_flat_segment_does_not_force_contamination():
    times = np.arange(40, dtype=np.int64) * 3_600_000_000_000
    baseline, scales, diagnostics = robust_trend_segment(times, np.full(40, 100.0), controls())
    np.testing.assert_array_equal(baseline, 100)
    np.testing.assert_array_equal(scales, 1 / 4.5)
    np.testing.assert_array_equal(diagnostics["sparse_component"], 0)
    assert diagnostics["objective"] == 0
    assert diagnostics["converged"]


# TEST LOGIC: changing spread units and translating the level preserve normalized decisions.
def test_trend_scale_and_translation_equivariance():
    times = np.arange(90, dtype=np.int64) * 3_600_000_000_000
    values = np.r_[np.full(45, 100.0), np.full(45, 120.0)]
    values[20] += 30
    values[70] += 30
    baseline, scales, diagnostics = robust_trend_segment(times, values, controls())
    converted, converted_scale, converted_diagnostics = robust_trend_segment(times, 0.01 * values + 7, controls(abs_floor=0.01))
    np.testing.assert_allclose(converted, 0.01 * baseline + 7, atol=1e-10)
    np.testing.assert_allclose(converted_scale, scales * 0.01, atol=1e-10)
    assert diagnostics["converged"] and converted_diagnostics["converged"]
    assert diagnostics["objective"] == pytest.approx(converted_diagnostics["objective"])
    np.testing.assert_allclose(converted_diagnostics["sparse_component"], 0.01 * diagnostics["sparse_component"], atol=1e-10)


# TEST LOGIC: long, lasting levels stay near their observations while two transient prints stand out.
def test_trend_distinguishes_long_regimes_from_sparse_prints():
    times = np.arange(90, dtype=np.int64) * 3_600_000_000_000
    values = np.r_[np.full(45, 100.0), np.full(45, 120.0)]
    values[[20, 70]] += 30
    baseline, scales, diagnostics = robust_trend_segment(times, values, controls())
    assert diagnostics["converged"]
    flagged = np.abs(values - baseline) > np.maximum(1.0, 4.5 * scales)
    np.testing.assert_array_equal(np.flatnonzero(flagged), [20, 70])
    assert np.max(np.abs(baseline[:45] - 100)) < 0.2
    assert np.max(np.abs(baseline[45:] - 120)) < 0.2


# TEST LOGIC: target residual and observations outside the elapsed horizon cannot train its scale.
def test_local_scale_excludes_target_and_far_times():
    times = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    residuals = np.array([999, 0, 30, 0, 999], dtype=float)
    scale, count = _local_residual_scale(times, residuals, 2, controls(window=4, horizon="15ns"))
    assert count == 2
    assert scale == 1 / 4.5
    residuals[2] = 1e9
    assert _local_residual_scale(times, residuals, 2, controls(window=4, horizon="15ns")) == (scale, count)


# TEST LOGIC: the helper contract requires root to combine simultaneous rows before optimization.
@pytest.mark.parametrize("times,values", [([0, 0], [100, 101]), ([1, 0], [100, 101]), ([0], [np.nan]), ([], []), ([0], [100, 101])])
def test_trend_rejects_invalid_cohort_segments(times, values):
    with pytest.raises(ValueError):
        robust_trend_segment(times, values, controls())
