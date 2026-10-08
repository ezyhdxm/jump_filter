"""Ensure the benchmark's independent labels and coverage-aware metrics remain honest."""

# SETUP LOGIC: Import the evaluation layer separately from its executable runner.
import numpy as np
import pandas as pd
import pytest

from benchmarks.compare import evaluate, make_case


# TEST LOGIC: A mixed outcome penalizes missed contamination and incorrect deletion separately.
def test_metrics_count_abstention_and_retained_contamination():
    result = pd.DataFrame(
        {"true_contamination": [False, True, True, False, False],
         "true_fair_spread": [100.0] * 5,
         "spread": [100.0, 120.0, 120.0, 102.0, 105.0],
         "jf_is_outlier": [False, True, False, True, False],
         "jf_weight": [1.0, 0.1, 1.0, 0.5, 0.0],
         "jf_regime_change": [False] * 5}
    )
    metrics = evaluate(result)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 2)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["scored_fraction"] == 0.8
    assert metrics["retained_fraction"] == 0.4
    assert metrics["retained_rmse"] == pytest.approx(np.sqrt(200))
    assert metrics["retained_bias"] == 10.0


# TEST LOGIC: A clean market with no flags has undefined precision/recall rather than fake perfection.
def test_clean_unflagged_case_uses_undefined_rates():
    result = pd.DataFrame(
        {"true_contamination": [False, False], "true_fair_spread": [100.0, 100.0],
         "spread": [100.0, 101.0], "jf_is_outlier": [False, False],
         "jf_weight": [1.0, 1.0], "jf_regime_change": [False, False]}
    )
    metrics = evaluate(result)
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["recall"])
    assert np.isnan(metrics["f1"])
    assert metrics["false_positive_rate"] == 0.0


# TEST LOGIC: True repricing changes fair value without making the whole regime contaminated.
def test_regime_construction_truth_and_seed_reproducibility():
    frame = make_case("regime", 104729, n=120)
    repeated = make_case("regime", 104729, n=120)
    pd.testing.assert_frame_equal(frame, repeated)
    assert frame.loc[59, "true_fair_spread"] == 100.0
    assert frame.loc[60, "true_fair_spread"] == 116.0
    assert not frame.loc[52:68, "true_contamination"].any()
    assert frame["true_contamination"].sum() == 4
    changed = make_case("regime", 130363, n=120)
    assert not frame["spread"].equals(changed["spread"])
