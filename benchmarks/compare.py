"""Historical fitting stress tests, with independent construction truth and support reporting.

Run: python -m benchmarks.compare --output reports --methods hampel robust_trend
No synthetic number is an estimated accuracy or probability on real bond transactions.
"""

# SETUP LOGIC: Numerical tools, the public engine, and local report utilities.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from jump_filter import FilterConfig, METHODS, filter_trades

# CONFIGURATION LOGIC: Earlier seeds remain reproducible; new stress uses held-out primes.
SCENARIOS = (
    "clean", "noisy_flat", "trend", "regime", "sparse", "burst", "one_sided",
    "dense_gradual", "steep_trend", "clustered_40pct", "bidask_bimodal",
    "jump_adjacent", "heteroscedastic", "irregular", "sparse_mixed",
    "turning_continuous", "turning_drop", "calendar_sessions", "liquidity_shift",
)
DEFAULT_SEEDS = (32452843, 49979687, 67867967)
LEGACY_SEEDS = (104729, 130363, 155921)


def latent_path(scenario: str, n: int) -> np.ndarray:
    """Keep genuine drift, curvature, and abrupt repricing outside bad-trade labels."""
    # Input: scenario='trend', n=3.
    # Output: fair=[100.0,100.03,100.06].
    # CORE LOGIC: STEP 1 — Add efficient-market drift in trade-cohort coordinates.
    index = np.arange(n)
    fair = np.full(n, 100.0)
    if scenario == "trend":
        fair += index * 0.03
    if scenario == "steep_trend":
        fair += index * 0.75
    if scenario == "dense_gradual":
        fair += index * 0.12 + 3.0 * np.sin(index / 35.0)

    # Trick: These are lasting market changes, never blanket contamination labels.
    # Input: scenario='regime', n=6, fair=[100,100,100,100,100,100].
    # Output: fair=[100,100,100,116,116,116].
    # CORE LOGIC: STEP 2 — Add a permanent price jump to the appropriate cases.
    if scenario in {"regime", "jump_adjacent", "sparse_mixed"}:
        fair[n // 2 :] += 16.0
    if scenario == "calendar_sessions":
        fair[40:] += 20.0

    # Input: scenario='turning_continuous', n=6, index=[0,1,2,3,4,5].
    # Output: fair=[100,100.3,100.6,100.9,100.4,99.9].
    # Trick: A continuous slope reversal is genuine information, not a level-jump label.
    # CORE LOGIC: STEP 3 — Construct rising markets that subsequently reverse direction.
    if scenario in {"turning_continuous", "turning_drop"}:
        split = n // 2
        fair = 100.0 + 0.3 * np.minimum(index, split)
        fair -= (0.5 if scenario == "turning_continuous" else 0.25) * np.maximum(index - split, 0)
        if scenario == "turning_drop":
            fair[split:] -= 20.0
    return fair


def contamination_positions(scenario: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Set locations independently of every detector; stress blocks include clean prints."""
    # Input: scenario='clean', n=100.
    # Output: positions=[], dtype=int.
    # CORE LOGIC: STEP 1 — Reserve no-label controls for clean and legitimate side trades.
    if scenario in {"clean", "bidask_bimodal"}:
        return np.array([], dtype=int)
    if scenario == "burst":
        anchors = (n // 4, n // 2, 3 * n // 4)
        return np.concatenate([np.arange(start, start + 4) for start in anchors])

    # Trick: Each block contains exactly 40% contaminated cohorts; clean gaps interrupt runs.
    # Input: n=100, anchors=[25,50,75], width=10, dirty=4.
    # Output: offsets=[0,1,5,6], positions=[25,26,30,31,50,51,55,56,75,76,80,81].
    # CORE LOGIC: STEP 2 — Create three partially contaminated blocks with short bursts.
    if scenario == "clustered_40pct":
        width = max(10, n // 10)
        dirty = int(width * 0.4)
        offsets = np.r_[np.arange((dirty + 1) // 2), np.arange(width // 2, width // 2 + dirty // 2)]
        anchors = (n // 4, n // 2, 3 * n // 4)
        return np.concatenate([start + offsets for start in anchors])

    # Input: scenario='regime', n=70, rng=np.random.default_rng(7).
    # Output: eligible=[20,21,22,23,24,25,26,44,45,46,47,48,49], count=4,
    # positions=[46,26,48,45].
    # CORE LOGIC: STEP 3 — Retain legacy interior injection rules for old comparisons.
    eligible = np.arange(20, n - 20)
    if scenario == "regime":
        eligible = eligible[np.abs(eligible - n // 2) > 8]
    if scenario == "calendar_sessions":
        eligible = eligible[eligible != 40]
    count = max(4, int(n * 0.035))
    positions = rng.choice(eligible, size=min(count, len(eligible)), replace=False)

    # Trick: Neighboring bad prints at an actual jump were absent from the earlier benchmark.
    # Input: scenario='jump_adjacent', n=100, sampled positions=[21,30,70,78].
    # Output: sorted positions=[21,30,48,49,50,51,70,78].
    # CORE LOGIC: STEP 4 — Include distortions immediately around the market transition.
    if scenario in {"jump_adjacent", "turning_drop"}:
        adjacent = n // 2 + np.array([-2, -1, 0, 1])
        positions = np.unique(np.r_[positions, adjacent])
    if scenario == "calendar_sessions":
        positions = np.unique(np.r_[positions, 70])
    return positions


def observation_times(scenario: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return elapsed hours, including density changes and genuine support-breaking gaps."""
    # Input: scenario='dense_gradual', size=4.
    # Output: hours=[0.0,0.0333333333,0.0666666667,0.1], approximately.
    # CORE LOGIC: STEP 1 — Define dense and ordinary calendar grids.
    hours = np.arange(size, dtype=float)
    if scenario == "dense_gradual":
        hours /= 30.0

    # Input: scenario='liquidity_shift', size=6, gaps=[2,2,1/60,1/60,8,8].
    # Output: hours=[0,2,2.0166666667,2.0333333333,10.0333333333,18.0333333333], approximately.
    # CORE LOGIC: STEP 2 — Separate trade density changes from shifts in efficient spread.
    if scenario == "liquidity_shift":
        gaps = np.full(size, 2.0)
        gaps[size // 3 : 2 * size // 3] = 1.0 / 60.0
        gaps[2 * size // 3 :] = 8.0
        return np.cumsum(gaps) - gaps[0]

    # Trick: Random interarrivals use calendar time; rows cannot stand in for elapsed time.
    # Input: scenario='sparse', size=3, rng=np.random.default_rng(7).
    # Output: hours=[0,15,31] from interarrivals=[19,15,16].
    # CORE LOGIC: STEP 3 — Generate independent sparse or highly irregular arrivals.
    if scenario in {"sparse", "sparse_mixed"}:
        gaps = rng.integers(8, 20, size).astype(float)
        if scenario == "sparse_mixed":
            gaps[size // 3 : 2 * size // 3] = 0.25
        hours = np.cumsum(gaps) - gaps[0]
    if scenario == "irregular":
        gaps = np.exp(rng.uniform(np.log(0.05), np.log(8.0), size))
        hours = np.cumsum(gaps) - gaps[0]
        hours[size // 3 :] += 40.0
    return hours


def calendar_times(size: int, rng: np.random.Generator) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Generate weekday New York sessions, including overnight/weekend closures.

    No exchange holiday calendar is inferred. The benchmark explicitly treats every
    Monday–Friday date as open; the public API separately accepts supplied closures.
    """
    # Input: size=3, rng=np.random.default_rng(7).
    # Output: count=1, days=['2025-01-02 00:00 America/New_York'],
    # session=[0,0,0], minute=[3,34,75].
    # CORE LOGIC: STEP 1 — Allocate enough weekday sessions to include a real weekend closure.
    count = int(np.ceil(size / 20))
    days = pd.bdate_range("2025-01-02", periods=count, tz="America/New_York")
    session = np.repeat(np.arange(count), 20)[:size]
    minute = np.sort(rng.choice(np.arange(630), 20 * count, replace=True).reshape(count, 20), axis=1).ravel()[:size]

    # Input: size=3, minute=[3,34,75].
    # Output: minute=[0,34,75]; a truncated three-cohort session has no forced closing print.
    # Trick: Opening prints reveal overnight repricing; full 20-cohort sessions also end at minute629.
    # CORE LOGIC: STEP 2 — Include the declared session open without creating overnight prints.
    minute[np.arange(0, size, 20)] = 0
    minute[np.arange(19, size, 20)] = 629

    # Input: days=['2025-01-02'], session=[0,0,0], minute=[0,34,75], size=3, count=1.
    # Output: second=[0,1,2], times=['2025-01-02 13:00Z','2025-01-02 13:34:01Z',
    # '2025-01-02 14:15:02Z'], clock_hours=[0,0.5669444444,1.2505555556], approximately.
    # Trick: Distinct seconds prevent duplicate cohorts; closed hours do not add support age.
    # CORE LOGIC: STEP 3 — Produce irregular observed times and an independent compressed clock.
    second = np.tile(np.arange(20), count)[:size]
    local = days[session] + pd.to_timedelta(8, unit="h") + pd.to_timedelta(minute, unit="min") + pd.to_timedelta(second, unit="s")
    clock_hours = session * 10.5 + minute / 60.0 + second / 3600.0
    return local.tz_convert("UTC"), clock_hours


def ordinary_noise(scenario: str, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Side-of-market prints remain legitimate; bid/ask spread is not a commission label."""
    # Input: scenario='clean', size=3, rng=np.random.default_rng(7).
    # Output: noise≈[0.00073809,0.17924732,-0.16448271], valid_wide_side=[False,False,False].
    # CORE LOGIC: STEP 1 — Allow a real change in clean observation variance.
    scale = np.full(size, 0.6)
    if scenario == "heteroscedastic":
        scale[size // 2 :] = 3.0
    if scenario == "liquidity_shift":
        scale[size // 3 : 2 * size // 3] = 0.2
        scale[2 * size // 3 :] = 3.0
    noise = rng.normal(0.0, scale, size)
    valid_wide_side = np.zeros(size, dtype=bool)

    # Trick: Rare wider side trades are clean labels even though a univariate rule may reject them.
    # Input: signs=[-1,1,1,-1], wide=[False,True,False,False], normal_noise=[0,0,0,0].
    # Output: noise=[-2,7,2,-2], valid_wide_side=[False,True,False,False].
    # CORE LOGIC: STEP 2 — Construct a symmetric two-mode market plus legitimate wide prints.
    if scenario == "bidask_bimodal":
        sides = rng.choice([-1.0, 1.0], size)
        valid_wide_side[rng.choice(np.arange(size), max(3, size // 30), replace=False)] = True
        premium = np.where(valid_wide_side, 7.0, 2.0)
        noise += sides * premium
    return noise, valid_wide_side


def make_case(scenario: str, seed: int, n: int = 480) -> pd.DataFrame:
    """Build observed inputs plus evaluation-only truth, never supplied as filter features."""
    # SETUP LOGIC: Isolate random state; reject frames too small for fixed stress episodes.
    if scenario not in SCENARIOS or n < 100:
        raise ValueError("Use a listed scenario and at least 100 observations.")
    rng = np.random.default_rng(seed)
    size = max(100, n // 3) if scenario in {"sparse", "sparse_mixed"} else n
    fair = latent_path(scenario, size)
    noise, valid_wide_side = ordinary_noise(scenario, size, rng)
    hours = observation_times(scenario, size, rng)
    times = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    if scenario == "calendar_sessions":
        times, hours = calendar_times(size, rng)

    # Input: fair=[100,100,100], noise=[0.1,-0.2,0.3].
    # Output: observed=[100.1,99.8,100.3].
    # CORE LOGIC: STEP 1 — Separate efficient prices and ordinary market-side variation.
    observed = fair + noise
    positions = contamination_positions(scenario, size, rng)
    offsets = rng.uniform(12.0, 25.0, len(positions))
    if scenario == "steep_trend":
        offsets = rng.uniform(7.0, 12.0, len(positions))

    # Trick: Clustered blocks contain systematic one-sided markups, not clean level changes.
    # Input: observed=[100,100,100,100], positions=[1,3], offsets=[20,-15].
    # Output: observed=[100,120,100,85], truth=[False,True,False,True].
    # CORE LOGIC: STEP 2 — Inject signed distortions independently of detector parameters.
    if scenario not in {"burst", "one_sided", "clustered_40pct"}:
        offsets *= rng.choice([-1.0, 1.0], len(positions))
    observed[positions] += offsets
    truth = np.zeros(size, dtype=bool)
    truth[positions] = True

    # FILE IO LOGIC: Package truth separately; the engine receives only CUSIP/time/spread below.
    return pd.DataFrame({
        "CUSIP": f"SYNTHETIC_{scenario.upper()}",
        "time": times, "evaluation_hours": hours,
        "spread": observed, "true_fair_spread": fair,
        "true_contamination": truth, "true_valid_wide_side": valid_wide_side,
    })


def downstream_fit(frame: pd.DataFrame, retained: np.ndarray) -> np.ndarray:
    """One fixed local OLS downstream fit, shared by every method and the unfiltered control.

    This estimator is a diagnostic, not the project's inferred mid price. It uses up to
    21 closest accepted timestamp cohorts within 3 days, 6 minimum cohorts, and never
    borrows across an adjacent gap longer than 1 day. Targets include rejected rows.
    """
    # Input: time=['1970-01-01T00:00Z','1970-01-01T01:00Z','1970-01-02T03:00Z'],spread=[100,100,120]; evaluation_hours is absent.
    # Output: stamps=[0,3600000000000,97200000000000],hours=[0,1,27],values=[100,100,120],fitted=[NaN,NaN,NaN],segment=[0,0,1].
    # Trick: Normalize every pandas datetime resolution to nanoseconds before dividing by a nanosecond duration.
    # Input events are already ordered; supplied evaluation_hours instead measures the chosen trading clock.
    # CORE LOGIC: STEP 1 — Build consistent elapsed-hour coordinates and separate unsupported gap segments.
    stamps = pd.to_datetime(frame["time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    hours = frame["evaluation_hours"].to_numpy(dtype=float) if "evaluation_hours" in frame else (stamps - stamps.min()) / pd.Timedelta("1h").value
    values = frame["spread"].to_numpy(dtype=float)
    fitted = np.full(len(frame), np.nan)
    segment = np.cumsum(np.r_[False, np.diff(hours) > 24.0])

    # Trick: A target's rejection never removes it from evaluation; support limits are common.
    # Input: hours=[0,1,2,3,4,5,6], target=3, retained=[True]*7.
    # Output: candidate=[0,1,2,3,4,5,6], neighbors=[3,2,4,1,5,0,6].
    # CORE LOGIC: STEP 2 — Select the same bounded neighborhood rule for every target.
    for target in range(len(frame)):
        valid = retained & (segment == segment[target]) & (np.abs(hours - hours[target]) <= 72.0)
        candidate = np.flatnonzero(valid)
        order = np.argsort(np.abs(hours[candidate] - hours[target]), kind="stable")
        neighbors = candidate[order[:21]]
        if len(neighbors) < 6:
            continue
        fitted[target] = _ols_at_target(hours[neighbors], values[neighbors], hours[target])
    return fitted


def _ols_at_target(hours: np.ndarray, values: np.ndarray, target: float) -> float:
    """Fit an ordinary intercept/slope centered on the requested target timestamp."""
    # Input: hours=[0,1,2,3,4,5], values=[100,101,102,103,104,105], target=2.
    # Output: design=[[1,-2],[1,-1],[1,0],[1,1],[1,2],[1,3]], beta≈[102,1], estimate≈102.
    # Trick: OLS is deliberately not robust, so deletion can improve or degrade its fit.
    # CORE LOGIC: STEP 1 — Produce a fixed downstream estimate without consulting truth.
    design = np.column_stack((np.ones(len(hours)), hours - target))
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(beta[0])


def _rmse(residual: np.ndarray) -> float:
    # Input: residual=[3,4].
    # Output: 3.5355339059, approximately; residual=[] instead returns NaN.
    # CORE LOGIC: STEP 1 — Keep empty-error comparisons undefined.
    return float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan


def evaluate(result: pd.DataFrame) -> dict[str, float | int]:
    """Separate labels, raw retained quality, reference accuracy, and downstream fitting."""
    # Input: truth=[False,True,True,False], predicted=[False,True,False,True].
    # Output: tp=1,fp=1,fn=1,tn=1.
    # CORE LOGIC: STEP 1 — Abstained bad trades still count as false negatives.
    truth = result["true_contamination"].to_numpy(dtype=bool)
    predicted = result["jf_is_outlier"].to_numpy(dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))

    # Input: tp=1,fp=1,fn=1,tn=1.
    # Output: precision=0.5,recall=0.5,f1=0.5,false_positive_rate=0.5.
    # Trick: Zero denominators produce NaN, not a fictitious perfect classifier.
    # CORE LOGIC: STEP 2 — Measure classifications independently of fit support.
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    false_positive_rate = fp / (fp + tn) if fp + tn else np.nan

    # Input: status=['ok','outlier','insufficient_history'], eligible=[True,False,False].
    # Output: scored=[True,True,False], retained=[True,False,False].
    # Trick: Positive influence weight alone is not sufficient evidence of fit eligibility.
    # CORE LOGIC: STEP 3 — Require exported eligibility and supported decision statuses.
    supported_status = result["jf_status"].isin(["ok", "outlier", "provisional_jump"])
    scored = supported_status.to_numpy() & result["jf_weight"].gt(0).to_numpy()
    retained = result["jf_fit_eligible"].to_numpy(dtype=bool) & scored & ~predicted
    fair = result["true_fair_spread"].to_numpy(dtype=float)
    residual = result["spread"].to_numpy(dtype=float) - fair
    retained_residual = residual[retained]

    # Input: fair=[100,100,100], baseline=[100,103,NaN], clean=[True,True,False].
    # Output: reference_error=[0,3], reference_rmse≈2.121320344.
    # Trick: All evaluable clean targets count, including clean rows rejected by the detector.
    # CORE LOGIC: STEP 4 — Score the reference on clean targets with explicit coverage.
    baseline = result["jf_baseline"].to_numpy(dtype=float)
    target = ~truth & result["evaluation_target"].to_numpy(dtype=bool)
    reference_target = target & np.isfinite(baseline)
    reference_error = baseline[reference_target] - fair[reference_target]
    downstream = result["downstream_fitted"].to_numpy(dtype=float)
    downstream_target = target & np.isfinite(downstream)
    downstream_error = downstream[downstream_target] - fair[downstream_target]

    # Input: target=[True,True,False], raw_fit=[101,103,NaN], fair=[100,100,100].
    # Output: raw_fit_error=[1,3], raw_fit_rmse≈2.236067977.
    # CORE LOGIC: STEP 5 — Retain common target denominators and inspect legitimate side trades.
    raw_fit = result["unfiltered_fitted"].to_numpy(dtype=float)
    raw_fit_error = raw_fit[target] - fair[target]
    wide = result["true_valid_wide_side"].to_numpy(dtype=bool)
    wide_fpr = float(np.mean(predicted[wide])) if wide.any() else np.nan
    target_count = int(target.sum())
    clean_count = int((~truth).sum())

    # Input: solver_converged=[None,True,False], status=['ok','ok','solver_not_converged'].
    # Output: solver_converged_fraction=0.5, solver_failure_fraction=1/3.
    # Trick: Non-solver methods have an undefined convergence rate, not an invented success.
    # CORE LOGIC: STEP 6 — Preserve numerical failure as a visible abstention outcome.
    solver_known = result["jf_solver_converged"].notna()
    solver_converged = float(result.loc[solver_known, "jf_solver_converged"].astype(bool).mean()) if solver_known.any() else np.nan
    solver_failed = float(result["jf_status"].eq("solver_not_converged").mean())
    solver_iterations = int(result["jf_solver_iterations"].max())

    # Input: residual=[0,20,0], retained_residual=[0,0].
    # Output: original_rmse≈11.547005384, retained_rmse=0, retained_bias=0, retained_squared_error=0.
    # CORE LOGIC: STEP 7 — Summarize conditional raw-trade quality independently of reference fitting.
    original_rmse = _rmse(residual)
    retained_rmse = _rmse(retained_residual)
    retained_bias = float(np.mean(retained_residual)) if retained.any() else np.nan
    retained_squared_error = float(np.sum(retained_residual**2))

    # Input: reference_error=[0,3], reference_target=[True,True,False], target_count=3.
    # Output: reference_clean_rmse≈2.121320344, reference_squared_error=9, reference_n=2, reference_coverage=2/3.
    # CORE LOGIC: STEP 8 — Keep reference error and missing-reference coverage together.
    reference_rmse = _rmse(reference_error)
    reference_squared_error = float(np.sum(reference_error**2))
    reference_n = int(reference_target.sum())
    reference_coverage = reference_n / target_count if target_count else np.nan

    # Input: downstream_error=[0,8], raw_fit_error=[2,2,2], downstream_target=[True,True,False], target_count=3.
    # Output: downstream_rmse≈5.656854249, downstream_squared_error=64, downstream_n=2, downstream_coverage=2/3, unfiltered_fit_rmse=2.
    # CORE LOGIC: STEP 9 — Compare the common downstream model before and after filtering.
    downstream_rmse = _rmse(downstream_error)
    downstream_squared_error = float(np.sum(downstream_error**2))
    downstream_n = int(downstream_target.sum())
    downstream_coverage = downstream_n / target_count if target_count else np.nan
    unfiltered_fit_rmse = _rmse(raw_fit_error)

    # Input: truth=[False,True,False], predicted=[False,True,False], scored=[True,True,False], retained=[True,False,False].
    # Output: flagged_fraction=1/3, scored_fraction=2/3, retained_fraction=1/3, excluded_fraction=2/3, clean_retention=1/2, retained_n=1, contamination_n=1.
    # CORE LOGIC: STEP 10 — Expose exclusion and support rather than optimizing conditional RMSE alone.
    flagged_fraction = float(predicted.mean())
    scored_fraction = float(scored.mean())
    retained_fraction = float(retained.mean())
    excluded_fraction = 1 - retained_fraction
    clean_retention = float(np.sum(retained & ~truth) / clean_count) if clean_count else np.nan
    retained_n = int(retained.sum())
    contamination_n = int(truth.sum())
    regime_events = int(result["jf_regime_change"].sum())

    # FILE IO LOGIC: Include squared sums/counts for honest pooled RMSE and all exclusion fractions.
    return {
        "n": len(result), "contamination_n": contamination_n, "clean_n": clean_count,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
        "f1": f1, "false_positive_rate": false_positive_rate, "valid_wide_side_fpr": wide_fpr,
        "original_rmse": original_rmse, "retained_rmse": retained_rmse,
        "retained_bias": retained_bias,
        "retained_squared_error": retained_squared_error, "retained_n": retained_n,
        "flagged_fraction": flagged_fraction, "scored_fraction": scored_fraction,
        "retained_fraction": retained_fraction, "excluded_fraction": excluded_fraction,
        "clean_retained_fraction": clean_retention,
        "reference_clean_rmse": reference_rmse, "reference_clean_n": reference_n,
        "reference_squared_error": reference_squared_error,
        "reference_clean_coverage": reference_coverage,
        "downstream_clean_rmse": downstream_rmse, "downstream_clean_n": downstream_n,
        "downstream_squared_error": downstream_squared_error,
        "downstream_clean_coverage": downstream_coverage,
        "unfiltered_clean_fit_rmse": unfiltered_fit_rmse, "evaluation_target_n": target_count,
        "solver_converged_fraction": solver_converged, "solver_failure_fraction": solver_failed,
        "solver_iterations_max": solver_iterations,
        "regime_events": regime_events,
    }


def benchmark_config(scenario: str, method: str) -> FilterConfig:
    # CONFIGURATION LOGIC: Explicit supplied weekday calendar; no exchange holidays are guessed.
    if scenario == "calendar_sessions":
        return FilterConfig(method=method, time_basis="trading")
    return FilterConfig(method=method)


def run_comparison(seeds: tuple[int, ...], n: int, methods: tuple[str, ...] = METHODS,
                   scenarios: tuple[str, ...] = SCENARIOS) -> pd.DataFrame:
    """Evaluate fixed defaults; truth columns never enter the engine input frame."""
    # CONFIGURATION LOGIC: Validate requested subsets rather than silently skip misspellings.
    if not methods or set(methods) - set(METHODS):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    if not scenarios or set(scenarios) - set(SCENARIOS):
        raise ValueError(f"scenarios must be a nonempty subset of {SCENARIOS}")
    rows = []

    # TEST LOGIC: Reuse observed inputs and a common unfiltered-fit target set for paired trials.
    for scenario in scenarios:
        for seed in seeds:
            frame = make_case(scenario, seed, n)
            raw_fit = downstream_fit(frame, np.ones(len(frame), dtype=bool))
            for method in methods:
                start = perf_counter()
                result = filter_trades(frame[["CUSIP", "time", "spread"]], benchmark_config(scenario, method))
                elapsed = perf_counter() - start
                result = pd.concat([result, frame.drop(columns=["CUSIP", "time", "spread"])], axis=1)
                result["evaluation_target"] = np.isfinite(raw_fit)
                result["unfiltered_fitted"] = raw_fit
                result["downstream_fitted"] = downstream_fit(result, result["jf_fit_eligible"].to_numpy(dtype=bool))
                rows.append({"scenario": scenario, "seed": seed, "method": method, "seconds": elapsed, **evaluate(result)})

    # FILE IO LOGIC: Export one auditable row per paired trial.
    return pd.DataFrame(rows)


def write_reports(results: pd.DataFrame, destination: Path, seeds: tuple[int, ...]) -> None:
    """Write metrics with construction details and probability-calibration limitations."""
    # FILE IO LOGIC: Save detailed trials in the explicitly selected directory.
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "synthetic_trials.csv", index=False)

    # Input: FPR trials=[0.0,0.1,0.2].
    # Output: false_positive_rate_mean=0.1, false_positive_rate_std=0.1 (sample deviation).
    # Trick: Seed-level variation is descriptive; trades inside a stress path are dependent.
    # CORE LOGIC: STEP 1 — Aggregate repeated independent constructions, not dependent prints.
    metrics = ["precision", "recall", "f1", "false_positive_rate", "valid_wide_side_fpr",
               "retained_rmse", "retained_bias", "scored_fraction", "retained_fraction",
               "clean_retained_fraction", "reference_clean_rmse", "reference_clean_coverage",
               "downstream_clean_rmse", "downstream_clean_coverage", "unfiltered_clean_fit_rmse",
               "solver_converged_fraction", "solver_failure_fraction", "seconds"]
    grouped = results.groupby(["scenario", "method"])[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    summary = grouped.reset_index()

    # FILE IO LOGIC: Keep provenance and a full per-method configuration next to results.
    summary.to_csv(destination / "synthetic_summary.csv", index=False)
    manifest = {
        "kind": "historical fitting synthetic construction-truth stress test",
        "seeds": list(seeds), "scenarios": list(results["scenario"].unique()),
        "methods": list(results["method"].unique()),
        "config": {method: FilterConfig(method=method).to_dict() for method in results["method"].unique()},
        "time_basis_by_scenario": {scenario: "trading" if scenario == "calendar_sessions" else "wall" for scenario in results["scenario"].unique()},
        "units": "synthetic basis points; no method/scenario-specific tuning",
        "downstream_fit": "Fixed unweighted OLS, nearest 21 accepted cohorts, 3D horizon, min 6, split gaps >1D; evaluate all common clean targets, including rejected clean prints.",
        "limitation": "Synthetic recovery is not real TRACE/BondCliQ precision or recall. Three columns cannot identify customer side, commissions or economic cause. Scores/weights are not calibrated probabilities. Coverage and error must be read together.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # UI LOGIC: Produce a readable comparison without an optional table-formatting package.
    lines = ["# Historical fitting synthetic comparison", "", manifest["limitation"], "",
             f"Held-out construction seeds: {', '.join(map(str, seeds))}.", "",
             "All methods use defaults. Values are means across seeds. The fit error evaluates clean targets that the unfiltered fit can support, including rejected clean prints; finite-estimate coverage is reported separately. Raw retained RMSE is conditional on acceptance and cannot establish fitted-mid quality. Clean precision/recall may be undefined.", "",
             "| Scenario | Method | Precision | Recall | Clean FPR | Fit eligible | Clean fit RMSE | Fit coverage |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        values = [row.precision_mean, row.recall_mean, row.false_positive_rate_mean,
                  row.retained_fraction_mean, row.downstream_clean_rmse_mean, row.downstream_clean_coverage_mean]
        formatted = [f"{value:.3f}" if np.isfinite(value) else "—" for value in values]
        lines.append(f"| {row.scenario} | {row.method} | {' | '.join(formatted)} |")
    (destination / "SYNTHETIC_COMPARISON.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    # CONFIGURATION LOGIC: Offer subsets for development; full validation keeps all cases.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--n", type=int, default=480)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()

    # FILE IO LOGIC: Run and publish locally, reporting only the useful destination.
    results = run_comparison(tuple(args.seeds), args.n, tuple(args.methods), tuple(args.scenarios))
    write_reports(results, args.output, tuple(args.seeds))
    print(f"Saved {len(results)} synthetic trials to {args.output.resolve()}")


# SETUP LOGIC: Keep importable evaluation helpers separate from executable invocation.
if __name__ == "__main__":
    main()
