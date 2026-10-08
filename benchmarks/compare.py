"""Reproducible synthetic stress tests; these are not real TRACE accuracy estimates.

Run from the repository root: python -m benchmarks.compare --output reports
"""

# SETUP LOGIC: Import numerical tools, the engine contract, and local report utilities.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from jump_filter import FilterConfig, METHODS, filter_trades


# CONFIGURATION LOGIC: Fixed evaluation families and independent construction seeds.
SCENARIOS = ("clean", "noisy_flat", "trend", "regime", "sparse", "burst", "one_sided")
DEFAULT_SEEDS = (104729, 130363, 155921)


def latent_path(scenario: str, n: int) -> np.ndarray:
    """Construct a known efficient spread, including genuine repricing when requested."""
    # Input: scenario='trend', n=3.
    # Output: fair=[100.0, 100.03, 100.06].
    # CORE LOGIC: STEP 1 — Separate smooth drift from trade-specific contamination.
    fair = np.full(n, 100.0)
    if scenario == "trend":
        fair += np.arange(n) * 0.03

    # Trick: A permanent level change is clean market information, never an injected label.
    # Input: scenario='regime', n=6, fair=[100,100,100,100,100,100].
    # Output: fair=[100,100,100,116,116,116].
    # CORE LOGIC: STEP 2 — Add a real market regime change at the midpoint.
    if scenario == "regime":
        fair[n // 2 :] += 16.0
    return fair


def contamination_positions(
    scenario: str, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Select trade locations while keeping enough surrounding reference history."""
    # Trick: Bursts last four trades and can resemble a repricing to a causal algorithm.
    # Input: scenario='burst', n=100.
    # Output: positions=[25,26,27,28,50,51,52,53,75,76,77,78].
    # CORE LOGIC: STEP 1 — Define the clean case and contiguous stress episodes.
    if scenario == "clean":
        return np.array([], dtype=int)
    if scenario == "burst":
        anchors = (n // 4, n // 2, 3 * n // 4)
        return np.concatenate([np.arange(start, start + 4) for start in anchors])

    # Trick: Excluding 20 observations at each edge avoids evaluating unsupported endpoints.
    # Input: scenario='regime', n=70.
    # Output: eligible=[20,21,22,23,24,25,26,44,45,46,47,48,49], count=4.
    # CORE LOGIC: STEP 2 — Build candidate positions away from the true regime boundary.
    eligible = np.arange(20, n - 20)
    if scenario == "regime":
        eligible = eligible[np.abs(eligible - n // 2) > 8]
    count = max(4, int(n * 0.035))

    # Input: eligible=[4,5,6,7,8,9], count=2, rng=np.random.default_rng(7).
    # Output: positions=[8,7].
    # CORE LOGIC: STEP 3 — Sample independent contamination locations without duplicates.
    return rng.choice(eligible, size=min(count, len(eligible)), replace=False)


def make_case(scenario: str, seed: int, n: int = 480) -> pd.DataFrame:
    """Make observed trades and their hidden synthetic truth in common spread units."""
    # SETUP LOGIC: Reject undersized evaluation inputs and isolate random state per case.
    if scenario not in SCENARIOS or n < 100:
        raise ValueError("Use a listed scenario and at least 100 observations.")
    rng = np.random.default_rng(seed)
    size = max(100, n // 3) if scenario == "sparse" else n
    fair = latent_path(scenario, size)

    # Trick: Noise is normal with standard deviation 0.6 in the same units as spread.
    # Input: fair=[100,100,100], rng=np.random.default_rng(7).
    # Output: observed≈[100.00073809,100.17924732,99.83551729].
    # CORE LOGIC: STEP 1 — Add small ordinary trading noise to the known efficient path.
    observed = fair + rng.normal(0.0, 0.6, size)

    # Trick: Sparse interarrival times vary from 8 to 19 hours; no gap exceeds max_gap='1D'.
    # Input: scenario='sparse', size=3, rng=np.random.default_rng(7).
    # Output: hours=[0,15,31] from sampled interarrivals=[19,15,16].
    # CORE LOGIC: STEP 2 — Produce irregular calendar spacing for sparse bonds.
    hours = np.arange(size)
    if scenario == "sparse":
        hours = np.cumsum(rng.integers(8, 20, size))
        hours = hours - hours[0]

    # Trick: All burst/one-sided premiums share a sign; other stress cases mix buy/sell effects.
    # Input: positions=[1,3], observed=[100,100,100,100], offsets=[20,-15].
    # Output: observed=[100,120,100,85], truth=[False,True,False,True].
    # CORE LOGIC: STEP 3 — Inject labeled distortions distinct from fair-market variation.
    positions = contamination_positions(scenario, size, rng)
    offsets = rng.uniform(12.0, 25.0, len(positions))
    if scenario not in {"burst", "one_sided"}:
        offsets *= rng.choice([-1.0, 1.0], len(positions))
    observed[positions] += offsets
    truth = np.zeros(size, dtype=bool)
    truth[positions] = True

    # FILE IO LOGIC: Package the observed columns and evaluation-only truth in a frame.
    return pd.DataFrame(
        {
            "CUSIP": f"SYNTHETIC_{scenario.upper()}",
            "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h"),
            "spread": observed,
            "true_fair_spread": fair,
            "true_contamination": truth,
        }
    )


def evaluate(result: pd.DataFrame) -> dict[str, float | int]:
    """Score classification and retained observations against construction truth."""
    # Input: truth=[False,True,True,False], predicted=[False,True,False,True].
    # Output: tp=1, fp=1, fn=1, tn=1.
    # CORE LOGIC: STEP 1 — Count classifications without dropping abstained observations.
    truth = result["true_contamination"].to_numpy(dtype=bool)
    predicted = result["jf_is_outlier"].to_numpy(dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))

    # Trick: Precision/recall are undefined when their denominator is zero and stay NaN.
    # Input: tp=1, fp=1, fn=1, tn=1.
    # Output: precision=0.5, recall=0.5, f1=0.5, false_positive_rate=0.5.
    # CORE LOGIC: STEP 2 — Calculate rates with explicit empty-denominator behavior.
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    false_positive_rate = fp / (fp + tn) if fp + tn else np.nan

    # Trick: Retained RMSE excludes abstention (weight=0); compare it alongside coverage.
    # Input: observed=[100,120,101], fair=[100,100,100], predicted=[False,True,False], weights=[1,0.1,1].
    # Output: retained_residual=[0,1], retained_rmse≈0.707107, retained_bias=0.5.
    # CORE LOGIC: STEP 3 — Measure the quality of observations the filter retains.
    scored = result["jf_weight"].to_numpy() > 0
    retained = ~predicted & scored
    residual = result["spread"].to_numpy() - result["true_fair_spread"].to_numpy()
    retained_residual = residual[retained]
    retained_rmse = float(np.sqrt(np.mean(retained_residual**2))) if retained.any() else np.nan
    retained_bias = float(np.mean(retained_residual)) if retained.any() else np.nan
    original_rmse = float(np.sqrt(np.mean(residual**2)))

    # FILE IO LOGIC: Return a serializable metric row, including audit counts.
    return {
        "n": len(result), "contamination_n": int(truth.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": false_positive_rate,
        "original_rmse": original_rmse, "retained_rmse": retained_rmse,
        "retained_bias": retained_bias, "flagged_fraction": float(predicted.mean()),
        "scored_fraction": float(scored.mean()), "retained_fraction": float(retained.mean()),
        "regime_events": int(result["jf_regime_change"].sum()),
    }


def run_comparison(seeds: tuple[int, ...], n: int) -> pd.DataFrame:
    """Evaluate every method on identical seeded cases using the documented defaults."""
    # SETUP LOGIC: Allocate independent rows for each method, seed, and scenario.
    rows = []

    # TEST LOGIC: Hold each generated frame fixed for paired trials; never tune using hidden truth.
    for scenario in SCENARIOS:
        for seed in seeds:
            frame = make_case(scenario, seed, n)
            for method in METHODS:
                start = perf_counter()
                result = filter_trades(frame, FilterConfig(method=method))
                elapsed = perf_counter() - start
                metrics = evaluate(result)
                rows.append({"scenario": scenario, "seed": seed, "method": method, "seconds": elapsed, **metrics})

    # FILE IO LOGIC: Convert trial records into the detailed export table.
    return pd.DataFrame(rows)


def write_reports(results: pd.DataFrame, destination: Path, seeds: tuple[int, ...]) -> None:
    """Save reproducible raw results and compact, clearly qualified comparison outputs."""
    # FILE IO LOGIC: Create only the caller-selected report directory.
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "synthetic_trials.csv", index=False)

    # Trick: Seeds, not individual trades, are the independent evaluation units for variability.
    # Input: scenario='clean', method='hampel', FPR trials=[0.0,0.1,0.2].
    # Output: false_positive_rate_mean=0.1, false_positive_rate_std=0.1 (sample deviation).
    # CORE LOGIC: STEP 1 — Summarize repeated construction seeds per stress family and method.
    metrics = ["precision", "recall", "f1", "false_positive_rate", "retained_rmse", "retained_bias", "scored_fraction", "retained_fraction", "seconds"]
    grouped = results.groupby(["scenario", "method"])[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    summary = grouped.reset_index()

    # FILE IO LOGIC: Keep parameters and qualification adjacent to the comparison numbers.
    summary.to_csv(destination / "synthetic_summary.csv", index=False)
    manifest = {
        "kind": "synthetic construction-truth stress test",
        "seeds": list(seeds), "scenarios": list(SCENARIOS), "methods": list(METHODS),
        "config": "FilterConfig defaults per method; no scenario-specific tuning",
        "units": "synthetic spread units, interpreted as basis points in this benchmark",
        "limitation": "Not estimated precision or recall on real TRACE/BondCliQ trades. Construction labels define injected distortions; real market jumps and trade premiums can be observationally indistinguishable with three columns.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # UI LOGIC: Present a readable report without requiring a Markdown table dependency.
    lines = ["# Synthetic comparison", "", manifest["limitation"], "", f"Independent construction seeds: {', '.join(map(str, seeds))}.", "", "All methods use the documented default configuration. Standard deviations are across seeds. Clean-case precision and recall can be undefined and appear blank in CSV. Retained RMSE includes unflagged observations with positive fit weight; unsupported observations have zero weight. Scored coverage makes that abstention tradeoff visible. False positives are a fraction of true clean observations. RMSE describes retained raw trades, not fitted-mid accuracy.", "", "| Scenario | Method | Precision | Recall | False positives | Scored coverage | Retained RMSE |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        values = [row.precision_mean, row.recall_mean, row.false_positive_rate_mean, row.scored_fraction_mean, row.retained_rmse_mean]
        formatted = [f"{value:.3f}" if np.isfinite(value) else "—" for value in values]
        lines.append(f"| {row.scenario} | {row.method} | {' | '.join(formatted)} |")
    (destination / "SYNTHETIC_COMPARISON.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    # CONFIGURATION LOGIC: Accept explicit sizes/seeds and keep reports in an ignored directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--n", type=int, default=480)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    # FILE IO LOGIC: Run and publish local synthetic reports; print only the useful destination.
    results = run_comparison(seeds, args.n)
    write_reports(results, args.output, seeds)
    print(f"Saved {len(results)} synthetic trials to {args.output.resolve()}")


# SETUP LOGIC: Enable both module invocation and importable benchmark helpers.
if __name__ == "__main__":
    main()
