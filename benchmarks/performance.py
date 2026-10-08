"""Reproducible engine throughput; every method runs in a fresh child process.

Run from the checkout: python -m benchmarks.performance --output performance.json
First-call timings include initialization/JIT; repeat timings retain only the input.
The synthetic frame is a workload specification, not a transaction-cost model.
"""

# SETUP LOGIC: Standard library measurements and deterministic numerical tools.
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import pandas as pd

# SETUP LOGIC: Unix reports peak RSS; Windows retains time metrics with explicit null memory.
try:
    import resource
except ImportError:
    resource = None

from jump_filter import FilterConfig, METHODS, filter_trades, summarize

# CONFIGURATION LOGIC: Explicit synthetic weekday sessions, with no inferred holidays.
SESSION_SECONDS = 10 * 3600 + 30 * 60
DEFAULT_ROWS = 1_001_000
DEFAULT_BONDS = 13_000
DEFAULT_SEED = 32452843


def allocate_counts(rows, bonds, scenario="balanced"):
    """Return exact positive group sizes; skew reserves half the extra rows for 1%."""
    # INPUT VALIDATION LOGIC: A CUSIP cannot appear without at least one source row.
    if isinstance(rows, bool) or isinstance(bonds, bool) or rows < bonds or bonds < 1:
        raise ValueError("require rows >= bonds >= 1")
    if scenario not in ("balanced", "skewed"):
        raise ValueError("scenario must be balanced or skewed")

    # Input: rows=10,bonds=3,scenario='balanced' -> Output: counts=[4,3,3].
    # Trick: the first remainder groups receive one extra row; no rounding loses records.
    # CORE LOGIC: STEP 1
    counts = np.ones(bonds, dtype=np.int64)
    extra = rows - bonds
    if scenario == "balanced" or bonds == 1:
        counts += extra // bonds
        counts[:extra % bonds] += 1
        return counts

    # Input: rows=10,bonds=3,scenario='skewed' -> Output: heavy=1,counts=[4,3,3].
    # Trick: skew applies to extra rows after allocating one per CUSIP, preserving all IDs.
    # CORE LOGIC: STEP 2
    heavy = max(1, bonds // 100)
    concentrated = extra // 2
    remaining = extra - concentrated
    counts[:heavy] += concentrated // heavy
    counts[:concentrated % heavy] += 1
    counts[heavy:] += remaining // (bonds - heavy)
    counts[heavy:heavy + remaining % (bonds - heavy)] += 1
    return counts


def _active_seconds(groups, positions, counts, rng):
    """Generate unequal interarrival times with later, less liquid observations."""
    # Input: groups=[0,0,1,1],positions=[0,1,0,1],counts=[2,2],
    # sampled increments=[30,40,50,60] -> Output: scaled increments=[30,480,50,720].
    # Trick: liquidity affects clean event cadence; duplicate timestamp cohorts occur every17th row.
    # CORE LOGIC: STEP 1
    increments = rng.integers(20, 181, size=len(groups), dtype=np.int64)
    increments *= np.where(positions >= counts[groups] // 2, 12, 1)
    increments[(positions > 0) & (positions % 17 == 0)] = 0

    # Input: groups=[0,0,1,1],increments=[30,480,50,720],counts=[2,2] ->
    # Output: starts=[0,2],cumulative=[30,510,560,1280],elapsed=[30,510,50,770].
    # Trick: subtract each group's preceding prefix sum, so one bond never inherits another's age.
    # CORE LOGIC: STEP 2
    starts = np.r_[0, np.cumsum(counts[:-1])]
    cumulative = np.cumsum(increments)
    before = np.zeros(len(counts), dtype=np.int64)
    before[1:] = cumulative[starts[1:] - 1]
    return cumulative - before[groups]


def _calendar_times(elapsed):
    """Invert active seconds onto 08:00–18:30 New York weekdays."""
    # Input: elapsed=[0,37799,37800,75600] seconds ->
    # Output: local=['2026-09-14 08:00:00','2026-09-14 18:29:59',
    # '2026-09-15 08:00:00','2026-09-16 08:00:00'].
    # Trick: close is exclusive; gaps cross only explicit weekday sessions, without interpolation.
    # CORE LOGIC: STEP 1
    sessions, within = np.divmod(elapsed, SESSION_SECONDS)
    dates = pd.bdate_range("2026-09-14", periods=int(sessions.max()) + 1)
    local = dates.take(sessions) + pd.Timedelta(hours=8) + pd.to_timedelta(within, unit="s")

    # Input: local=['2026-09-14 08:00:00','2026-09-15 08:00:00'] ->
    # Output: UTC=['2026-09-14 12:00:00Z','2026-09-15 12:00:00Z'].
    # Trick: localizing after adding clock hours honors DST; it does not guess market holidays.
    # CORE LOGIC: STEP 2
    return local.tz_localize("America/New_York").tz_convert("UTC").as_unit("ns")


def _spreads(groups, positions, counts, rng):
    """A clean turning path and liquidity noise, with deterministic injected extremes."""
    # Input: groups=[0,0,0],positions=[1,2,3],counts=[4] ->
    # Output: turning=[2,2,2],fair=[100.003,100.006,97.001] (approximately).
    # Trick: the slope reversal and subsequent permanent3bp drop are market-path changes.
    # CORE LOGIC: STEP 1
    turning = counts[groups] // 2
    fair = 100 + groups % 100 + 0.003 * np.minimum(positions, turning)
    fair -= 0.005 * np.maximum(positions - turning, 0)
    fair -= 3 * (positions > turning)

    # Input: groups=[0,0,0],positions=[1,2,3],turning=[2,2,2],
    # fair=[100.003,100.006,97.001],sampled normal=[0,0,0] ->
    # Output: sigma=[0.4,1.6,1.6],distortion=[0,0,0],spread=[100.003,100.006,97.001].
    # Trick: generated extremes are workload stress only; no economic-cause labels are asserted.
    # CORE LOGIC: STEP 2
    sigma = np.where(positions >= turning, 1.6, 0.4)
    noise = rng.normal(0, sigma)
    distortion = np.where((positions > 0) & (positions % 37 == 0), 18.0, 0.0)
    distortion *= np.where(groups % 2 == 0, 1, -1)
    return fair + noise + distortion


def make_workload(rows=DEFAULT_ROWS, bonds=DEFAULT_BONDS, *, scenario="balanced", seed=DEFAULT_SEED):
    """Exactly rows and CUSIPs, duplicate indices/cohorts, irregular open-session trades."""
    # CONFIGURATION LOGIC: Use an independent reproducible generator for all numerical draws.
    counts = allocate_counts(rows, bonds, scenario)
    rng = np.random.default_rng(seed)

    # Input: rows=5,bonds=2,counts=[3,2] ->
    # Output: groups=[0,0,0,1,1],starts=[0,3],positions=[0,1,2,0,1].
    # Trick: broadcast group starts to positional rows before any source-order permutation.
    # CORE LOGIC: STEP 1
    groups = np.repeat(np.arange(bonds, dtype=np.int64), counts)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    positions = np.arange(rows) - starts[groups]

    # Input: counts=[2,1],groups=[0,0,1],positions=[0,1,0],rng=np.random.default_rng(7) ->
    # Output: elapsed=[172,1612,1560]seconds,times=['2026-09-14T12:02:52Z',
    # '2026-09-14T12:26:52Z','2026-09-14T12:26:00Z'],spreads approximately
    # [99.89034485785511,98.57805305798836,100.27252674372524].
    # Trick: the RNG is shared in this order, so event and noise draws remain reproducible.
    # CORE LOGIC: STEP 2
    elapsed = _active_seconds(groups, positions, counts, rng)
    times = _calendar_times(elapsed)
    spreads = _spreads(groups, positions, counts, rng)

    # Input: rows=3,bonds=2,groups=[0,0,1],times=['2026-09-14T12:00Z','2026-09-14T12:01Z',
    # '2026-09-14T12:02Z'],spreads=[100,101,102] -> Output: original rows have
    # CUSIP=['BEN000001','BEN000001','BEN000002'],index=[0,0,1],matching time/spread.
    # Trick: duplicate index labels test positional restoration; they are not duplicate-row IDs.
    # CORE LOGIC: STEP 3
    identifiers = np.array([f"BEN{bond + 1:06d}" for bond in range(bonds)], dtype=object)
    frame = pd.DataFrame(dict(CUSIP=identifiers[groups], time=times, spread=spreads))
    frame.index = np.arange(rows) // 2

    # Input: source indices=[0,0,1],permutation=[2,0,1] ->
    # Output: reordered indices=[1,0,0],all three source records preserved exactly.
    # Trick: shuffle makes sorting cost explicit; the benchmark never presents pre-grouped input.
    # CORE LOGIC: STEP 4
    frame = frame.iloc[rng.permutation(rows)]

    # AUDIT LOGIC: Attach explicit workload provenance after source-order construction.
    frame.attrs["performance_workload"] = dict(seed=seed, scenario=scenario, rows=rows, bonds=bonds,
                                               timezone="America/New_York", open="08:00", close="18:30",
                                               holidays=[], shuffled=True, duplicate_indices=True)
    return frame


def _peak_rss_mib():
    # MEASUREMENT LOGIC: ru_maxrss is bytes on macOS, KiB on Linux; convert either to MiB.
    if resource is None:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _environment():
    # MEASUREMENT LOGIC: Package versions and machine context travel with every timing report.
    versions = {}
    for name in ("numpy", "pandas", "numba", "scipy", "jump-filter"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                logical_cpu_count=os.cpu_count(), packages=versions)


def _audit_counts(frame, annotated):
    # VALIDATION LOGIC: Counts are meaningless if a method drops, duplicates, or reorders source rows.
    assert len(annotated) == len(frame)
    assert annotated.index.equals(frame.index)
    assert annotated[list(frame.columns)].equals(frame)
    assert np.array_equal(annotated["jf_row_id"].to_numpy(), np.arange(len(frame)))
    statuses = annotated["jf_status"].value_counts(dropna=False).to_dict()
    assert sum(statuses.values()) == len(frame)
    assert not (annotated["jf_fit_eligible"] & annotated["jf_is_outlier"]).any()
    return dict(status={str(key): int(value) for key, value in statuses.items()},
                flagged=int(annotated["jf_is_outlier"].sum()),
                fit_eligible=int(annotated["jf_fit_eligible"].sum()),
                session_boundary=int(annotated["jf_session_boundary"].sum()),
                solver_failed=int(annotated["jf_status"].eq("solver_not_converged").sum()))


def _engine_from_request(request):
    # SETUP LOGIC: Optional baseline source permits measured before/after without checkout mutation.
    path = request.get("engine_file")
    if not path:
        source = Path(inspect.getfile(filter_trades))
        return filter_trades, hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("jump_filter._benchmark_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return module.filter_trades, digest


def measure_case(request):
    # CONFIGURATION LOGIC: All trials share explicit filter settings, not UI-dependent defaults.
    settings = dict(method=request["method"], time_basis="trading", window=request["window"],
                    horizon="3D", max_gap="1D", min_neighbors=6)
    config = FilterConfig(**settings)
    engine, engine_sha256 = _engine_from_request(request)
    kwargs = {}
    if "backend" in inspect.signature(engine).parameters:
        kwargs["backend"] = request["backend"]
    elif request["backend"] == "numba":
        raise ValueError("this engine source does not support a numba backend")

    # MEASUREMENT LOGIC: Generation and checks are outside the timed filter operation.
    generation_start = perf_counter()
    frame = make_workload(request["rows"], request["bonds"], scenario=request["scenario"], seed=request["seed"])
    generation_seconds = perf_counter() - generation_start
    sizes = frame.groupby("CUSIP", sort=False).size()
    assert len(sizes) == request["bonds"] and int(sizes.sum()) == request["rows"]
    duplicate_cohorts = int(frame.duplicated(["CUSIP", "time"]).sum())
    trials = []

    # MEASUREMENT LOGIC: Delete prior annotated output so warm calls cannot overlap output frames.
    for repeat in range(request["repeats"]):
        gc.collect()
        wall_start, cpu_start = perf_counter(), process_time()
        annotated = engine(frame, config, **kwargs)
        cpu_seconds, wall_seconds = process_time() - cpu_start, perf_counter() - wall_start
        engine_peak_rss = _peak_rss_mib()
        audit = _audit_counts(frame, annotated)
        summary_start = perf_counter()
        overview = summarize(annotated)
        summary_seconds = perf_counter() - summary_start
        assert len(overview) == request["bonds"] and int(overview["rows"].sum()) == request["rows"]
        trials.append(dict(call=repeat + 1, phase="first_process_call" if repeat == 0 else "warm_process_call",
                           wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                           rows_per_second=len(frame) / wall_seconds, engine_process_peak_rss_mib=engine_peak_rss,
                           summary_seconds=summary_seconds, process_peak_rss_after_summary_mib=_peak_rss_mib(),
                           audit=audit, engine_metadata=annotated.attrs.get("jump_filter", {})))
        del annotated, overview

    # REPORTING LOGIC: RSS is process high-water, including input/imports; it is not incremental RAM.
    return dict(request=request, configuration=config.to_dict(), environment=_environment(),
                engine_source_sha256=engine_sha256, generation_seconds=generation_seconds,
                input_deep_memory_mib=float(frame.memory_usage(index=True, deep=True).sum() / 1024**2),
                workload=dict(rows=len(frame), cusips=len(sizes), min_trades=int(sizes.min()),
                              median_trades=float(sizes.median()), max_trades=int(sizes.max()),
                              duplicate_timestamp_rows=duplicate_cohorts), trials=trials)


def measure_workspace(request):
    # SETUP LOGIC: Workspace timing requires dashboard extras but never opens or renders a UI.
    from jump_filter.dashboard import ReviewWorkspace
    generation_start = perf_counter()
    frame = make_workload(request["rows"], request["bonds"], scenario=request["scenario"], seed=request["seed"])
    generation_seconds = perf_counter() - generation_start
    mapping = dict(cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC")
    config = FilterConfig(method=request["method"], time_basis="trading", window=request["window"])

    # MEASUREMENT LOGIC: Source indexing is timed once; a cache lookup cannot hide initialization.
    gc.collect()
    start = perf_counter()
    workspace = ReviewWorkspace(frame, mapping)
    index_seconds = perf_counter() - start
    assert len(workspace.positions) == request["bonds"]
    chosen = request["selected_cusip"]
    if chosen not in workspace.positions:
        raise ValueError(f"selected CUSIP {chosen!r} is absent")
    records = []
    first_record = None

    # MEASUREMENT LOGIC: The first selected review includes filtering/statistics; repeat calls hit the applied-result cache.
    for repeat in range(request["repeats"]):
        wall_start, cpu_start = perf_counter(), process_time()
        record = workspace.review(config, scope="selected", bond=chosen)
        cpu_seconds, wall_seconds = process_time() - cpu_start, perf_counter() - wall_start
        result = record["result"]
        positions = workspace.positions[chosen]
        assert result[list(frame.columns)].equals(frame.iloc[positions])
        assert np.array_equal(result["jf_row_id"].to_numpy(), positions)
        assert record["scope"] == "selected" and len(result) == len(positions)
        first_record = record if first_record is None else first_record
        records.append(dict(call=repeat + 1, phase="first_selected_review" if repeat == 0 else "cached_selected_review",
                            wall_seconds=wall_seconds, cpu_seconds=cpu_seconds, reviewed_rows=len(result),
                            backend=result.attrs["jump_filter"].get("backend", "python"),
                            same_record_as_first=record is first_record, process_peak_rss_mib=_peak_rss_mib()))

    # REPORTING LOGIC: Chart construction, frontend rendering, upload parsing, and exports are excluded.
    return dict(request=request, configuration=config.to_dict(), environment=_environment(),
                generation_seconds=generation_seconds, index_seconds=index_seconds,
                source_rows=len(frame), source_cusips=len(workspace.positions), selected_cusip=chosen,
                cache_entries=len(workspace.cache), cache_bytes=workspace.cache_bytes, trials=records)


def _parser():
    # CONFIGURATION LOGIC: Methods run serially; engine methods are isolated in separate processes.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--bonds", type=int, default=DEFAULT_BONDS)
    parser.add_argument("--scenario", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["hampel", "local_piecewise", "consensus"])
    parser.add_argument("--backend", choices=("auto", "python", "numba"), default="auto")
    parser.add_argument("--mode", choices=("engine", "workspace"), default="engine")
    parser.add_argument("--selected-cusip", default="BEN000001", help="workspace mode: exact instrument to review")
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--engine-file", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("performance.json"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    return parser


def main():
    # CONFIGURATION LOGIC: Reject invalid workloads before launching an expensive child.
    args = _parser().parse_args()
    if args.worker:
        request = json.loads(args.worker)
        measurement = measure_workspace if request.get("mode") == "workspace" else measure_case
        print(json.dumps(measurement(request), allow_nan=False))
        return
    allocate_counts(args.rows, args.bonds, args.scenario)
    if args.repeats < 1:
        raise ValueError("repeats must be at least1")
    if args.mode == "workspace" and (args.backend != "auto" or args.engine_file):
        raise ValueError("workspace mode follows the dashboard's auto backend and current source")
    if args.engine_file:
        args.engine_file = str(Path(args.engine_file).resolve(strict=True))
    scope = "filter_trades only; input generated before timing" if args.mode == "engine" else "workspace source indexing and selected review; no frontend rendering"
    report = dict(schema_version=1, timing_scope=scope,
                  first_call_definition="first call in a fresh process; persistent compiled caches are not erased",
                  memory_definition="process high-water RSS includes imports, input, and output; macOS bytes/Linux KiB converted to MiB; unavailable on Windows",
                  cases=[])

    # FILE IO LOGIC: JSON carries reproducible requests; subprocess argument lists avoid shell quoting.
    for method in args.methods:
        request = dict(rows=args.rows, bonds=args.bonds, scenario=args.scenario, seed=args.seed,
                       method=method, backend=args.backend, window=args.window,
                       repeats=args.repeats, engine_file=args.engine_file,
                       mode=args.mode, selected_cusip=args.selected_cusip)
        completed = subprocess.run([sys.executable, "-m", "benchmarks.performance", "--worker", json.dumps(request)],
                                   check=True, capture_output=True, text=True)
        case = json.loads(completed.stdout)
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        times = ", ".join(f"{trial['phase']}={trial['wall_seconds']:.3f}s" for trial in case["trials"])
        print(f"{method}: {args.rows:,} source rows, {args.bonds:,} CUSIPs; {times}", flush=True)


# SETUP LOGIC: Imports are side-effect free; only a direct CLI invocation creates reports.
if __name__ == "__main__":
    main()
