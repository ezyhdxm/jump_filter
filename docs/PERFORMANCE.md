# Performance and capacity

The dashboard now separates interactive bond review from full-population batch screening. Selecting a bond does not need to refit every other CUSIP. The accelerated engine has been measured on **1,001,000 trades and exactly 13,000 CUSIPs**; this is an actual run, not a linear extrapolation from a small demo.

## What changed

- **Selected bond** is the default review population in both Streamlit and Jupyter. **All bonds** explicitly requests portfolio-wide filtering, statistics, and exports.
- A source-position index is built once. Switching CUSIPs materializes only the requested source rows, preserving duplicate dataframe index labels and global source row IDs.
- Applied reviews use an LRU cache bounded by 16 entries and 64 MiB. Large batch results remain the current applied review but are not additionally retained in that cache. Pending controls do not silently change exported decisions.
- Engine preprocessing uses array boundaries for CUSIPs, gap segments, and timestamp cohorts instead of constructing a pandas dataframe for every trade timestamp. Datetime-typed input has a vectorized parsing path; mixed/invalid records retain the conservative validation path.
- Optional Numba compilation accelerates Hampel, local linear, jump reversion, local piecewise, and consensus. It preserves the original neighborhood definitions, duplicate-cohort exclusion, six IRLS rounds, session clock, and audit fields. Rounding-near cutoff cases are recomputed using the reference regression helpers.
- `backend="auto"` uses the Python implementation below 10,000 valid in-session rows to avoid compiler startup during small interactive reviews. Larger supported local methods use the compiled backend when the speed extra is available. `backend="python"` forces the reference implementation; `backend="numba"` requests the compiler for supported methods. The actual backend is recorded in `annotated.attrs["jump_filter"]["backend"]`.
- Charts use a strict 20,000-point drawing budget, preserving period endpoints and prioritizing outlier/transition markers. If priority markers exceed the budget, those markers are sampled too; displayed-versus-total counts disclose the reduction. Filtering, statistics, and annotated exports continue to use every trade in the explicitly selected review population. Presentation limits do not alter reference neighborhoods or fitting selection.

Rolling IQR, multiscale, robust trend, and causal EWMA retain their Python numerical implementations. In particular, the joint Huber-TV optimization has different computational costs; the local-method capacity results below are not a throughput promise for that solver.

## How cost scales

Let \(n\) be source rows, \(c\le n\) distinct CUSIP/timestamp cohorts, and \(w\) the configured count cap. Preprocessing sorts valid rows by instrument, timestamp, and source position, with approximately \(O(n\log n)\) sorting cost. Local methods inspect at most \(w\) reference cohorts per target; local regressions retain six fixed IRLS rounds. Their numerical work therefore grows roughly with \(cw\), including bounded median/MAD calculations, rather than with a dense \(n\times n\) system. Array/output storage grows with \(n\); original source columns are also copied into the annotated result.

The CUSIP count matters through group/segment boundaries and observed cohorts. An average of 77 trades per bond does not describe a bond with tens of thousands of events or the iteration count of the global robust-trend solver. The separate skew workload and the selected-bond workflow address that distinction directly.

## Reproducible workload

`benchmarks/performance.py` creates exact row and bond counts with seed `32452843`:

- **Balanced:** 77 trades per CUSIP, totaling 1,001,000 rows.
- **Skewed:** 130 CUSIPs have 3,801 trades each; the median CUSIP has 39 trades. Half of the extra rows, after assigning one per CUSIP, go to the most active 1% of bonds.
- Event times are irregular and restricted to explicitly configured New York weekday sessions, 08:00–18:30 with exclusive close. Weekends and overnight gaps remain in UTC timestamps; the filter uses trading time. No holiday calendar is claimed and no trades are interpolated into closures.
- Event cadence becomes less liquid halfway through each history. The fair path has a slope reversal followed by a persistent drop; observations contain changing clean scatter and injected extremes.
- Timestamp cohorts deliberately repeat every seventeenth position, dataframe index labels repeat, and source rows are shuffled. Every benchmark checks complete source preservation, positional row IDs, total status counts, and consistency between outlier and fitting-eligible masks.

These constructions measure software capacity. Their outlier counts do not establish accuracy on real executions, identify markups/commissions, or establish an economic mid.

## Measured results

Measurements were taken on macOS 26.6.2 / Apple arm64, 10 logical CPUs, Python 3.13.9, NumPy 2.3.4, pandas 2.3.3, Numba 0.68.0, and SciPy 1.16.3. The numerical kernel is not configured for parallel reductions. Methods run sequentially in separate fresh child processes, with the input generated before timing. Warm calls reuse the same input but delete the previous annotated output.

These are completed measurements of the guarded compiled implementation. First-process calls include initialization and cache loading; the benchmark does not erase persistent compiler caches. They should not be interpreted as guaranteed compilation-from-scratch times.

| Workload | Method | First process call | Warm call | Maximum process peak RSS |
|---|---|---:|---:|---:|
| Balanced, 1,001,000 rows / 13,000 CUSIPs | Hampel | 2.533 s | 2.078 s | 1.23 GiB |
| Balanced, 1,001,000 rows / 13,000 CUSIPs | Local piecewise | 10.419 s | 9.580 s | 1.37 GiB |
| Balanced, 1,001,000 rows / 13,000 CUSIPs | Consensus | 14.991 s | 14.975 s | 1.32 GiB |
| Skewed, 1,001,000 rows / 13,000 CUSIPs | Hampel | 5.834 s | 2.077 s | 1.62 GiB |
| Skewed, 1,001,000 rows / 13,000 CUSIPs | Local piecewise | 8.788 s | 8.706 s | 1.44 GiB |
| Skewed, 1,001,000 rows / 13,000 CUSIPs | Consensus | 13.829 s | 13.534 s | 1.47 GiB |

RSS is the **whole process high-water mark**, including imports, input, temporary arrays, annotated output, and preceding summary/repeat phases. It is neither incremental engine RAM nor a promise for the memory of a running notebook with other objects. macOS byte values and Linux KiB values are converted to MiB; the benchmark reports null when that measurement is unavailable on Windows. Upload parsing, chart construction, browser rendering, notebook startup, and file export are excluded from filter timings.

### Optional Quantity and support-trend rules

Version 0.4.0 was also measured on the balanced **1,001,000-row / 13,000-CUSIP** workload above, with a raw-amount Quantity column cycling through 500,000, 1,000,000 and 2,000,000 by source position. The method was `local_linear`, the clock was trading time, and all other algorithm settings used `FilterConfig` defaults. Both optional rules used their default 1,000,000 notional cutoff and 10 bp distance cap; spreads were already in bp. Numerical kernels were warmed on 13,013 source rows before measurement.

| Warm engine call | Wall time | Final flags | Fit-eligible trades |
|---|---:|---:|---:|
| Robust local linear, optional rules off | 7.521 s | 32,408 | 968,592 |
| Robust local linear, both rules on, first trial | 18.099 s | 32,841 | 968,159 |
| Robust local linear, both rules on, second trial | 17.323 s | 32,841 | 968,159 |

The independent left/right support fits and additional row-level audit columns add work and memory. Whole-process peak RSS across this measurement process was **1.87 GiB**, including its warm-up and all three trials. These timings exclude workload generation, summary calculation, file export, notebook communication and chart rendering. They establish completed engine runs on this machine; the synthetic flag counts do not validate trade-screening accuracy on private data. Both rules remain off by default, and selected-bond review continues to avoid filtering the full universe on each interaction.

### Why the earlier dashboard felt slow

Measured pre-optimization timings on the same generator and environment were:

| Previous workload | Method | Filter wall time |
|---|---|---:|
| 10,010 trades / 130 CUSIPs | Hampel | 0.656 s |
| 10,010 trades / 130 CUSIPs | Local piecewise | 2.943 s |
| 10,010 trades / 130 CUSIPs | Consensus | 4.221 s |
| 13,000 CUSIPs with one trade each | Hampel | 3.808 s |
| One selected CUSIP with 77 trades | All nine methods, individually | 0.007–0.036 s per method |

The one-trade-per-CUSIP case abstains throughout, demonstrating that pandas group/cohort overhead consumed time even without useful fitting support. The previous dashboard also filtered the full source population when the user needed to inspect one bond. The small-case timings above are measured independently; they are not estimates of previous million-row runtime.

### Interactive workspace measurements

Workspace measurements separately time the one-time 1,001,000-row source index, first selected-bond review including statistics, and cached revisit. They exclude widget construction and Plotly/frontend rendering. Source indexing took 29–32 ms.

| Selected review from the million-row source | First review including statistics | Cached revisit | Actual backend |
|---|---:|---:|---|
| Hampel, 77-trade bond | 14 ms | 26 microseconds | Python |
| Local piecewise, 77-trade bond | 33 ms | 31 microseconds | Python |
| Consensus, 77-trade bond | 43 ms | 29 microseconds | Python |
| Consensus, 3,801-trade heavy bond | 1.596 s | 30 microseconds | Python |

Every cached revisit returned the same applied record. These values demonstrate that a large source universe need not impose full-universe filtering latency on each interaction; a heavier individual history still costs more on its first review.

The native Notebook 7 frontend was also exercised with the balanced million-row source. Its CUSIP dropdown initially held 100 choices; searching `BEN013000` returned that instrument plus the current selection. Choosing it reviewed 77 of the 1,001,000 source rows, with live statistics and a responsive native chart. The ordinary demo separately verified explicit batch Apply and independent method-comparison charts. These frontend checks establish working interactions; the timing table above excludes frontend rendering and widget communication.

### First-ever compiler initialization

A separate consensus run used a newly empty `NUMBA_CACHE_DIR`: 10,010 trades / 130 CUSIPs took **4.284 s** on the first call, including compilation, then **0.153 s** in the same process. The latter is about 27.6 times faster than the measured 4.221 s pre-optimization reference call on that exact workload. This comparison includes algorithm execution, preprocessing, and output publication; it excludes input generation. It is not a claim of that speedup for every method or every machine. The automatic small-review Python threshold prevents this compilation cost during ordinary short-bond inspection.

## Run the capacity checks

Install the compiler extra from the checkout:

```bash
# SETUP LOGIC: Install the optional native compiler dependencies.
python -m pip install -e '.[speed]'
```

Run all three capacity methods serially, with first-process and warm calls:

```bash
# MEASUREMENT LOGIC: Run exact-capacity workloads serially and retain machine-readable reports.
python -m benchmarks.performance --rows 1001000 --bonds 13000 \
  --methods hampel local_piecewise consensus --backend numba --repeats 2 \
  --output performance_balanced.json
python -m benchmarks.performance --rows 1001000 --bonds 13000 --scenario skewed \
  --methods hampel local_piecewise consensus --backend numba --repeats 2 \
  --output performance_skewed.json
```

Install dashboard dependencies before measuring the shared review workspace:

```bash
# SETUP LOGIC: Workspace measurements use the optional dashboard dependencies.
python -m pip install -e '.[dashboard,speed]'
# MEASUREMENT LOGIC: Measure source indexing, first selected review, and cached revisit.
python -m benchmarks.performance --mode workspace --rows 1001000 --bonds 13000 \
  --methods consensus --selected-cusip BEN000001 --repeats 2 \
  --output performance_workspace.json
```

JSON reports retain exact requests, configuration, package/platform context, engine source fingerprint, wall and CPU time, process peak RSS, summary time, and validated audit counts. The capacity runs share engine fingerprint `87f105c0e6df`; subsequent annotation-only Input/Output example corrections preserved the executable AST. For a reference-backend comparison use `--backend python` with a manageable row count; forcing Python on the full million-row consensus population can take considerably longer.

For large inputs, prefer timezone-aware datetime columns and Parquet with preserved CUSIP strings. Keep the interactive scope at **Selected bond**, tune against representative high- and low-liquidity instruments, then explicitly request **All bonds** or run the batch API for the intended fitting population. A larger window, many independent gap segments, long heavy-bond histories, mixed string timestamp parsing, or a different machine can change both runtime and memory.
