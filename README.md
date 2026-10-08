# jump_filter

An auditable bond-trade spread filter and research dashboard. Pass a pandas DataFrame with a bond identifier, timestamp and spread; retain every row and receive outlier flags, local references, scores, thresholds, support counts and explanations.

The engine screens statistical deviations. These three columns cannot identify retail trades, commissions, distressed sellers, or the true market mid. Zero and negative spreads are valid observations. Input spread units are preserved.

## Install and launch

```bash
# SETUP LOGIC: install the independent package and interactive extras.
python -m pip install -e '.[dashboard,app,parquet,speed]'
# UI LOGIC: start the local browser dashboard.
python -m streamlit run jump_filter/app.py
```

Open **jump_filter_dashboard.ipynb** for the notebook dashboard, or use the browser dashboard with a synthetic demonstration or CSV upload. Both provide CUSIP and nine-method selection, paired sliders and exact inputs, plots, red outlier markers, reference bands, statistics, method comparisons and exports. **Method & mathematics** explains each algorithm in English with formulas, numeric examples, parameter effects and limits; it follows the selected method immediately. Local diagnostics show reference density, span, gaps and the method's noise scale. Apply publishes a reproducible snapshot; pending settings do not silently alter downloads.

The workbench uses a compact light layout with six review metrics, method-specific tuning controls, and expandable statistics and mathematical details. Inactive parameters remain available but disabled. Charts share a consistent palette and distinguish outliers and transitions by marker shape as well as color; method comparisons use horizontal bars with readable names and explicit evaluation coverage. Notebook charts resize to their output pane, and the notebook marks settings as Ready, Applied or Pending so the displayed review remains easy to audit.

Choose a CUSIP before Apply. **Selected bond · interactive** is the default review population, so tuning one bond does not screen the entire source dataset. First visits compute that bond with the applied settings; cached revisits reuse the result. Select **All bonds · batch** and Apply explicitly for portfolio statistics and complete-source annotations. Each table, export and settings file identifies the applied population. Browser downloads are prepared individually on request.

## Run inside Jupyter Notebook

The notebook dashboard uses native `ipywidgets` controls and Plotly `FigureWidget` charts. It runs inside Jupyter Notebook 7 or JupyterLab 4, with no Streamlit server required. Install the notebook extra in the environment that will run both Jupyter and the kernel:

```bash
# SETUP LOGIC: install the notebook frontend, kernel and native chart dependencies.
python -m pip install -e '.[notebook,speed]'
# SETUP LOGIC: register this environment without writing a separate user-level kernel.
python -m ipykernel install --sys-prefix --name jump-filter --display-name "Jump Filter"
# UI LOGIC: open the supplied dashboard notebook in Jupyter Notebook.
python -m notebook jump_filter_dashboard.ipynb
```

Select the **Jump Filter** kernel and run all cells. For JupyterLab, use `python -m jupyterlab jump_filter_dashboard.ipynb` instead. The dashboard appears in a notebook output cell; CUSIP selection, method selection, sliders, exact inputs, Apply, comparison, mathematics, statistics and exports all work there.

If you already have a notebook environment, run `%pip install -e '.[dashboard,speed]'` from the repository directory **in that notebook**, restart its kernel and run all cells. `%pip` installs into the active kernel. If the frontend and kernel use different environments, install `jupyterlab_widgets` and `anywidget` in the frontend environment, and the dashboard and speed extras in the kernel environment. The notebook extra includes the frontend and kernel dependencies. Plotly 6 uses `anywidget` for native charts.

Reopen the notebook and run all cells when starting a new session; live controls depend on a running kernel. Full standalone HTML charts remain available through Export for viewing without a kernel.

Notebook sections replace one complete output snapshot when a review changes. Repeated Apply, bond navigation, and comparison refreshes keep one chart in each section. A bare `show_filter(data).run()` displays the dashboard once in its cell; the returned panel can still be displayed from a later cell. After upgrading from an earlier version, restart the kernel and rerun the dashboard cell to replace existing widget models.

Version 0.4.1 fixes a Plotly `AssertionError` that can appear below the dashboard after Apply or CUSIP navigation. Live figures preserve native numerical arrays instead of reconstructing them from Plotly's encoded JSON representation; missing hover metadata still travels as JSON-safe nulls. Update the package in the active notebook environment, restart the kernel, clear old outputs and rerun the dashboard cell. Recreating the existing chart widgets is necessary for the fix to take effect.

The fresh-install browser check used Python 3.13, Notebook 7.6.3, JupyterLab 4.6.4, ipykernel 7.4.0, ipywidgets 8.1.9, anywidget 0.11.0 and Plotly 7.1.0. It verified bond/method selection, paired parameter controls, Apply, native charts, mathematical explanations and method comparison inside Notebook. The test suite also executes the supplied notebook in an actual Jupyter kernel.

## Large portfolios

Install the optional `speed` extra for native acceleration. `filter_trades(..., backend="auto")` uses Python for small reviews (fewer than 10,000 valid active rows), avoiding compiler startup while exploring a bond. Larger supported workloads use Numba when installed. `backend="python"` forces the reference implementation; `backend="numba"` forces acceleration for supported methods. The actual backend is recorded in `review.attrs["jump_filter"]["backend"]`.

The accelerated methods are Hampel, local linear, jump/reversion, local piecewise and consensus. They share vectorized input preparation and one compiled call across all bonds and segments. Decisions near numerical cutoff boundaries are checked by the original regression solver. IQR, multiscale, Huber-TV and causal EWMA retain their Python algorithms with improved input/group preparation. These methods have different runtimes; a fast Hampel result does not establish the cost of every method. The first compiled batch can incur compilation, and later calls reuse the cache.

The interactive workbench indexes the source once and keeps a bounded per-bond cache (16 entries, 64 MiB). `panel.result` is the applied population: one bond in interactive mode, or the full source in explicit batch mode. An active batch result remains available even when it exceeds the cache budget. Switching bonds retains the last applied method and calendar while pending edits remain unapplied.

Charts have a 20,000-point display budget. Small bonds show every timed finite trade; larger bonds prioritize review markers and retain extrema from ordered event buckets. The caption reports displayed versus full rows and outliers, including when review markers themselves exceed the budget. Scoring, statistics and annotated exports use all records in the applied population. Histogram counts are aggregated from all finite residuals before sending a fixed-size payload to the frontend. Use `trade_figure(..., max_points=None)` when a complete interactive drawing is required.

Use a timezone-aware datetime column for the fastest batch preparation. Object/string timestamps retain scalar validation for mixed offsets, naive timezone handling and DST ambiguity. Benchmark results use a three-column frame with typed UTC timestamps; CSV parsing, extra source columns, chart rendering and file exports have additional costs. See [measured million-trade benchmarks and reproducible commands](docs/PERFORMANCE.md).

## DataFrame API

```python
# SETUP LOGIC: imports do not read data or open a dashboard.
from jump_filter import FilterConfig, filter_trades, summarize, show_filter, select_fit_data

# CONFIGURATION LOGIC: every spread threshold uses the input spread unit.
config = FilterConfig(method="local_piecewise", window=31, horizon="3D",
                      max_gap="1D", min_neighbors=6, threshold=4.5,
                      abs_floor=1.0, time_basis="trading",
                      session_timezone="America/New_York",
                      session_open="08:00", session_close="18:30")

# Input: df has CUSIP='A' at hourly NY times08:00..16:00 on2026-09-14,
# spread=[100,100,100,100,130,100,100,100,100],config as above ->
# Output: all9source rows retained; flags=[False,False,False,False,True,False,False,False,False],
# fit_eligible=[False,False,False,False,False,False,False,False,False].
# Trick: only positions3..5 have three references per side; positions3/5 disagree,
# so they abstain as ambiguous_transition. Other edge cohorts have insufficient_history.
# CORE LOGIC: STEP 1
review = filter_trades(df, config, cusip_col="CUSIP", time_col="time",
                       spread_col="spread", timezone="America/New_York")

# UI LOGIC: inspect every bond, then open the notebook workbench.
print(summarize(review))
dashboard = show_filter(df, config=config, cusip_col="CUSIP", time_col="time",
                        spread_col="spread", timezone="America/New_York", unit="bp")

# Input: statuses=['ok','outlier','ambiguous_transition'],flags=[False,True,False],
# jf_fit_eligible=[True,False,False] -> Output: fit_data contains only row0,weight1.
# Trick: abstentions are excluded even when their outlier flag is False.
# CORE LOGIC: STEP 2
fit_data = select_fit_data(review, policy="hard")
```

`show_filter` takes the source frame; annotations reserve the `jf_` prefix. Column mappings must be distinct. Original order, duplicate index labels and original columns remain intact. Naive timestamps use the configured timezone; aware timestamps convert to UTC. Invalid dates, ambiguous/nonexistent DST times, numeric epochs, nonfinite spreads and missing/blank identifiers receive `invalid_input` and zero suggested fit weight. Convert numeric epochs explicitly before passing them.

For BondCliQ, map `spread_col="BM_SPREAD"` when appropriate. If your `BM_SPREAD` is in percentage points, an `abs_floor` of `0.01` equals 1 bp. A display label does not convert data; convert percentage points to bp explicitly before using a 1-bp floor.

## Optional fitting rules

Both dashboards include an **Optional fitting rules** section. Each rule starts off, has a separate enable switch, and uses the same Apply/comparison/export workflow as the selected algorithm.

- **Quantity-aware uncertainty:** choose any quantity column and its units. Original amounts use multiplier 1; the default small-trade boundary is strictly below 1,000,000. Small uncertain trades with independent suspicious evidence are rejected. Otherwise valid known-size uncertain trades are explicitly retained as `retained_uncertain`. Confirmed algorithm outliers remain excluded at every size. Missing quantity does not change the quantity decision.
- **Maximum support-trend distance:** reject a trade more than 10 bp from a sufficiently supported, bracketed robust local trend, regardless of quantity. Configure the boundary through a slider or exact input. Set input spread units per bp explicitly: 1 for bp, 0.01 for percentage points, or 0.0001 for decimal rates. An unsupported trend cannot trigger this rule.

The DataFrame API accepts `quantity_col="your_column"`. Enable rules using `FilterConfig(quantity_rule=True, quantity_threshold=1_000_000, quantity_multiplier=1, max_deviation_rule=True, max_deviation_bps=10, spread_units_per_bp=1)`. Omitting the switches preserves the original three-column behavior. The public notebook contains a synthetic `Quantity` column with both small and large trades.

Original algorithm evidence, final fitting eligibility, uncertain retention, and rule-specific exclusions are reported separately. The optional centered support trend can use future observations even when the selected base method is causal EWMA. See [the exact rules, mathematical definitions, boundaries, and worked example](docs/OPTIONAL_RULES.md).

## Methods

| Method | Local reference and flag rule | Future data |
|---|---|---|
| `hampel` | Leave-cohort-out median and Gaussian-calibrated MAD | Yes |
| `rolling_iqr` | Local Tukey fences using linear-interpolated quartiles | Yes |
| `local_linear` | Elapsed-time local Huber regression; robust residual scale | Yes |
| `local_piecewise` | Independent left/right robust slopes projected to the target; disagreement abstains | Yes |
| `jump_reversion` | Large deviation between agreeing before/after medians | Yes |
| `multiscale` | Hampel confirmation across 1×, 2× and 4× count/time neighborhoods | Yes |
| `robust_trend` | Joint Huber-loss / first-difference total-variation level estimation per segment | Yes |
| `consensus` | Hampel or local-linear candidate, confirmed by median or independent side-trend reversion; protects persistent level differences | Yes |
| `causal_ewma` | Clipped past-only state, provisional innovations and persistent-change confirmation | No |

The API keeps `consensus` as its initial method; it is a starting configuration, not a validated optimum for every bond. For historical fitting, future observations are useful evidence. Compare `local_piecewise` around a rise-then-fall turning point, `local_linear` on a stable slope, and `multiscale` on contaminating bursts. `robust_trend` estimates piecewise **constant** levels, not a second-order smooth trend: its penalty is on ordered differences, without elapsed-time weights. Longer biased runs can be absorbed as regimes. `causal_ewma` is an online comparator whose prior provisional flags are never revised.

`window` limits neighboring **distinct timestamps**. Centered local references exclude the target cohort; each reference timestamp contributes its median. The joint `robust_trend` baseline includes the target with bounded influence, while its diagnostic scale excludes it. `horizon` bounds time in the chosen clock; `max_gap` splits the series and resets state. References never cross CUSIPs or segment boundaries. Missing centered support is not backfilled from the other side. Insufficient support is an explicit abstention.

## Sessions, irregular events and changing liquidity

The browser and notebook start with a configurable **trading clock**. The API preserves `time_basis="wall"` for compatibility; select `"trading"` explicitly for cumulative open-session time. Regular weekday hours, timezone and full-day holidays are editable. The sample 08:00–18:30 New York schedule is a research assumption; OTC bonds do not have one universal trading calendar. Use an authoritative interval table for your market, including early closes and special sessions. [SIFMA](https://www.sifma.org/resources/general/holiday-schedule) publishes fixed-income holiday and early-close recommendations; these should not be substituted with an equity calendar.

```python
# SETUP LOGIC: the session schedule is caller-supplied; all boundaries are aware.
import pandas as pd
# CONFIGURATION LOGIC: example early close and next regular session, [open,close).
schedule = pd.DataFrame({"open": ["2026-11-27T08:00-05:00", "2026-11-30T08:00-05:00"],
                         "close": ["2026-11-27T14:00-05:00", "2026-11-30T18:30-05:00"]})
# FILE IO LOGIC: the table overrides regular weekdays, hours and holidays.
review = filter_trades(df, config, session_schedule=schedule)
```

The trading clock integrates the open-session indicator. It removes declared overnight/weekend/holiday closures, while an 80-minute no-trade interval **during** an open session remains 80 minutes. It does not insert observations, assume constant liquidity, or assume no overnight repricing. Outside-session trades remain in the output as `outside_session`, with no fitting eligibility. The plot preserves actual UTC time. Calendar interval fingerprints and the clock choice are recorded in output metadata.

Count plus clock-time bounds adapt neighborhood length to event density. Local MAD, IQR, or detrended residual scale adapts the deviation band to local noise. Diagnostics expose support, reference span, density, gaps and scale. A sudden density or noise change can still make the transition uncertain: `local_piecewise` abstains when independent side predictions disagree, rather than labeling that disagreement an execution anomaly.

All methods use a raw deviation boundary of at least `abs_floor`. Scores are standardized magnitudes, not probabilities. `jf_weight` is an optional bounded-influence suggestion, not a calibrated likelihood or a replacement for review. Regime markers describe affected rows, not unique changepoint events. Dense persistent markup flow may be indistinguishable from a genuine level move with only these fields.

## Outputs and exports

| Field | Meaning |
|---|---|
| `jf_is_outlier`, `jf_status`, `jf_reason` | Flag, evaluation/abstention state, explanation |
| `jf_baseline`, `jf_scale`, `jf_residual` | Local diagnostic reference, scale, signed spread deviation |
| `jf_score`, `jf_threshold` | Absolute standardized score and raw-unit deviation cutoff |
| `jf_n_reference`, `jf_regime_change` | Distinct-time support and persistent-level candidate |
| `jf_weight`, `jf_row_id`, `jf_time`, `jf_method` | Suggested influence, source position, normalized UTC time, method |
| `jf_fit_eligible` | Final hard-fitting decision; optional quantity rules can explicitly retain uncertain rows |
| `jf_clock_time`, `jf_gap_minutes` | Selected-clock nanoseconds and gap from prior valid record, in minutes |
| `jf_wall_gap_minutes`, `jf_session_boundary` | Original UTC gap and a crossed declared closure |
| `jf_algorithm_*` | Original status, flag, reason, weight and eligibility when optional rules are enabled |
| `jf_quantity_notional`, `jf_policy_*` | Normalized quantity and rule decisions when optional rules are enabled |
| `jf_support_*` | Independent support trend, reliability, deviation, and bp distance for optional rules |
| `jf_reference_span_minutes`, `jf_reference_density_per_hour` | Actual reference range and distinct-cohort count / hours; undefined spans stay missing |
| `jf_n_votes`, `jf_n_scales` | Multiscale confirmations and evaluable scales |
| `jf_solver_converged`, `jf_solver_iterations` | Huber-TV convergence state; nonconvergence abstains for the whole segment |

`summarize` reports all source rows, evaluated and accepted counts, flags, invalid inputs, limited support, regime candidates, score statistics, flag rate and coverage. Flag rate divides by evaluated rows. Coverage divides by every row; missing support cannot inflate apparent accuracy.

The dashboard exports complete annotations for the applied review population, per-bond summaries, the exact applied settings and a standalone interactive HTML chart. Settings identify the selected-bond or all-bonds scope. No proprietary data are bundled or uploaded to a service. The repository demonstration is generated locally with a fixed seed.

## CLI and validation

```bash
# FILE IO LOGIC: write audit annotations and an optional population summary.
jump-filter trades.csv annotated.csv --method consensus --summary summary.csv
# TEST LOGIC: validate correctness and compare reproducible synthetic scenarios.
python -m pip install -e '.[dev]'
python -m pytest -q
```

See [all nine mathematical explanations](docs/MATHEMATICS.md), [offline solver details](docs/OFFLINE_METHODS.md) and [research references](docs/RESEARCH.md). See [historical fitting validation](docs/VALIDATION.md) for measured results, coverage, reproducibility and limits. Synthetic precision/recall are evidence about the included scenarios, not real bond-trade accuracy.

The eleven synthetic bonds include continuous turns, abrupt drops, changing liquidity and genuine market-closure gaps. Historical screening can use future trades. Choose hard exclusion or soft influence explicitly with `select_fit_data`, and calibrate parameters on real observations.
