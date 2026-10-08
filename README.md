# jump_filter

An auditable bond-trade spread filter and research dashboard. Pass a pandas DataFrame with a bond identifier, timestamp and spread; retain every row and receive outlier flags, local references, scores, thresholds, support counts and explanations.

The engine screens statistical deviations. These three columns cannot identify retail trades, commissions, distressed sellers, or the true market mid. Zero and negative spreads are valid observations. Input spread units are preserved.

## Install and launch

```bash
# SETUP LOGIC: install the independent package and interactive extras.
python -m pip install -e '.[dashboard,app,parquet]'
# UI LOGIC: start the local browser dashboard.
python -m streamlit run jump_filter/app.py
```

Open **jump_filter_dashboard.ipynb** for the notebook dashboard, or use the browser dashboard with a synthetic demonstration or CSV upload. Both provide CUSIP and method selection, paired sliders and exact inputs, time-series plots, red outlier markers, reference bands, residual diagnostics, per-bond statistics, method comparisons and exports. Apply publishes a reproducible snapshot; pending settings do not silently alter downloads.

## DataFrame API

```python
# SETUP LOGIC: imports do not read data or open a dashboard.
from jump_filter import FilterConfig, filter_trades, summarize, show_filter

# CONFIGURATION LOGIC: every spread threshold uses the input spread unit.
config = FilterConfig(method="consensus", window=31, horizon="3D",
                      max_gap="1D", min_neighbors=6, threshold=4.5,
                      abs_floor=1.0)

# FILE IO LOGIC: annotate a caller-supplied DataFrame; its rows are preserved.
review = filter_trades(df, config, cusip_col="CUSIP", time_col="time",
                       spread_col="spread", timezone="America/New_York")

# UI LOGIC: inspect every bond, then open the notebook workbench.
print(summarize(review))
dashboard = show_filter(df, config=config, cusip_col="CUSIP", time_col="time",
                        spread_col="spread", timezone="America/New_York", unit="bp")

# Input: statuses=['ok','outlier','insufficient_history'],flags=[False,True,False].
# Output: eligible=[True,True,False],fit_data contains only the first source row.
# Trick: abstentions are excluded even when their outlier flag is False.
# CORE LOGIC: STEP 1
eligible = review["jf_status"].isin(["ok", "outlier", "provisional_jump"])
fit_data = review.loc[eligible & ~review["jf_is_outlier"]].copy()
```

`show_filter` takes the source frame; annotations reserve the `jf_` prefix. Column mappings must be distinct. Original order, duplicate index labels and original columns remain intact. Naive timestamps use the configured timezone; aware timestamps convert to UTC. Invalid dates, ambiguous/nonexistent DST times, numeric epochs, nonfinite spreads and missing/blank identifiers receive `invalid_input` and zero suggested fit weight. Convert numeric epochs explicitly before passing them.

For BondCliQ, map `spread_col="BM_SPREAD"` when appropriate. If your `BM_SPREAD` is in percentage points, an `abs_floor` of `0.01` equals 1 bp. A display label does not convert data; convert percentage points to bp explicitly before using a 1-bp floor.

## Methods

| Method | Local reference and flag rule | Future data |
|---|---|---|
| `hampel` | Leave-cohort-out median and Gaussian-calibrated MAD | Yes |
| `local_linear` | Elapsed-time local Huber regression; robust residual scale | Yes |
| `jump_reversion` | Large deviation between agreeing before/after medians | Yes |
| `consensus` | Hampel or local-linear candidate, confirmed by median or independent side-trend reversion; protects persistent level differences | Yes |
| `causal_ewma` | Clipped past-only state, provisional innovations and persistent-change confirmation | No |

The conservative retrospective default is a starting configuration, not a validated optimum for every bond. `causal_ewma` never revises earlier flags: initial prints of a genuine shift can be provisional anomalies; confirmation accepts the current cohort and resets history. Prefer retrospective confirmation for historical cleaning and causal scoring for a live pipeline.

`window` limits neighboring **distinct timestamps**. A target timestamp and every trade sharing it are excluded from its retrospective references. Each reference timestamp contributes its median. `horizon` additionally bounds elapsed time; `max_gap` splits the series and resets state. There is no borrowing across CUSIPs, or across a long trading gap. Retrospective windows split the count between earlier and later timestamps without backfilling a missing side. Insufficient support is an explicit abstention, not a normal trade.

All methods use a raw deviation boundary of at least `abs_floor`. Scores are standardized magnitudes, not probabilities. `jf_weight` is an optional bounded-influence suggestion, not a calibrated likelihood or a replacement for review. Regime markers describe affected rows, not unique changepoint events. Dense persistent markup flow may be indistinguishable from a genuine level move with only these fields.

## Outputs and exports

| Field | Meaning |
|---|---|
| `jf_is_outlier`, `jf_status`, `jf_reason` | Flag, evaluation/abstention state, explanation |
| `jf_baseline`, `jf_scale`, `jf_residual` | Local diagnostic reference, scale, signed spread deviation |
| `jf_score`, `jf_threshold` | Absolute standardized score and raw-unit deviation cutoff |
| `jf_n_reference`, `jf_regime_change` | Distinct-time support and persistent-level candidate |
| `jf_weight`, `jf_row_id`, `jf_time`, `jf_method` | Suggested influence, source position, normalized UTC time, method |

`summarize` reports all source rows, evaluated and accepted counts, flags, invalid inputs, limited support, regime candidates, score statistics, flag rate and coverage. Flag rate divides by evaluated rows. Coverage divides by every row; missing support cannot inflate apparent accuracy.

The dashboard exports full annotations, per-bond summaries, the exact applied settings and a standalone interactive HTML chart. No proprietary data are bundled or uploaded to a service. The repository demonstration is generated locally with a fixed seed.

## CLI and validation

```bash
# FILE IO LOGIC: write audit annotations and an optional population summary.
jump-filter trades.csv annotated.csv --method consensus --summary summary.csv
# TEST LOGIC: validate correctness and compare reproducible synthetic scenarios.
python -m pip install -e '.[dev]'
python -m pytest -q
```

See [research and references](docs/RESEARCH.md) for formulas, assumptions and primary literature. See [synthetic validation](docs/VALIDATION.md) for measured results, reproducibility and limits. Synthetic precision/recall are evidence about the included scenarios, not real bond-trade accuracy.

中文：Notebook 和浏览器界面均支持选 CUSIP、切换方法、slider + 精确输入框、异常点标记、统计表与结果导出。引擎只标记、不删除原数据；真实 mid 拟合应显式选择可评估且未被标记的记录，并在实际数据上校准参数。
