# Validation and synthetic comparison

The engine passed **78 behavioral tests** in 5.67 seconds in the development environment. The comparison below contains **105 trials**: seven construction scenarios, three independent seeds, and five methods. Every method received the same observed trades and its documented default configuration. These results measure recovery of deliberately injected distortions. They do not estimate accuracy on real TRACE or BondCliQ data, identify retail customers, or establish a fitted mid price.

## Reproduce

From an installed checkout:

```bash
# TEST LOGIC: Verify public behavior and independent benchmark metrics.
python -m pytest tests -q
# FILE IO LOGIC: Save reproducible construction-truth comparisons locally.
python -m benchmarks.compare --n 480 --seeds 104729 130363 155921 --output reports
```

The runner writes `synthetic_trials.csv`, `synthetic_summary.csv`, `manifest.json`, and `SYNTHETIC_COMPARISON.md`. The local `reports/` directory is ignored by Git. The committed runner generates the underlying trades from the stated seeds; no proprietary trade file is required. Tests import the independent evaluation code as well as the public engine API.

Defaults are `window=31`, `horizon='3D'`, `max_gap='1D'`, `min_neighbors=6`, `threshold=4.5`, `abs_floor=1.0`, `reversion_tolerance=2.0`, `alpha=0.2`, and `persistence=3`. Spread values and `abs_floor` use the input spread unit. Reversion tolerance, score threshold, and EWMA alpha are dimensionless.

## Construction and metrics

Each ordinary scenario has 480 timestamp cohorts; sparse cases have 160. Ordinary trades occur hourly. Sparse cases have independent interarrival times of 8–19 hours. The efficient spread starts at 100 synthetic basis points. Ordinary noise has standard deviation 0.6 bp. Trend cases add 0.03 bp per observation; regime cases permanently add 16 bp at the midpoint. Those efficient-market changes are clean observations.

The `clean` case has no injected distortion. `noisy_flat`, `trend`, `regime`, and `one_sided` cases inject 16 distortions per seed, and `sparse` injects five. Distortions are uniformly drawn between 12 and 25 bp; ordinary cases mix signs, while `one_sided` uses positive premiums. `burst` injects three four-trade positive episodes, totaling 12 distorted trades per seed. Regime injections exclude the immediate midpoint neighborhood, and isolated injections exclude 20 observations at each endpoint. The benchmark therefore does not measure detection of contaminated endpoints or every possible overlap between a jump and a distortion.

Each method processes 9,120 observed rows across its 21 trials: 243 injected distortions and 8,877 clean trades. Precision is TP/(TP+FP), recall is TP/(TP+FN), and the clean false-positive rate is FP/(FP+TN). Abstention counts as an unflagged observation for those classification measures; missed labeled distortions still count as false negatives.

**Scored coverage** is the fraction with positive `jf_weight`. **Retained RMSE** compares unflagged, positive-weight raw spreads with their constructed efficient spread; unsupported zero-weight observations are excluded. It measures contamination left in observations accepted for a subsequent fit. It does not measure fitted-mid RMSE. Coverage must be considered alongside RMSE because abstention can remove difficult observations. No uncertainty or calibration claim is attached to `jf_score` or `jf_weight`.

## Pooled results

Counts and classification rates below pool all 21 trials per method. Scored coverage is weighted by observation count. RMSE pools retained squared errors before taking the square root; it is not an unweighted mean of scenario RMSEs.

| Method | TP / FP / FN | Precision | Recall | Clean false positives | Scored coverage | Retained RMSE (bp) |
|---|---:|---:|---:|---:|---:|---:|
| Consensus | 241 / 7 / 2 | 97.18% | 99.18% | 0.079% | 98.62% | 0.688 |
| Hampel | 243 / 5 / 0 | 97.98% | 100.00% | 0.056% | 99.90% | 0.598 |
| Local linear | 240 / 14 / 3 | 94.49% | 98.77% | 0.158% | 99.90% | 0.697 |
| Jump reversion | 240 / 13 / 3 | 94.86% | 98.77% | 0.146% | 98.62% | 0.713 |
| Causal EWMA | 210 / 10 / 33 | 95.45% | 86.42% | 0.113% | 93.48% | 0.844 |

Hampel has the strongest pooled result in this particular construction. The injected isolated distortions are large relative to ordinary noise, and a median is well suited to them. This result supports including Hampel as a comparison and a practical baseline; it does not establish a universal winner.

## Consensus by scenario

Values below are means across the three seeds; the generated summary CSV also records sample standard deviations across seeds. Precision and recall are undefined for the unflagged clean case and are shown as a dash.

| Scenario | Precision | Recall | Clean false positives | Scored coverage | Original raw RMSE (bp) | Retained RMSE (bp) |
|---|---:|---:|---:|---:|---:|---:|
| Clean | — | — | 0.000% | 98.75% | 0.597 | 0.598 |
| Noisy flat | 100.00% | 100.00% | 0.000% | 98.75% | 3.391 | 0.601 |
| Trend | 100.00% | 100.00% | 0.000% | 98.75% | 3.391 | 0.601 |
| Regime | 100.00% | 100.00% | 0.000% | 98.75% | 3.400 | 0.600 |
| Sparse | 66.55% | 86.67% | 1.505% | 96.25% | 3.462 | 1.472 |
| Burst | 100.00% | 100.00% | 0.000% | 98.75% | 2.790 | 0.598 |
| One sided | 100.00% | 100.00% | 0.000% | 98.75% | 3.387 | 0.601 |

Sparse observations are the clearest failure mode at these defaults. Two-sided projections rely on fewer local observations, and causal history often falls below the six-cohort requirement within a three-day horizon. Sparse-case results across methods make that tradeoff explicit:

| Method | Precision | Recall | Clean false positives | Scored coverage | Retained RMSE (bp) |
|---|---:|---:|---:|---:|---:|
| Consensus | 66.55% | 86.67% | 1.505% | 96.25% | 1.472 |
| Hampel | 79.37% | 100.00% | 0.860% | 98.13% | 0.579 |
| Local linear | 55.56% | 80.00% | 2.151% | 98.13% | 1.492 |
| Jump reversion | 54.76% | 86.67% | 2.366% | 96.25% | 1.472 |
| Causal EWMA | — | 0.00% | 0.000% | 11.04% | 0.608 |

The causal sparse RMSE of 0.608 bp accompanies only 11.04% scored coverage and zero recovered distortions. It is not evidence of successful filtering. Causal EWMA also recovers only half of the four-trade burst distortions: persistence confirmation starts accepting the temporary new level. On genuine regime cases, the unavoidable onset ambiguity produces a 0.503% clean false-positive rate. Previously emitted causal decisions are never rewritten after confirmation.

## Default choice and practical use

`consensus` remains a useful default for **retrospective candidate review**: it requires reversion support plus a robust median or local-linear deviation, and it protects lasting changes. The trend and regime construction tests support those intended behaviors. Its endpoint abstention is intentional. The table gives no reason to use it exclusively: compare Hampel for stable and sparse bonds, and local linear when strong smooth drift makes a stationary median reference inappropriate.

Use `causal_ewma` when decisions must be made without future trades. Inspect scored coverage before using its output. Sparse bonds may need a longer horizon, a lower support requirement, or an explicit abstention policy; these choices alter stale-price and false-positive risk and should be assessed on representative labeled trades. The three columns cannot distinguish a true sudden repricing from a temporary trade premium at the moment it arrives.

Before automatic removal from a mid-price fit, review flagged and abstained observations on actual bonds, including bid/ask multimodality, trade-frequency changes, news events, and varying contamination proportions. The synthetic benchmark uses only three seeds and one fixed noise family. It does not test unknown commission schedules, customer/dealer side information, trade sizes, systematic widespread markup, or independent real-data labels.

## Behavioral test coverage

The 78 tests cover preservation of original columns, row order, duplicate indexes and caller data; isolation by CUSIP; unordered inputs; equal-time cohort order invariance; reference support unaffected by duplicate prints; invalid records; DST ambiguity and nonexistent times; numeric epoch rejection; empty inputs; configuration bounds, including `NaT` durations; insufficient support and zero weight; zero-MAD paths; isolated spikes; gaps; lasting repricing; trend spikes; dimensionally consistent scaling; custom mappings; and the bundled demo/statistics API.

Every method is checked for consistency between outlier flags, the exported/displayed residual threshold, and reduced influence weight. Causal prefix tests perturb future prices and truncate future rows while retaining a simultaneous-trade cohort, then require all earlier decisions and estimates to remain identical. Separate benchmark tests verify classification counts, coverage-aware error metrics, undefined clean-case precision/recall, independent seeds, and genuine regime changes excluded from contamination truth.
