# Validation for historical fitting

The evaluation objective is to choose observations suitable for a later spread or mid-price fit. Eight methods can use future trades; causal EWMA remains an optional comparison. The benchmark separates anomaly recovery, exclusion of good trades, quality of retained observations, accuracy of the algorithm's reference, and the result of one fixed downstream fit. A low error with little usable support is not evidence of a useful filter.

The expanded suite contains **19 independent construction scenarios × 3 held-out seeds × 9 methods = 513 paired trials**. The held-out seeds are `32452843`, `49979687`, and `67867967`, distinct from the earlier demonstration seeds `104729`, `130363`, and `155921`. The constructions and seeds are fixed; no method receives scenario-specific numerical tuning or synthetic truth as an input feature. The trading calendar is an explicit input convention for the calendar scenario, not a parameter optimized on labels.

These trials recover deliberately injected distortions. They do not estimate precision or recall on actual TRACE or BondCliQ records, establish an economically identified mid, or identify retail, commission, or distressed trades. Scores and suggested influence weights are not calibrated probabilities.

## Reproduce

```bash
# TEST LOGIC: Verify row integrity, mathematical behavior, fitting policies, and session clocks.
python -m pytest tests -q
# FILE IO LOGIC: Save the full fixed-default historical-fitting comparison locally.
python -m benchmarks.compare --n 480 --seeds 32452843 49979687 67867967 --output reports/offline-heldout
# TEST LOGIC: A small explicit subset helps investigate a failure without changing constructions.
python -m benchmarks.compare --n 480 --methods local_linear local_piecewise --scenarios turning_continuous turning_drop --output reports/turning
```

The runner writes detailed `synthetic_trials.csv`, per-scenario `synthetic_summary.csv`, `manifest.json`, and `SYNTHETIC_COMPARISON.md`. The manifest records the full configuration and time basis. Reports are generated from the stated seeds and require no proprietary trade file; local `reports/` files are ignored by Git. A compact committed snapshot is available in [SYNTHETIC_SUMMARY.csv](SYNTHETIC_SUMMARY.csv) and [SYNTHETIC_MANIFEST.json](SYNTHETIC_MANIFEST.json). The benchmark passes only `CUSIP`, `time`, and `spread` to the filter, then attaches construction truth for evaluation.

## What is being simulated

Ordinary scenarios contain 480 distinct timestamp cohorts, normally one per hour. Sparse scenarios contain 160. All values use synthetic basis points. The efficient spread begins at 100 bp; ordinary clean noise has standard deviation 0.6 bp. Genuine efficient-market changes are never blanket bad-trade labels.

| Scenario | Efficient-market behavior and trade construction |
|---|---|
| `clean` | Constant efficient spread; no injected bad trades. |
| `noisy_flat` | Constant efficient spread with isolated offsets of 12–25 bp, mixed signs. |
| `trend` | Efficient spread increases 0.03 bp per cohort; isolated offsets remain distinct from the trend. |
| `regime` | Efficient spread permanently increases 16 bp at the midpoint; injected errors avoid an eight-cohort transition neighborhood. |
| `sparse` | Independent 8–19 hour interarrivals, fewer local reference cohorts, and isolated offsets. |
| `burst` | Three four-cohort positive distortion episodes, surrounded by a constant efficient level. |
| `one_sided` | Isolated positive 12–25 bp trade premiums rather than symmetric contamination. |
| `dense_gradual` | Two-minute observations; efficient spread is `100 + 0.12 i + 3 sin(i/35)`, with smooth drift and curvature. |
| `steep_trend` | Efficient spread increases 0.75 bp per cohort; injected deviations are smaller, 7–12 bp. |
| `clustered_40pct` | Three local blocks; 40% of each block is contaminated in two short runs, with genuinely clean prints inside each block. The full sample is not 40% contaminated. |
| `bidask_bimodal` | Entirely clean labels: symmetric ordinary side prints around ±2 bp, plus rare legitimate wider side prints around ±7 bp. These are not assumed to be commissions or bad trades. |
| `jump_adjacent` | A genuine 16 bp repricing plus explicit bad trades at indices `midpoint−2` through `midpoint+1`. |
| `heteroscedastic` | Clean noise standard deviation changes from 0.6 to 3 bp; isolated 12–25 bp offsets are injected independently. |
| `irregular` | Interarrivals span 0.05–8 hours on a log-uniform distribution; a 40-hour inactivity gap separates support. |
| `sparse_mixed` | Sparse 8–19 hour arrivals with a dense 15-minute middle region and a genuine 16 bp level change. |
| `turning_continuous` | Efficient spread rises 0.3 bp per cohort, then declines 0.5 bp per cohort, continuously at the turning point. |
| `turning_drop` | The rising path drops 20 bp at the midpoint, then declines 0.25 bp per cohort; errors also occur immediately around the transition. |
| `calendar_sessions` | Irregular weekday observations only from 08:00 to 18:30 New York time, about 20 per session, with overnight and weekend closures. A genuine 20 bp repricing occurs at the first Monday open after Thursday/Friday sessions, alongside independent isolated trade distortions. The benchmark explicitly uses the trading clock. |
| `liquidity_shift` | Three regions: two-hour arrivals/0.6 bp noise, one-minute arrivals/0.2 bp noise, and eight-hour arrivals/3 bp noise. Density and variance changes do not label an entire regime bad. |

Ordinary isolated contamination uses `max(4, floor(0.035 n))` interior observations. The sparse cases have fewer distortions because their cohort count is lower. The turning-drop and jump-adjacent cases additionally force transition-adjacent errors. The clean and bimodal cases have no injected distortion. This is still a small family of synthetic price/noise models, not a comprehensive model of bond microstructure.

Calendar sessions use Monday–Friday dates and the stated New York hours; no exchange holiday calendar is inferred. Public tests separately supply a holiday and an explicit session schedule with an early close. In `time_basis="trading"`, durations are accumulated open-session time: `horizon="3D"` means 72 open hours, not three sessions. Closed overnight/weekend time does not age support, but a genuine absence of trades while the session is open does. Most scenarios retain the API's backward-compatible wall-clock basis.

Removing closed time from observation age does **not** assume the efficient spread stays constant overnight. Information and genuine repricing can arrive while trades are absent. The calendar case therefore includes a real Monday opening level change, and the engine separately reports actual wall-clock gaps and session boundaries. A short trading-clock gap alone cannot prove that a new-session print is contaminated; the retrospective methods must inspect future support or abstain around an uncertain transition.

## Metrics and target denominators

For construction label `b_i` and exported outlier flag `f_i`, precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and clean false-positive rate is `FP/(FP+TN)`. A bad trade with insufficient support or a failed solver still counts as a false negative. Undefined rates remain `NaN`; a clean case with no flags has undefined precision and recall, rather than an invented 100% score.

**Scored coverage** counts supported decision statuses with positive influence weight. **Fit eligibility** uses the exported `jf_fit_eligible` field and additionally checks that the row is supported and unflagged. **Clean retention** is the fraction of all truly clean observations retained. **Exclusion fraction** includes rejected, unsupported, ambiguous-transition, invalid, and solver-failure rows. These denominators expose the cost of avoiding uncertain points.

The retained raw-trade error is

\[
\operatorname{RMSE}_{\rm retained}=
\sqrt{|A|^{-1}\sum_{i\in A}(y_i-m_i)^2},
\qquad
A=\{i:\text{fit eligible, supported, unflagged}\}.
\]

Here `m_i` is the constructed latent efficient spread, not a midpoint observed by the filter. This error is conditional on acceptance. Deleting every difficult point can reduce it while leaving too little data to fit; it must be read alongside eligibility and clean retention.

The engine reference error evaluates `jf_baseline` on **all common evaluable clean targets**, including clean observations rejected by the method. A target is externally evaluable when the fixed downstream fit using all observations can support it. Missing algorithm references reduce reported reference coverage; they do not silently disappear from the target denominator. Reference RMSE remains undefined if no finite estimate exists.

A separate **fixed downstream local OLS** evaluates the fitting objective. For target time `t_i`, it uses up to 21 nearest accepted timestamp cohorts within 72 hours of the appropriate clock, requires at least six, and does not borrow across an adjacent gap exceeding 24 hours. It minimizes

\[
\min_{a_i,c_i}\sum_{j\in N_i\cap A}
\bigl(y_j-a_i-c_i(t_j-t_i)\bigr)^2,
\qquad \widehat m_i=a_i.
\]

Every method receives the same downstream estimator and the same clean target set. Rejected clean targets remain in evaluation. The unfiltered comparison fits the identical estimator with all observations. Downstream error is reported together with finite-estimate coverage. An unsupported segment yields missing estimates, not a zero error.

This OLS fit is deliberately ordinary rather than robust so that deleting distorted observations has a measurable consequence. It is a diagnostic downstream model, not a claim that this is the best bond mid-price model. It also smooths across real jumps, so genuine transition regions can have material fit error even with an excellent filter. A production model with explicit change points, trade-side adjustment, size/venue controls, or other predictors can give a different ranking.

Pooled RMSE uses the sum of squared errors divided by the sum of corresponding target counts before taking a square root. It is not the average of scenario RMSEs. Per-scenario tables show means and sample standard deviations across independent construction seeds; dependent trade rows are not treated as independent replicates. Solver convergence, solver-failure row fraction, and iteration limits remain explicit in the detailed reports.

## Measured pooled results

The completed comparison contains 513 trials and 228,960 observed rows across methods. Each method sees the same 25,440 observations: 886 injected distortions and 24,554 genuinely clean trades. The table pools counts rather than averaging per-scenario classification rates.

| Method | TP / FP / FN | Precision | Recall | Clean false positives | Fit eligible |
|---|---:|---:|---:|---:|---:|
| Hampel | 761 / 40 / 125 | 95.01% | 85.89% | 0.163% | 96.81% |
| Rolling IQR | 576 / 19 / 310 | 96.81% | 65.01% | 0.077% | 97.62% |
| Local linear | 727 / 34 / 159 | 95.53% | 82.05% | 0.138% | 96.97% |
| Jump reversion | 621 / 34 / 265 | 94.81% | 70.09% | 0.138% | 96.01% |
| Multiscale | 700 / 24 / 186 | 96.69% | 79.01% | 0.098% | 97.15% |
| Local piecewise | 612 / 41 / 274 | 93.72% | 69.07% | 0.167% | 91.25% |
| Robust trend | 788 / 138 / 98 | 85.10% | 88.94% | 0.562% | 96.32% |
| Consensus | 787 / 27 / 99 | 96.68% | 88.83% | 0.110% | 95.39% |
| Causal EWMA | 612 / 93 / 274 | 86.81% | 69.07% | 0.379% | 92.48% |

The unfiltered raw-trade RMSE is **3.601 bp**. The fixed downstream fit without filtering has clean-target RMSE **1.132 bp**. The next table uses squared-error pooling, and retains separate denominators for finite reference and downstream estimates.

| Method | Accepted raw RMSE (bp) | Reference RMSE (bp) | Reference coverage | Fixed-fit RMSE (bp) | Fixed-fit coverage |
|---|---:|---:|---:|---:|---:|
| Hampel | 1.457 | 0.940 | 99.98% | 0.628 | 99.97% |
| Rolling IQR | 2.197 | 1.209 | 99.98% | 0.925 | 99.97% |
| Local linear | 1.831 | 0.893 | 99.98% | 0.879 | 99.97% |
| Jump reversion | 2.107 | 1.246 | 99.98% | 0.780 | 99.86% |
| Multiscale | 1.749 | 2.449 | 100.00% | 0.654 | 100.00% |
| Local piecewise | 1.674 | 1.191 | 98.55% | 0.881 | 99.11% |
| Robust trend | 1.401 | 0.278 | 99.98% | 0.760 | 99.97% |
| Consensus | 1.581 | 0.970 | 99.52% | 0.720 | 99.86% |
| Causal EWMA | 1.657 | 1.639 | 95.57% | 0.843 | 97.20% |

All 57 robust-trend trials converged in 64–1,063 iterations with the documented 2,000-iteration limit and tolerance `1e-4`; the final reports have zero solver-failure rows. Deliberately limited-iteration tests still require honest abstention. Numerical convergence does not imply correct economic classification.

Hampel gives the lowest pooled error for this fixed downstream fit. Robust trend gives the lowest error for its own reference, but rejects more clean trades, especially in steep and turning paths. The difference matters: a smooth reference and a good selection of raw fitting observations are separate outcomes. These constructions do not support replacing every other method with one default winner.

## Selected stress results

Values below are means across the three held-out seeds. The complete per-scenario CSV includes all nine methods and seed standard deviations.

| Stress | Method | Distortion recall | Clean false positives | Fixed-fit RMSE (bp) | Fixed-fit coverage |
|---|---|---:|---:|---:|---:|
| Steep trend | Local linear | 100.00% | 0.000% | 0.134 | 100.00% |
| Steep trend | Consensus | 97.92% | 0.000% | 0.143 | 100.00% |
| Steep trend | Robust trend | 100.00% | 3.664% | 0.135 | 100.00% |
| Steep trend | Multiscale | 0.00% | 0.000% | 0.423 | 100.00% |
| 40% local contamination | Hampel | 100.00% | 0.079% | 0.136 | 100.00% |
| 40% local contamination | Multiscale | 100.00% | 0.000% | 0.136 | 100.00% |
| 40% local contamination | Local linear | 25.73% | 0.000% | 2.838 | 100.00% |
| 40% local contamination | Rolling IQR | 0.00% | 0.000% | 3.008 | 100.00% |
| Sparse | Multiscale | 100.00% | 0.000% | 0.186 | 100.00% |
| Sparse | Local piecewise | 73.33% | 2.796% | 0.196 | 71.06% |
| Sparse | Causal EWMA | 20.00% | 0.215% | 0.371 | 10.35% |
| Continuous turn | Local linear | 100.00% | 0.000% | 0.243 | 100.00% |
| Continuous turn | Local piecewise | 95.83% | 0.072% | 0.316 | 100.00% |
| Continuous turn | Robust trend | 100.00% | 1.868% | 0.320 | 100.00% |
| Abrupt turning drop | Local linear | 84.74% | 0.000% | 0.984 | 100.00% |
| Abrupt turning drop | Local piecewise | 74.65% | 0.073% | 1.053 | 100.00% |
| Abrupt turning drop | Robust trend | 98.33% | 1.376% | 1.050 | 100.00% |
| Business sessions + Monday jump | Consensus | 96.08% | 0.000% | 1.311 | 100.00% |
| Business sessions + Monday jump | Local piecewise | 92.16% | 0.072% | 1.445 | 100.00% |
| Business sessions + Monday jump | Causal EWMA | 98.04% | 0.576% | 1.633 | 100.00% |
| Liquidity/variance changes | Consensus | 93.75% | 0.431% | 0.489 | 100.00% |
| Liquidity/variance changes | Multiscale | 93.75% | 1.509% | 0.515 | 100.00% |
| Liquidity/variance changes | Local piecewise | 81.25% | 0.144% | 0.537 | 99.57% |

The business-session fit error includes the real Monday repricing, where an ordinary local linear downstream model smooths across the jump. In the explicit opening-regression fixture, every retrospective method retained or abstained on the clean new-session observations rather than flagging the genuine opening change. Causal EWMA temporarily flags the first two clean prints because it has no future confirmation.

In the all-clean bid/ask case, the false-positive rate on **rare legitimate wider side trades** is 0% for IQR, 2.08% for multiscale, 6.25% for local piecewise, 8.33% for Hampel/local linear, 10.42% for consensus, 12.50% for jump reversion, 14.58% for robust trend, and 18.75% for causal EWMA. These are wrongful removals under the construction labels. The overall clean false-positive rates are smaller because wider side trades are rare. Their separate metric prevents that dilution from hiding the problem.

## Interpreting the comparison

A stationary median or IQR band can have excellent precision while missing smaller trade distortions on a steep trend: the genuine slope inflates the reference dispersion. Local linear methods address smooth trends but can be biased near slope reversals or true jumps. The local piecewise rule uses separate past/future slopes and abstains when their projected levels disagree; that avoids labeling a genuine repricing bad, at the cost of losing fitting candidates near uncertain transitions.

Multiple scales can find contaminated local blocks that defeat one local fit, but a median-only multiscale rule is still sensitive to genuine drift. IQR fences protect wide ordinary observations, and can consequently miss widespread one-sided contamination. Robust Huber/TV smoothing jointly estimates a historical trajectory and limits isolated print influence, but its penalization can flatten legitimate sharp curvature or fit a sufficiently persistent contaminated cluster as a level change. Its convergence is a prerequisite for using its output.

The bid/ask construction is a useful negative control: all side prints, including rare wider ones, have clean labels. A nonzero false-positive rate in that scenario measures wrongful removal rather than successful recovery of commissions. Lower raw spread variance alone cannot justify classifying these observations as bad.

Sparse and changing-liquidity cases make the error/coverage tradeoff visible. A causal model can retain very little data when support ages beyond its horizon; a low conditional RMSE then says little about the resulting fit. The purpose of comparing methods is to expose these failure modes, not to declare a universal winner from this synthetic family.

## Behavioral checks

The tests cover original column/index/row preservation and no caller mutation; CUSIP isolation; unordered input; equal-time cohort order invariance; duplicate prints not increasing reference support; invalid identifiers, prices and timestamps; DST ambiguity; unsupported sparse segments; zero-MAD data; isolated offsets; genuine lasting changes; smooth and steep trends; scale/translation equivariance with nondefault method controls; exact exported threshold/flag consistency; bounded influence weights; and hard versus soft fitting policies.

Historical tests explicitly require later support to change an earlier estimate; causal tests require prefix invariance when future rows are removed or changed. Numerical tests check Huber/TV convex objectives and solver diagnostics, and require a failed solver to abstain with zero weight and no fit eligibility. Session tests distinguish closed overnight/weekend/holiday time from genuine intraday inactivity, require aware timestamps in supplied schedules, and verify clock/span/density diagnostics by CUSIP. Separate metric tests verify that rejected clean rows remain in fit-error evaluation, that positive weight alone cannot override an unsupported status, and that solver failures remain visible.
