# Research and design notes

`jump_filter` marks bond trades whose spread is unusual relative to nearby trades in the same CUSIP. Its input is a dataframe containing only CUSIP, time, and spread. It preserves the original records and reports diagnostic flags instead of silently deleting or replacing observations.

## What the three columns can identify

The observed spread can be thought of as a latent market level plus execution effects, spread-construction effects, and measurement noise. These components are not identifiable from three columns alone. A large deviation could be a customer markup, a markdown, a liquidity concession, a genuine credit event, an unusual but valid bid or offer, or a data problem.

Consequently, an `outlier` flag means **suspect for fitting a local central spread**. It does not establish that the trade was retail, distressed, incorrectly reported, or charged a commission. Identifying those causes would need additional information such as trade size, customer/dealer side, transaction type, quotes, execution venue, or market factors. The engine's `baseline` is a **robust local reference**, not an observed or proven mid price. With asymmetric order flow or a persistent customer-trade bias, even a robust reference can remain biased.

[Edwards, Harris, and Piwowar (2007)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01240.x) document that corporate-bond transaction costs decrease with trade size, supporting the motivation for screening execution-sensitive records. Their result does not identify the size or cause of any individual trade in this engine. [Choi and Huh (2017)](https://www.federalreserve.gov/econres/feds/customer-liquidity-provision-implications-for-corporate-bond-transaction-costs.htm) show that customers can also provide liquidity and that average customer bid/ask measures can understate the costs of demanding liquidity. That finding supports retaining explanations and reviewability rather than equating every unusual execution with bad data.

## Common time and scale conventions

- Each CUSIP is analyzed independently after sorting timestamps. An observation from one bond never establishes another bond's reference.
- Records at the same timestamp form a **cohort**. Centered methods exclude the entire target cohort from its reference neighborhood. This avoids self-inclusion and arbitrary dependence on input order for simultaneous trades.
- Each reference timestamp contributes its cohort's median spread. `window=31` caps the number of neighboring **distinct timestamps**, not raw trade records: at most 15 earlier and 16 later cohorts. Missing support on one side is not filled by adding more on the other. This prevents a burst of simultaneous prints from dominating the neighborhood merely because it has many rows.
- `horizon="3D"` limits how far reference timestamps can lie from the target. Observation-count and elapsed-time bounds apply together; a sparse bond does not borrow arbitrarily old observations to fill its window.
- `max_gap="1D"` starts a new segment after a longer trading gap. The resulting loss of reference coverage is explicit. It is safer to abstain than to infer an execution anomaly solely from a stale pre-gap level.
- `min_neighbors=6` requires enough neighboring distinct timestamps. Insufficient reference information produces an abstention rather than a definitive clean/outlier classification.
- `threshold=4.5` controls standardized deviation sensitivity. `abs_floor=1.0` provides a minimum absolute deviation boundary in the input spread's units. If spread is in basis points, the floor is one basis point; the engine does not silently convert units. With threshold $q$ and floor $a$, the internal scale floor is $a/q$, so the raw residual boundary $\max(a,q\sigma)$ is never smaller than $a$.

Centered methods are retrospective: later trades can change earlier references and flags. Their availability at the time of the target trade must not be assumed in a backtest. The `causal_ewma` method normally scores a whole same-time cohort using the state established by strictly earlier timestamps and then updates the state. A confirmed regime change is the explicit exception: the confirming cohort establishes a new reference before its individual rows are scored, as detailed below.

## Implemented methods

The five methods below share input validation, CUSIP separation, timestamp cohorts, gap segmentation, abstention, and diagnostic output. They offer different evidence for excluding a print. A threshold score is a diagnostic statistic, not a calibrated probability that a trade is bad.

### 1. `hampel`: local median and MAD

Let $z_j$ be the median spread of reference cohort $j$, and let $y_i$ be the raw spread of the target row. For the distinct-timestamp reference neighborhood $\mathcal{N}_i$, estimate a central spread with a median and dispersion with the median absolute deviation:

$$
m_i=\operatorname{median}_{j\in\mathcal{N}_i}(z_j),\qquad
\sigma_i=\max\left(1.4826\operatorname{median}_{j\in\mathcal{N}_i}|z_j-m_i|,a/q\right).
$$

The target is a candidate when $|y_i-m_i|>\max(a,q\sigma_i)$. A value exactly on the boundary is not flagged. The normal-consistency multiplier 1.4826 supplies a familiar scale; it does **not** assume bond-trade residuals are Gaussian or make the selected threshold a Gaussian significance test.

This method is simple and resistant to isolated extreme prints. It can miss clustered outliers when contaminated trades dominate its neighborhood, and can mistake the first prints of a real level change for outliers. Repeated spread values can make MAD zero, so the absolute boundary floor is essential. These limitations and the median/MAD formula are directly described in [Pearson et al. (2016)](https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf). Our timestamp cohorts, bounded irregular-time neighborhood, and annotation-only output are engineering adaptations of the Hampel identifier; this engine does not perform the article's replacement filter.

### 2. `local_linear`: robust local trend residual

Represent neighboring timestamps as elapsed time relative to the target, scaled by $d=\max_j|t_j-t_i|$ with a one-nanosecond numerical lower bound: $u_j=(t_j-t_i)/d$. A local linear model predicts a reference at the target:

$$
z_j\approx\beta_{0,i}+\beta_{1,i}u_j,\qquad \widehat y_i=\beta_{0,i}.
$$

Initialize the intercept at the neighborhood median and slope at zero. Six iteratively reweighted least-squares rounds combine elapsed-distance weights with Huber residual weights. At each round, with residual $r_j=z_j-\widehat z_j$ and its median-centered MAD scale $s$, use

$$
w_j^{\mathrm{time}}=\max\left([1-\min(|u_j|,0.999)^3]^3,0.05\right),\qquad
w_j^{\mathrm{robust}}=\min\left(1,\frac{1.345s}{\max(|r_j|,10^{-12})}\right).
$$

Both the iterative and final residual scales have floor $a/q$. The final detection rule is $|y_i-\widehat y_i|>\max(a,q\sigma_i^{\mathrm{residual}})$. The small time-weight floor retains numerical support at the neighborhood's furthest timestamps. Using elapsed time rather than row number allows unevenly spaced trades to describe the trend correctly.

This approach is useful when a bond's central spread drifts over the neighborhood. A constant median reference can classify an otherwise ordinary observation on a slope as unusual. A local line can still perform poorly across abrupt changes, highly curved trends, or neighborhoods with little distinct-time support. It therefore shares the gap and minimum-neighbor rules.

[Cleveland (1979)](https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf) motivates robust local fitting through proximity weights and iterative downweighting of residuals. `local_linear` is a small robust local-regression detector inspired by that principle; it is **not** a claim to reproduce the exact LOWESS/LOESS algorithm, bandwidth selection, inference, or published implementation.

### 3. `jump_reversion`: departure followed by return

A large move alone is weak evidence that an execution is unsuitable for estimating a central level. A move followed by a return to the old neighborhood is stronger evidence of a temporary print anomaly. For example, a sequence near 100, then 130, then near 100 is consistent with an isolated execution effect; a sequence near 100 followed by several trades near 130 is consistent with a new market level.

Each side must supply at least $\max(2,\lfloor\texttt{min\_neighbors}/2\rfloor)$ distinct timestamps; with the default, that means three earlier and three later cohorts. Let $m_L$ and $m_R$ be the two sides' median cohort spreads. Concatenate within-side deviations, $D=\{z_j-m_L:j\in L\}\cup\{z_j-m_R:j\in R\}$, and calculate

$$
b_i=(m_L+m_R)/2,\qquad
\sigma_i^{\mathrm{side}}=\max(1.4826\operatorname{median}|D|,a/q),\qquad
\Delta_i=|m_R-m_L|.
$$

`reversion_tolerance=2.0`, denoted $v$, controls agreement between the two neighborhoods. Reversion is supported when $\Delta_i\leq\max(a,v\sigma_i^{\mathrm{side}})$. The target is a reversal candidate only if that agreement holds **and** $|y_i-b_i|>\max(a,q\sigma_i^{\mathrm{side}})$. A raw continuing-shift candidate is $\Delta_i>\max(a,q\sigma_i^{\mathrm{side}})$; `jump_reversion` protects it. `consensus` also checks independent local-trend projections before protecting a shift, as described below. Standalone `hampel` and `local_linear` still report their own threshold decisions, making their behavior at shifts visible for comparison.

The two-sided confirmation concerns neighborhood medians; it does not require the very next trade to return. It can identify a short burst when both sides agree, but can miss a burst large enough to contaminate either side.

This is an engineering extension for the three-column problem. It is related to the broader distinction between isolated contamination and persistent structural changes. [Killick, Fearnhead, and Eckley (2012)](https://arxiv.org/abs/1101.1438) develop a formal changepoint framework; the engine's local reversal/shift guard is **not PELT**, does not solve their penalized segmentation objective, and carries no guarantee of recovering optimal changepoints.

Future observations are intrinsic to reversal confirmation. At the end of a series, or where either side has insufficient support, the engine cannot establish that a trade will revert. The output must preserve that distinction. A short-lived genuine market move can look like a bad execution; three columns cannot resolve that ambiguity.

### 4. `causal_ewma`: online innovations with regime persistence

The causal detector maintains a central reference from earlier timestamp cohorts. Its detection scale is the floored MAD of prior cohort medians within the horizon/count history, rather than an EWMA variance estimate. Let $z_c$ denote the current cohort's median spread and $m_{c-1}$ the existing central state. Its cohort innovation is

$$
e_c=z_c-m_{c-1}.
$$

For an unconfirmed cohort, the next central state uses a clipped exponentially weighted step, with $B_c=\max(a,q\sigma_c)$:

$$
m_c=m_{c-1}+\alpha\operatorname{clip}(e_c,-B_c,B_c),\qquad \alpha=0.2.
$$

For innovations inside the boundary, this equals the ordinary EWMA update. Clipping limits the influence of a single large cohort instead of freezing the reference completely. Alpha is **per timestamp cohort**, not adjusted for elapsed time: dense trading therefore adapts faster in clock time. The horizon limits supporting history and scale estimation; the central EWMA state can retain older information until it decays, resets at a gap, or is reset by confirmation.

Each raw trade is normally flagged when $|y_i-m_{c-1}|>B_c$ and receives `provisional_jump`. Persistence, however, is counted using cohort medians: `persistence=3` consecutive large cohort innovations with the same sign establish a continuing shift. A small cohort innovation or a sign change breaks the previous run. A confirmed third cohort resets the central reference to its own median and clears previous scale/support history. Its rows are scored against this new median, so its central prints are accepted while an extreme individual print within that cohort can still be flagged. Previous flags are **not revised**, which preserves causal prefix consistency.

All completed cohort medians, including provisional jumps, enter the supporting history. Repeated contamination can therefore inflate the historical MAD; this method is not immune to masking by a long burst. After confirmation clears that history, subsequent cohorts abstain until enough new support has accumulated. With the default six-neighbor requirement, the next five timestamp cohorts are warmup observations.

That policy deliberately exposes the unavoidable online tradeoff: the first one or two prints of a true change may be provisionally excluded before the third cohort supplies confirmation. A future-informed reversal filter can revisit those prints, while a causal detector cannot know the future. Report both detection coverage and these shift-boundary exclusions when assessing this method.

The method is a transparent adaptive engineering baseline, not a Kalman filter, a fitted latent-mid model, or a statistical changepoint test. It abstains during warmup and after gap resets. It cannot identify a regime change immediately without accepting more false positives from isolated prints.

### 5. `consensus`: raw or trend-adjusted reversal confirmation

Raw before/after medians can disagree along an ordinary smooth trend. Requiring their agreement alone would miss anomalous prints on that trend. Conversely, short contaminated bursts can distort a side's fitted slope while its median remains usable. The default therefore offers two reversal alternatives, then gates them with robust level or trend evidence.

The raw alternative $R_i^{\mathrm{raw}}$ is the `jump_reversion` candidate above. For the projected alternative, independently fit the same robust local-linear model on the earlier and later neighborhoods, each evaluated at the target timestamp. Denote these projected centers by $p_L$ and $p_R$, their residual scales by $s_L$ and $s_R$, and the full-neighborhood local-linear residual scale by $s_{\mathrm{all}}$. Define

$$
b_i^{\mathrm{trend}}=(p_L+p_R)/2,\qquad
\tau_i=\max(s_L,s_R,s_{\mathrm{all}}),\qquad
\Delta_i^{\mathrm{trend}}=|p_R-p_L|.
$$

Projected reversal is supported when $\Delta_i^{\mathrm{trend}}\leq\max(a,v\tau_i)$. The target is a projected reversal candidate $R_i^{\mathrm{trend}}$ only if that agreement holds **and** $|y_i-b_i^{\mathrm{trend}}|>\max(a,q\tau_i)$. All three component scales retain the $a/q$ floor. Taking their maximum reduces sensitivity to unrealistically small residual scales from short side fits.

For example, reference cohort times $[0,1,2,4,5,6]$ with spreads $[100,101,102,104,105,106]$ describe a line. At target time 3, the raw side medians are 101 and 105, while independent left and right projections are both 103, up to floating-point precision. A target spread of 130 is then a large deviation from an agreed trend projection even when raw-median agreement fails. This example illustrates the projection alternative; it is not evidence that every short trend is reliably estimated.

The robust candidate gate uses a candidate from either level or trend evidence:

$$
C_i=H_i\lor L_i,
$$

where $H_i$ is the Hampel candidate and $L_i$ is the robust local-linear candidate. Define a raw shift $S_i^{\mathrm{raw}}$ as above and a projected shift $S_i^{\mathrm{trend}}$ by $\Delta_i^{\mathrm{trend}}>\max(a,q\tau_i)$. Consensus protects a continuing shift only when **both** diagnostics indicate a large level difference. Its final rule is

$$
F_i=C_i\land(R_i^{\mathrm{raw}}\lor R_i^{\mathrm{trend}})
\land\neg(S_i^{\mathrm{raw}}\land S_i^{\mathrm{trend}}).
$$

This is a **reversal-confirmed union**, not majority voting. The raw alternative supports bursts whose side medians remain stable, and the projected alternative supports anomalous deviations along smooth trends. For a flagged row, raw reversal takes precedence when it confirms the flag; a trend-only confirmation uses the projected bridge and scale for the displayed reference, band, residual, score, and suggested weight. For an unflagged row, the raw reference is used when raw agreement holds and projected reversal does not; otherwise the projected reference is used. This row-specific choice keeps a flag consistent with the reference actually shown for that row.

Both alternatives need adequate support on both sides. Side projections extrapolate to the target, so sharp curvature, short histories, or contaminated sides can still mislead them. Unresolved series edges remain abstentions. Long bursts can contaminate both the medians and local fits. Protecting a shift when both diagnostics agree is an engineering safeguard, not proof that it was genuine market repricing.

The default is a reasoned starting configuration for exploratory bond cleaning. It is not asserted to be universally optimal or statistically calibrated. The component methods share neighborhoods and scales, so agreement does not constitute independent evidence.

## How to evaluate the engine

A comparison on real unlabeled trades can report the flag rate and inspect disagreements; it cannot establish precision or recall without reviewed labels. The synthetic demonstration provides known anomalous executions and known continuing level changes, allowing reproducible comparisons. Synthetic results demonstrate behavior under the generator's assumptions and do not prove performance on every TRACE or BondCliQ dataset.

Useful evaluation cases include isolated deviations in both directions, short contaminated bursts, smooth trends, permanent level shifts, unequal trading intensity, long gaps, ordinary bid/ask dispersion, repeated timestamps, zero MAD, and malformed input. Use several seeds and preserve test scenarios that were not used to choose parameters.

Report precision, recall, F1, and eligible coverage together. Recall should expose anomalies that were missed because the detector abstained; an eligible-only score can otherwise hide failure on sparse bonds. Also report the rate of falsely flagged genuine shift observations and runtime. A lower overall flag rate alone is not evidence of a better method.

For causal evaluation, append arbitrary future data and check that existing rows keep the same outputs. For retrospective methods, changing old outputs after appending data is expected and should be explained. Random row-level train/test splits are unsuitable for tuning a time-sensitive detector; use blocked time splits and hold out CUSIPs or stress scenarios where possible.

## Reading the dashboard

Inspect the selected bond's raw trades, local reference, threshold band, flagged points, protected shifts, and abstentions together. Review the flagged-record table and method disagreements before constructing a downstream fit. The aggregate tables should distinguish flagged count, eligible count, input count, and coverage. Compare bonds by eligible flag rate as well as the all-row rate, because sparse history changes the opportunity to detect anomalies.

Parameter controls are deliberately interpretable: a larger deviation threshold reduces sensitivity; a larger time/window neighborhood offers more support but can blur changing regimes; a larger floor reduces sensitivity to tiny moves and quantized spreads; a longer persistence requirement delays recognition of a new causal regime. There is no general parameter setting that removes all retail execution effects while retaining every true market change.

中文使用提示：这里标记的是“可能不适合拟合局部中心 spread 的交易”，不能仅凭三列判断它一定是 retail、commission 或 desperate trade。图中的 baseline 是稳健的局部参考，不是真实 mid。`consensus` 依赖后续交易确认回归，适合历史研究；`causal_ewma` 严格使用过去信息，但真实跳变最初几笔也可能暂时被排除。调整 `abs_floor` 前先确认 spread 的单位，并同时查看 coverage、protected shift 和 abstention。

## References

1. Edwards, A. K., Harris, L. E., and Piwowar, M. S. (2007). *Corporate Bond Market Transaction Costs and Transparency*. Journal of Finance, 62, 1421–1451. [Publisher and abstract](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01240.x).
2. Choi, J., and Huh, Y. (2017). *Customer Liquidity Provision: Implications for Corporate Bond Transaction Costs*. Finance and Economics Discussion Series 2017-116. [Federal Reserve paper page](https://www.federalreserve.gov/econres/feds/customer-liquidity-provision-implications-for-corporate-bond-transaction-costs.htm).
3. Brownlees, C. T., and Gallo, G. M. (2006). *Financial Econometric Analysis at Ultra-High Frequency: Data Handling Concerns*. Computational Statistics & Data Analysis, 51, 2232–2245. [Author university repository](https://flore.unifi.it/handle/2158/210321). This research motivates careful financial tick cleaning; the repository verifies its abstract and bibliography, while its full PDF is access restricted. The engine does not claim an exact replication.
4. Pearson, R. K., Neuvo, Y., Astola, J., and Gabbouj, M. (2016). *Generalized Hampel Filters*. EURASIP Journal on Advances in Signal Processing, 2016:87. [Publisher-version PDF in the authors' university repository](https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf).
5. Cleveland, W. S. (1979). *Robust Locally Weighted Regression and Smoothing Scatterplots*. Journal of the American Statistical Association, 74, 829–836. [Original paper PDF hosted by the University of Washington](https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf).
6. Killick, R., Fearnhead, P., and Eckley, I. A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost*. Journal of the American Statistical Association, 107, 1590–1598. [Author-submitted paper](https://arxiv.org/abs/1101.1438).
