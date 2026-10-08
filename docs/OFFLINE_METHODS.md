# Additional retrospective methods: mathematics and limits

These methods select historical observations for a subsequent fit. Future
observations within the authorized history are usable evidence. They mark
statistical deviations, not retail identities, commission amounts, distressed
trades, or an identified executable mid price.

## Common cohort and segment contract

For each CUSIP, the engine sorts valid timestamps and reduces simultaneous
trades to one median per distinct timestamp. Write that series as
\((t_i,y_i)_{i=1}^n\). Timestamp cohorts, rather than row counts, receive equal
influence. Original trades are scored individually against their cohort's
baseline. The engine splits the series at gaps exceeding `max_gap`.

For local references, the whole target timestamp is excluded. At most
`floor(window/2)` earlier and `window-floor(window/2)` later cohorts are used,
and each must lie within the elapsed `horizon`. Insufficient distinct-time
support causes abstention. This prevents repeated reports at one timestamp
from masquerading as independent evidence. A cohort median can still be
contaminated if most trades at that timestamp are contaminated.

With `time_basis="trading"`, elapsed distances and gap splitting use cumulative
declared open-session time. Scheduled closures consume zero clock time, while
no-trade gaps during an open session remain real elapsed gaps. The default
weekday-hours template is a configuration, not a verified bond holiday
calendar; explicit holidays or an authoritative aware session table are needed
to describe actual holidays and early closes.

## Centered rolling IQR

Let \(Q_{1,i}\) and \(Q_{3,i}\) be the 25th and 75th sample quantiles of the
held-out local reference set, using NumPy's explicit `method="linear"` sample
quantiles. Let \(I_i=Q_{3,i}-Q_{1,i}\). Tukey fences with multiplier \(k\) are

\[
L_i=Q_{1,i}-kI_i,\qquad U_i=Q_{3,i}+kI_i.
\]

Their symmetric representation uses the **midhinge**, rather than the median:

\[
c_i=\frac{Q_{1,i}+Q_{3,i}}2,\quad
C_i=\max\{a,(k+\tfrac12)I_i\},\quad
\text{flag}(y)=\mathbf1\{|y-c_i|>C_i\},
\]

where \(a=\texttt{abs_floor}\) has the same units as `spread`. The floor expands
both fences symmetrically around the midhinge. Equality at a fence is retained.
The conventional multiplier 3 specifies the outer fences; it is not a
probability or a contamination rate. The [NIST statistical handbook](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)
describes inner and outer IQR fences.

For consistent diagnostic units, the engine reports

\[
s_i=\max\left\{\frac{I_i}{1.3489795003921634},
\frac{a}{(k+\tfrac12)1.3489795003921634}\right\},\qquad
q(y)=\frac{|y-c_i|}{s_i}.
\]

Thus the dimensionless score threshold is
\((k+1/2)1.3489795003921634\), while `jf_threshold` stores the actual raw-spread
cutoff \(C_i\). The general `threshold` parameter does not affect this method;
`iqr_multiplier` controls its sensitivity. The normal-distribution calibration
is descriptive, not a claim that transaction spreads are Gaussian.

**Example.** References `[98,99,100,101,102]` give `Q1=99`, `Q3=101`,
`IQR=2`, `midhinge=100`. With `k=3` and `abs_floor=1`, the acceptable band is
`[93,107]`. A spread of `110` is flagged; `107` is retained.

**When useful.** IQR supplies a simple complementary measure of distribution
width and does not require a fitted linear path. Its midhinge is affected by
asymmetry, and a real trend within a broad window widens its band. A long
contaminated burst can shift a quartile or inflate IQR, hiding itself. It does
not independently protect persistent regime changes.

## Offline Huber total variation with an explicit execution component

This method jointly estimates a piecewise-constant path across one complete
segment. Its temporal regularizer penalizes **level differences**, not slopes
or second derivatives. It is therefore robust fused-level estimation rather
than a spline or piecewise-linear trend filter.

First establish a translation and a spread scale:

\[
\mu=\operatorname{median}(y),\quad d_i=y_{i+1}-y_i,
\]
\[
s_0=\max\left\{a/h,
\frac{1.482602218505602}{\sqrt2}
\operatorname{median}|d-\operatorname{median}(d)|\right\},\quad
z_i=(y_i-\mu)/s_0,
\]

where \(h=\texttt{threshold}\). For one cohort, the differenced MAD is zero.
Dividing by \(\sqrt2\) is the Gaussian calibration for the difference of two
independent equal-variance level noises. Dependence, a changing drift, jumps,
or a heavily contaminated differenced series can invalidate that scale
interpretation. The floor keeps flat data numerically usable.

The implemented dimensionless convex objective is

\[
\min_{b,e}\quad
\frac12\sum_i(z_i-b_i-e_i)^2
+\delta\sum_i|e_i|
+\lambda\sum_{i=1}^{n-1}|b_{i+1}-b_i|,
\]

where \(\delta=\texttt{huber_delta}>0\) and
\(\lambda=\texttt{trend_penalty}>0\). The variable \(e_i\) is a sparse
execution-deviation component. It is not an estimated commission. Larger
\(\lambda\) favors fewer level changes; a smaller value follows observations
more closely and can absorb anomalous bursts into the path. Smaller
\(\delta\) reduces the influence of large residuals but also allows more
execution components.

Eliminating \(e\) gives the equivalent Huber objective:

\[
\min_b\sum_i\rho_\delta(z_i-b_i)+\lambda\|Db\|_1,
\quad (Db)_i=b_{i+1}-b_i,
\]

\[
\rho_\delta(r)=
\begin{cases}r^2/2,&|r|\le\delta,\\
\delta|r|-\delta^2/2,&|r|>\delta.
\end{cases}
\]

At an optimum, \(e_i=\operatorname{soft}(z_i-b_i,\delta)\). No fixed number or
fraction of trades is forced to be abnormal. This bounded-influence loss
uses the robust estimation idea in [Huber (1964)](https://doi.org/10.1214/aoms/1177703732).
The adjacent-difference penalty follows the fused-lasso/total-variation
construction in [Tibshirani et al. (2005)](https://doi.org/10.1111/j.1467-9868.2005.00490.x).
The explicit sparse-variable expression and its normalization are
implementation-specific choices here.

The resulting spread baseline is \(m_i=\mu+s_0b_i\). Unlike the local methods,
it is a **joint estimator that includes its own target cohort with bounded
influence**. It is not leave-one-out. For example, nine flat cohorts with one
spread of `130` and eight of `100`, `abs_floor=1`, `threshold=4.5`,
`huber_delta=2.5`, `trend_penalty=8` yield an approximately constant optimum
`100.0694444444`, not exactly `100`. The sparse execution component at the
exception is approximately `29.375` spread units.

The total-variation penalty gives each pair of adjacent cohorts one penalty;
it does not multiply or divide by elapsed time. Distinct-time cohort ordering
and market-gap splitting define its graph. Uneven observation density thus
changes how many data-fidelity terms support a plateau. `horizon` and `window`
bound only the final local scatter calculation for this method, not the
segment-wide optimization.

For final row scoring, compute cohort residuals \(r_i=y_i-m_i\), and use their
held-out local reference set \(N_i\):

\[
s_i=\max\{a/h,
1.482602218505602\operatorname{median}_{j\in N_i}
|r_j-\operatorname{median}_{k\in N_i}r_k|\},
\]
\[
C_i=\max(a,hs_i),\qquad
\text{flag}(y)=\mathbf1\{|y-m_i|>C_i\}.
\]

The target is excluded from this scatter calculation. The fitted residuals
themselves came from a joint fit, however, so this does not remove joint-fit
dependence. The nonzero sparse component is a useful diagnostic; its presence
alone does not determine the final flag.

**When useful.** A persistent change supported by many timestamps can pay one
TV change penalty, while a short excursion would pay two. This distinguishes
many lasting repricings from isolated prints. There is no universal separation:
a sustained execution premium can look like a real regime, and a short real
market dislocation can look like a bad-trade burst. Excessive smoothing shrinks
small or short regimes, smooth slopes become staircases, and heavily
contaminated histories can bias the normalization and the baseline.

## Solver, convergence, and scaling audit

Let \(r=z-b\) and \(v=Db\). Two-block ADMM splits the baseline block from the
joint separable \((r,v)\) block. With scaled duals \((u,w)\), the updates are

\[
b^+=(I+D^TD)^{-1}\{z-r-u+D^T(v-w)\},
\]
\[
r^+=\operatorname{prox}_{\rho_\delta/\eta}(z-b^+-u),\qquad
v^+=\operatorname{soft}(Db^++w,\lambda/\eta),
\]
\[
u^+=u+b^++r^+-z,\qquad w^+=w+Db^+-v^+.
\]

The Huber proximal map is \(x/(1+1/\eta)\) when
\(|x|\le\delta(1+1/\eta)\); otherwise it is
\(x-(\delta/\eta)\operatorname{sign}(x)\). A NumPy Thomas factorization solves
the positive-definite tridiagonal matrix \(I+D^TD\) in linear storage. The
factorization is reused because a common penalty \(\eta\) cancels from the
baseline solve. Initialization uses a three-cohort median with replicated
endpoints; this affects only the starting point.

The penalty begins at \(\eta=4\). At iterations 25, 50, 75, and 100 only, a
10:1 primal/dual residual imbalance can double or halve it, constrained to
`[0.125,64]`; scaled duals are adjusted accordingly. It then stays fixed.
Stopping uses both primal and dual norms against absolute-plus-relative
tolerances in standardized units. These choices follow the ADMM treatment in
[Boyd et al. (2011)](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
and the related [total-variation solver example](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html).

Diagnostics include `converged`, iteration count, feasible original-objective
value, residual norms, stopping tolerances, final solver penalty, normalization
scale, held-out reference counts, and the sparse component in spread units.
An iteration limit must cause engine abstention, zero fitting weight, and an
explicit nonconvergence status. Finite-iteration tolerances do not certify an
exact optimum; long flat plateaus can take substantially more iterations than
short noisy segments.
The practical default iteration limit is 2000, with `tolerance=1e-4` used for
both the absolute and relative terms. These numerical controls govern solver
accuracy, rather than the economic outlier cutoff.

Multiplying every spread and `abs_floor` by the same positive constant leaves
the normalized optimization, scores, and flags unchanged and scales the
returned baseline and raw thresholds by that constant. Translation shifts the
baseline by the same amount. Tune parameters against labeled economic
scenarios, retained-fit stability, and downstream validation; do not interpret
the score as an anomaly probability. Offline filtering may use future history
for a historical fit, but downstream model evaluation must still respect its
train/validation boundaries.
