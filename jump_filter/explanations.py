"""Shared mathematical method cards for browser and notebook trade review."""

# SETUP LOGIC: Regular expressions format curated presentation strings, never trade values.
import re


# UI LOGIC: Narrow panes need explicit line breaks in the few equations that remain wide after clause splitting.
_NARROW_BREAKS = (
    (r"\mathcal D_{\rm hard}", (r"=\{i:\texttt{jf\_status}", r"\ \land")),
    (r"\tau(t)=", (r"\mathbf1",)),
    (r"F_i=(H_i", (r"\land\neg",)),
    (r"R_i^{\rm tr}=", (r"\land",)),
    (r"F_i^{(f)}=", (r">\max",)),
    (r"\min_{b,h}", (r"+\delta", r"+\lambda")),
    (r"\min_b", (r"+\lambda",)),
    (r"\sigma_i=\max\{1.4826022185", (r"\operatorname{median}_{j", r"|e_j-",)),
    (r"\operatorname{prox}_{", (r"=\begin{cases}",)),
    (r"p\le", (r"+\max",)),
    (r"q_D\le", (r"+\rho",)),
)


def math_display_blocks(formula):
    """Return native LaTeX displays while leaving the exact source formula intact."""
    # UI LOGIC: The curated separators delimit independent definitions; case environments use neither separator.
    parts = re.split(r"(\\quad|,\\ )", formula)
    clauses = []
    for index in range(0, len(parts), 2):
        clause = parts[index].strip()
        if index + 1 < len(parts) and parts[index + 1].startswith(","):
            clause += ","
        if clause:
            clauses.append(clause)
    # UI LOGIC: Wrapping inserts only alignment/spacing tokens, retaining every original mathematical operator.
    displayed = []
    for clause in clauses:
        markers = next((breaks for prefix, breaks in _NARROW_BREAKS if clause.startswith(prefix)), ())
        lines = [clause]
        for marker in markers:
            last = lines.pop()
            before, found, after = last.partition(marker)
            lines.extend([before, marker + after] if found else [last])
        displayed.append(r"\begin{aligned}&" + r"\\&\qquad ".join(lines) + r"\end{aligned}" if len(lines) > 1 else clause)
    return tuple(displayed)

# CONFIGURATION LOGIC: Display metadata mirrors implemented rules; scores are never probabilities.
COMMON_PARAMETERS = ("window", "min_neighbors", "abs_floor")
PARAMETER_HELP = {
    "window": 'Maximum number of neighboring timestamps used by centered methods; robust_trend applies this limit only to the residual-scale neighborhood while fitting its baseline across the whole segment; the causal method uses at most window past cohorts.',
    "min_neighbors": 'Minimum number of reference timestamps. With fewer references, the method abstains; the trade is not certified as normal.',
    "threshold": 'MAD-standardized deviation threshold q. Increasing it reduces flags; equality at the boundary is not flagged. IQR does not use this parameter; iqr_multiplier controls its sensitivity.',
    "abs_floor": 'Absolute deviation floor a, in exactly the same units as the input spread. If spreads are in bp, 1.0 means 1 bp.',
    "reversion_tolerance": 'Tolerance v for the difference between left and right reference centers. Increasing it makes a return to the previous level easier to confirm.',
    "iqr_multiplier": 'IQR multiplier k for Tukey fences. Increasing it widens the acceptable interval.',
    "multiscale_votes": 'Minimum number of confirming scales among the three time/count scales; 2 requires a majority and 3 is more conservative.',
    "trend_penalty": 'Dimensionless TV penalty λ. Increasing it favors fewer level changes and can also smooth away genuine short-lived market moves.',
    "huber_delta": 'Huber cutoff δ for standardized residuals. Decreasing it limits the influence of an individual cohort on the offline trend.',
    "max_iter": 'Maximum number of offline-trend solver iterations. A nonconverged solution retains diagnostics and abstains. Default: 2000.',
    "tolerance": 'Numerical convergence tolerance for the offline-trend solver; this is separate from the score threshold used to identify financial outliers.',
    "alpha": 'EWMA update rate α per timestamp cohort; it is not adjusted for elapsed clock time.',
    "persistence": 'Number of consecutive large cohort innovations with the same sign; used only by the online comparator.',
}

# CONFIGURATION LOGIC: Each tuple is a title, explanatory prose, and an independently rendered LaTeX formula.
COMMON_STEPS = (
    ('1 · Data and centered neighborhood',
     'Sort each CUSIP independently by actual time. Trades with the same timestamp form a cohort whose median spread is z_j; score each original trade y_i separately. The centered before/after neighborhood below is used by Hampel, IQR, local_linear, reversion, consensus, multiscale, and the local residual scale of robust_trend, excluding the entire target timestamp cohort. Exceptions: robust_trend includes the target in its whole-segment baseline while limiting its influence; causal_ewma references come from the past (see its formula for the current-cohort exception when confirming a new regime). A gap > max_gap starts an independent segment. Both horizon and window constrain the neighborhood; missing references on one side are not replaced with more references from the other side.',
     r"z_j=\operatorname{median}\{y_r:t_r=t_j\},\quad L_i=\text{nearest }\lfloor W/2\rfloor\text{ earlier cohorts},\quad R_i=\text{nearest }(W-\lfloor W/2\rfloor)\text{ later cohorts},\quad\mathcal N_i=\{j\in L_i\cup R_i:|t_j-t_i|\le H\}"),
    ('2 · Scale, strict boundary and meaning',
     'q = threshold and a = abs_floor. The MAD multiplier 1.4826 provides normal-consistent scaling; it does not assume that trade residuals are normally distributed. Even zero MAD has a floor of a/q. Most methods use B_i below; IQR defines its own fence. jf_score measures deviation in scale units, not outlier probability. The baseline is a diagnostic reference, not an observed mid price.',
     r"s(v;c)=\max\{1.4826\operatorname{median}|v-c|,a/q\},\quad B_i=\max(a,q\sigma_i),\quad \texttt{jf\_score}_i=|y_i-b_i|/\sigma_i,\quad |y_i-b_i|>B_i\text{ uses a strict inequality}"),
)

# CONFIGURATION LOGIC: A trading clock strips scheduled closures without inventing missing active-market observations.
CLOCK_STEPS = (
    ('Market clock and liquidity',
     "time_basis='trading' compresses overnight closures, weekends, and explicit holidays using the configured local trading sessions; 'wall' uses actual calendar time. Plots show actual UTC and break the reference path at session boundaries. Centered time weights, horizon, max_gap, and gap diagnostics use the selected clock: scheduled closures do not count as trading-clock inactivity, but economic repricing can still occur overnight. local_piecewise can abstain when a new opening level disagrees with the earlier side. Sessions use [open,close). Trades outside the configured sessions remain in the original records but do not receive trading-clock scores; review the calendar rather than automatically treating them as execution outliers. The calendar is an editable research assumption, not a complete US bond-market calendar. For holidays and early closes, upload an authoritative session-schedule CSV in the browser or pass session_schedule to the API with timezone-aware open/close timestamps.",
     r"\tau(t)=\int_{t_0}^{t}\mathbf1\{u\text{ is inside a configured open session}\}\,du,\quad d_{ij}=|\tau(t_i)-\tau(t_j)|\ \text{(trading)},\quad d_{ij}=|t_i-t_j|\ \text{(wall)}"),
    ('Local support and volatility',
     'Using the same time basis, show the actual selected reference count, span, cohort density per hour, gap from the preceding cohort, and method-specific local volatility scale jf_scale. Also show actual UTC gaps in jf_wall_gap_minutes and cross-session transitions in jf_session_boundary. Density uses N rather than N−1 as its numerator; fewer than two references or span = 0 produce a missing value, not an artificial infinite density. When liquidity increases, the same count window covers a shorter time interval; when volatility increases, local MAD/IQR widens the band. Inspect these diagnostics together to distinguish insufficient support, changes in market direction, and individual trade deviations. They are diagnostics, not transaction-cost models or liquidity-risk probabilities.',
     r"N_i=|\mathcal N_i|,\quad \text{reference span}_i=\max_{j\in\mathcal N_i}\tau(t_j)-\min_{j\in\mathcal N_i}\tau(t_j),\quad \text{density}_i=N_i/(\text{span}_i/1\text{ hour}),\quad \text{local volatility}_i=\sigma_i"),
)

# CONFIGURATION LOGIC: Method cards use actual implementation formulas rather than generic algorithm names.
METHOD_EXPLANATIONS = {
    "hampel": {
        "name": 'Local Hampel · Median and MAD',
        "mode": 'Offline / centered · Uses earlier and later trades',
        "summary": 'Measure each trade against a robust center from earlier and later trades in the same bond. This is useful for isolated deviations resembling markups or markdowns.',
        "parameters": COMMON_PARAMETERS + ("threshold",),
        "steps": (("3 · Median reference and MAD decision",
                   'The baseline is the median of neighboring cohort medians; calculate MAD around that median. The target cohort contributes to neither the center nor the scale estimate. With sufficient support, an individual extreme print usually cannot move the center substantially.',
                   r"b_i=\operatorname{median}_{j\in\mathcal N_i}z_j,\quad\sigma_i=s((z_j)_{j\in\mathcal N_i};b_i),\quad F_i=\mathbf1\{|y_i-b_i|>\max(a,q\sigma_i)\}"),),
        "example": 'Reference spreads = [100,100,100,100,100,100] bp; target = 130 bp, q = 4.5, a = 1 bp: b = 100, MAD = 0, σ = 1/4.5 ≈ 0.222222 bp, B = 1 bp, score = 135; flag the outlier. A target of 101 bp is exactly on the boundary and is not flagged.',
        "tradeoffs": 'Simple and easy to audit. Genuine level changes can trigger false positives; consecutive outliers that form a majority of the neighborhood contaminate both the median and MAD. For a smooth trend, consider local_linear.',
        "reference": ("Pearson et al. (2016), Generalized Hampel Filters", "https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf"),
    },
    "rolling_iqr": {
        "name": 'Rolling IQR fences · Local quantile boundaries',
        "mode": 'Offline / centered · Uses earlier and later trades',
        "summary": 'Construct outer Tukey fences from the local interquartile range. This is an alternative robust scale to MAD and requires neither symmetric nor normal residuals.',
        "parameters": COMMON_PARAMETERS + ("iqr_multiplier",),
        "steps": (("3 · Quantiles and the actual outlier fence",
                   "Q1 and Q3 use NumPy's linear-interpolation empirical quantiles. Set the center to their midpoint so the symmetric plotted band matches Tukey fences exactly; abs_floor can only widen the band.",
                   r"Q_1=Q_{0.25}(z_{\mathcal N_i}),\ Q_3=Q_{0.75}(z_{\mathcal N_i}),\ I=Q_3-Q_1,\ b_i=(Q_1+Q_3)/2,\ B_i=\max\{a,(k+1/2)I\},\ F_i=\mathbf1\{y_i<b_i-B_i\ \lor\ y_i>b_i+B_i\}"),
                  ("4 · Score versus decision",
                   "When the floor is inactive, the boundary is exactly [Q1 − k IQR, Q3 + k IQR]. The output scale converts IQR to a normal-consistent scale for score comparison; the standardized fence multiplier is (k+0.5) × 1.3489795. The global threshold has no effect on this method's boundary, scale, or score.",
                   r"[b_i-B_i,b_i+B_i]=[Q_1-kI,Q_3+kI]\ \text{if }(k+1/2)I\ge a,\quad c_Q=1.3489795003921634,\quad\sigma_i=\max\{I/c_Q,a/[(k+1/2)c_Q]\},\quad B_i=(k+1/2)c_Q\sigma_i")),
        "example": 'References = [99,100,100,101,101,102] bp: Q1 = 100, Q3 = 101, IQR = 1. With k = 3 and a = 1, b = 100.5, B = 3.5, and the acceptable interval is [97,104] bp. Flag 105 bp; do not flag 104 bp. σ ≈ 0.741301 bp.',
        "tradeoffs": 'The deviation boundary has a direct quantile-fence interpretation. Small samples, quantized spreads, or outlier clusters can make quantiles unstable. There is no additional guard around genuine market jumps, and k is not an outlier probability.',
        "reference": ("NIST Engineering Statistics Handbook, Box Plot / Tukey fences", "https://www.itl.nist.gov/div898/handbook/eda/section3/boxplot.htm"),
    },
    "local_linear": {
        "name": 'Robust local linear · Local time trend',
        "mode": 'Offline / centered · Uses earlier and later trades',
        "summary": 'Fit a local trend, then measure the target print against it. Ordinary spread drift should not cause rejection solely because it differs from the window median.',
        "parameters": COMMON_PARAMETERS + ("threshold",),
        "steps": (("3 · Actual elapsed-time design and distance weights",
                   'Normalize time coordinates by the maximum actual time distance from the target, making the target prediction the intercept β0. Initialize β = [neighbor median, 0]. Tricube distance weights have a 0.05 floor to preserve numerical support from the farthest reference.',
                   r"d_i=\max\{1\text{ ns},\max_{j\in\mathcal N_i}|t_j-t_i|\},\ u_j=(t_j-t_i)/d_i,\ X_j=(1,u_j),\ v_j=\max\{[1-\min(|u_j|,0.999)^3]^3,0.05\}"),
                  ("4 · Six Huber IRLS rounds and residual decision",
                   'At each round, re-estimate residual MAD and solve weighted least squares. Fixing the iteration count at six is an engineering choice in this implementation; it draws on robust local regression without claiming an exact LOWESS reproduction. The final σ is the median-centered MAD of the final fitted residuals.',
                   r"r_j=z_j-X_j\beta,\ s=s(r;\operatorname{median}r),\ h_j=\min\{1,1.345s/\max(|r_j|,10^{-12})\},\ \beta\leftarrow\arg\min_\beta\sum_jv_jh_j(z_j-X_j\beta)^2;\quad b_i=\beta_0,\ \sigma_i=s(r^{\rm final};\operatorname{median}r^{\rm final}),\ F_i=\mathbf1\{|y_i-b_i|>B_i\}")),
        "example": 'Neighbor times = [0,1,2,4,5,6], spreads = [100,101,102,104,105,106] bp; target time = 3, spread = 130. The local linear prediction is b ≈ 103 bp and the final residuals are ≈ 0. With q = 4.5 and a = 1, B = 1 bp. The target residual is ≈ 27 bp and score ≈ 121.5; flag it. A normal 103 bp print is not flagged.',
        "tradeoffs": 'Useful for smooth drift and irregular trade intervals. Genuine abrupt jumps, curved trends, and contaminated short windows affect the result. Side extrapolations carry no market-structure guarantee.',
        "reference": ("Cleveland (1979), Robust Locally Weighted Regression", "https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf"),
    },
    "local_piecewise": {
        "name": 'Two-sided local piecewise trend · Independent side trends',
        "mode": 'Offline / independent side fits · Checks trend turning points',
        "summary": 'Fit the earlier and later trends separately and project both to the target time. Independent side trends can explain a normal rise-then-fall turning point, avoiding the mistake of treating the peak as an outlier relative to a single straight line.',
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Independent left/right robust linear predictions",
                   'Each side requires at least max(2,floor(min_neighbors/2)) cohorts. Independently apply local_linear distance weights and six Huber IRLS rounds to predict pL and pR at the target time. Left and right slopes may differ; the target cohort is excluded from both side fits.',
                   r"p_L=\widehat z_L(\tau(t_i)),\quad p_R=\widehat z_R(\tau(t_i)),\quad b_i=(p_L+p_R)/2,\quad\text{left slope and right slope need not agree}"),
                  ("4 · Agreement, local uncertainty and abstention",
                   'Take the maximum of the detrended residual MAD scales from the two sides, with a floor of a/q. Do not use the residual scale of one line spanning both sides, which would treat a normal turning point as noise. Evaluate the target around the common reference only if the side predictions agree within reversion_tolerance. On disagreement, still report the average baseline and score, but return ambiguous_transition, a level-change candidate, and weight 0: the trade is neither certified clean nor directly classified as an execution outlier. Missing support on either side also causes abstention.',
                   r"\sigma_i=\max(s_L,s_R,a/q),\quad\Delta_i=|p_R-p_L|,\quad A_i=\{\Delta_i\le\max(a,v\sigma_i)\},\quad F_i=A_i\land\{|y_i-b_i|>\max(a,q\sigma_i)\},\quad\neg A_i\Longrightarrow\texttt{ambiguous\_transition},\quad w_i=0,\ \texttt{jf\_fit\_eligible}=\texttt{False}")),
        "example": 'time = [0,1,2,3,4,5,6], spread = [100,101,102,103,102,101,100] bp. At target time = 3, the left trend extrapolates to approximately 103 and the right trend projects backward to approximately 103, so the normal 103 peak is not flagged. If the target print at that time is 130, both sides still predict 103, allowing the approximately 27 bp deviation to be flagged.',
        "tradeoffs": 'A continuous slope change is easier to confirm from both sides than an abrupt level jump. Irregular trades, contaminated short sides, and strong curvature can still make extrapolation unreliable. An unresolved transition with disagreeing side predictions reduces fitting coverage; include this abstention in the statistics.',
        "reference": ("Cleveland (1979), robust local regression (component inspiration)", "https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf"),
    },
    "jump_reversion": {
        "name": 'Jump and reversion · Return-to-level confirmation',
        "mode": 'Offline / two-sided confirmation · Requires support on both sides',
        "summary": 'Treat a middle deviation as a suspected execution effect only when the earlier and later centers return to the same level; protect a persistent new market level.',
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Independent before/after support",
                   'Each side requires at least M cohorts. Take their medians separately and estimate scale only from dispersion within each side, avoiding the mistake of treating a genuine difference between the side levels as random noise.',
                   r"M=\max\{2,\lfloor\texttt{min\_neighbors}/2\rfloor\},\ |L_i|,|R_i|\ge M,\ m_L=\operatorname{median}z_L,\ m_R=\operatorname{median}z_R,\ D=(z_L-m_L)\cup(z_R-m_R),\ b_i=(m_L+m_R)/2,\ \sigma_i=s(D;0)"),
                  ("4 · Reversion gate and persistent-shift guard",
                   'Confirm reversion only if the centers differ by no more than max(a,vσ); the target itself must also lie outside the deviation band. If the side centers differ by more than max(a,qσ), report a level-change candidate and protect the trade. Reversion need not occur at the immediately next trade.',
                   r"\Delta_i=|m_R-m_L|,\ A_i=\{\Delta_i\le\max(a,v\sigma_i)\},\ S_i=\{\Delta_i>\max(a,q\sigma_i)\},\ F_i=A_i\land\{|y_i-b_i|>B_i\}\land\neg S_i")),
        "example": 'Left [100,100,100], right [100,100,100], target 130 bp: b = 100, Δ = 0. With q = 4.5, a = 1, v = 2, reversion is confirmed and the deviation 30 > 1 is flagged. Change the right side to [130,130,130]: Δ = 30 > 1, so the persistent level change is protected.',
        "tradeoffs": 'Useful for protecting genuine level shifts. At the beginning or end of a series, missing support on one side causes abstention. A brief genuine market move can resemble an outlier, and a long burst can contaminate both sides. This is an engineering rule based on three columns, not a formal changepoint test.',
        "reference": ("Killick et al. (2012), changepoint framework (context; this rule is not PELT)", "https://arxiv.org/abs/1101.1438"),
    },
    "multiscale": {
        "name": 'Multiscale confirmation · Agreement across neighborhood sizes',
        "mode": 'Offline / centered · Uses earlier and later trades',
        "summary": 'Repeat the assessment over short, medium, and long neighborhoods to reduce deviations that appear exceptional only at one window size, aiming for greater robustness to clusters and window selection.',
        "parameters": COMMON_PARAMETERS + ("threshold", "multiscale_votes"),
        "steps": (("3 · Three count/time neighborhoods and Hampel votes",
                   'Use 1, 2, and 4 times the base window W and horizon H to form three scales. Each scale excludes the target timestamp cohort and, with at least min_neighbors references, computes its own median, MAD, and strict deviation boundary. The final decision requires multiscale_votes confirming scales. The minimum support count does not automatically increase with the window size.',
                   r"f\in\{1,2,4\},\quad(W_f,H_f)=(fW,fH),\quad b_i^{(f)}=\operatorname{median}z_{\mathcal N_i^{(f)}},\quad\sigma_i^{(f)}=s(z_{\mathcal N_i^{(f)}};b_i^{(f)}),\quad F_i^{(f)}=\mathbf1\{|y_i-b_i^{(f)}|>\max(a,q\sigma_i^{(f)})\},\quad F_i=\mathbf1\{\sum_fF_i^{(f)}\ge V\}"),
                  ("4 · Plot the evidence that supports the decision",
                   'When flagged, display the baseline, scale, and band from the confirming scale with the largest standardized score. When unflagged, use the largest score among available nonconfirming scales (or the smallest score among all scales if none is nonconfirming), keeping the band consistent with the final decision. Fewer than V supported scales cause abstention. There is no additional reversal/shift guard here: genuine jumps can also be flagged across scales.',
                   r"f^*=\arg\max_{f\in\mathcal A_i}|y_i-b_i^{(f)}|/\sigma_i^{(f)},\quad\mathcal A_i=\begin{cases}\{f:F_i^{(f)}=1\},&F_i=1\\\{f:F_i^{(f)}=0\text{ and supported}\},&F_i=0\end{cases},\quad(b_i,\sigma_i,B_i)=(b_i^{(f^*)},\sigma_i^{(f^*)},B_i^{(f^*)})")),
        "example": 'Three supported scales have center = [100,101,109], scale = [1,1,1] bp; target = 110, q = 4.5, a = 1: scores = [10,9,1], votes = [True,True,False]. With V = 2, flag the target and display the first scale with baseline = 100, B = 4.5. With V = 3, do not flag it.',
        "tradeoffs": 'The scales share data, so votes are neither independent evidence nor statistical significance. Requiring more votes lowers sensitivity; sparse data can cause abstention through insufficient support. There is no mandatory repricing guard; compare with consensus / robust_trend.',
        "reference": ("Pearson et al. (2016), generalized robust local filters (inspiration)", "https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf"),
    },
    "robust_trend": {
        "name": 'Offline robust TV trend · Robust segmented level',
        "mode": 'Offline / whole segment · Joint level and sparse-deviation estimation',
        "summary": 'Use earlier and later information across the entire trade segment, assigning persistent level changes to the trend b and sparse large deviations to the execution component h. This is another reference for historical filtering before fitting and requires calibration on real data.',
        "parameters": COMMON_PARAMETERS + ("threshold", "trend_penalty", "huber_delta", "max_iter", "tolerance"),
        "steps": (("3 · Normalize cohort spreads and separate level from sparse shocks",
                   'μ is the whole-segment cohort median. s0 uses the median-centered MAD of first differences / √2, with a floor of a/q. After normalization, both λ and δ are dimensionless. The objective jointly estimates the level b and sparse deviations h. The target cohort participates in the robust fit; this method is not leave-one-out.',
                   r"\mu=\operatorname{median}z,\quad s_0=\max\{1.4826022185\operatorname{MAD}(\Delta z)/\sqrt2,a/q\},\quad x_j=(z_j-\mu)/s_0,\quad\min_{b,h}\frac12\|x-b-h\|_2^2+\delta\|h\|_1+\lambda\sum_{j=1}^{n-1}|b_{j+1}-b_j|"),
                  ("4 · Equivalent Huber-TV objective and sparse component",
                   'For fixed b, h is a soft threshold, making the objective equivalent to Huber loss plus first-order total variation. TV penalizes the level difference between neighboring cohorts; each adjacent edge has weight 1 rather than scaling with actual elapsed time, so gap segmentation remains important. This fits a piecewise-constant fused level, not second-order linear trend filtering.',
                   r"h_j=\operatorname{soft}(x_j-b_j,\delta),\quad \operatorname{soft}(r,\delta)=\operatorname{sign}(r)\max(|r|-\delta,0),\quad\min_b\sum_j\rho_\delta(x_j-b_j)+\lambda\|Db\|_1,\quad\rho_\delta(r)=\begin{cases}r^2/2,&|r|\le\delta\\\delta|r|-\delta^2/2,&|r|>\delta\end{cases}"),
                  ("5 · Back to raw units, local scale and per-trade flags",
                   'baseline = μ + s0 b. Each raw trade uses the baseline of its cohort, while σ comes from the surrounding leave-cohort-out fitted-residual MAD (this method uses 1.4826022185; the earlier 1.4826 is an approximation). This is not a confidence interval for latent mid. ADMM convergence is controlled by max_iter / tolerance; nonconverged segments are not forced into clean/outlier labels.',
                   r"\widehat m_j=\mu+s_0b_j,\quad e_j=z_j-\widehat m_j,\quad b_i=\widehat m_{c(i)},\quad\sigma_i=\max\{1.4826022185\operatorname{median}_{j\in\mathcal N_i}|e_j-\operatorname{median}_{k\in\mathcal N_i}e_k|,a/q\},\quad F_i=\mathbf1\{|y_i-b_i|>\max(a,q\sigma_i)\}"),
                  ("6 · Solver / ADMM update and convergence",
                   'Let r = x − b and d = Db, with constraints b + r = x and Db − d = 0. ADMM places r and d in the same separable proximal block, preserving a two-block structure; u and v are scaled duals. Initialize the baseline with a 3-cohort median using edge replication and initial ρ = 4. Within the first 100 iterations, only at rounds 25, 50, 75, and 100, adjust ρ within [0.125,64] using the primal/dual ratio and rescale the duals. The Huber proximal formula below uses η = x − b − u; soft is defined above.',
                   r"b^+=(I+D^\top D)^{-1}[x-r-u+D^\top(d-v)],\quad r^+=\operatorname{prox}_{\rho_\delta/\rho}(x-b^+-u),\quad d^+=\operatorname{soft}(Db^++v,\lambda/\rho),\quad u^+=u+b^++r^+-x,\quad v^+=v+Db^+-d^+;\quad \operatorname{prox}_{\rho_\delta/\rho}(\eta)=\begin{cases}\eta/(1+1/\rho),&|\eta|\le\delta(1+1/\rho)\\\eta-(\delta/\rho)\operatorname{sign}(\eta),&\text{otherwise}\end{cases}"),
                  ('7 · Explicit stopping criterion',
                   'Check convergence using both primal and dual norms in normalized units, each with absolute plus relative tolerance; ε = tolerance. If these conditions fail when max_iter is exhausted, return solver_not_converged, zero fitting weight, and abstention. Solver accuracy and outlier sensitivity are separate settings.',
                   r"p=\|[b+r-x;Db-d]\|_2,\quad q_D=\rho\|r-r^{\rm old}-D^\top(d-d^{\rm old})\|_2,\quad p\le\epsilon[\sqrt{2n-1}+\max(\|[b;Db]\|_2,\|[r;d]\|_2,\|x\|_2)],\quad q_D\le\epsilon[\sqrt n+\rho\|u+D^\top v\|_2]")),
        "example": 'Whole segment [100,100,100,130,100,100,100] bp, q = 4.5, a = 1, δ = 2.5, λ = 8: s0 = 2/9 bp. The convex optimum has a common baseline ≈ 100.092593 bp (numerical tolerances may cause small differences). The 130 bp residual is ≈ 29.907407 bp, local scale = 2/9, B = 1; flag it. The six 100 bp prints are not flagged. This example requires window ≥ 6, min_neighbors = 6, and a horizon covering the whole segment.',
        "tradeoffs": 'A larger λ can oversmooth genuine jumps; a smaller λ can let the trend follow bad prints. For a large temporary internal block of height A and length L, fitting the block costs approximately 2λ|A| for its two TV edges; maintaining the baseline costs approximately L(δ|A|−δ²/2) under saturated Huber loss. This is only an approximate comparison in normalized units: long bursts are more likely to be treated as genuine levels, while brief genuine moves can be treated as outliers. Dense execution bias with a consistent direction is not identifiable from only these three columns.',
        "reference": ("Boyd et al. (2011), Distributed Optimization and Statistical Learning via ADMM", "https://web.stanford.edu/~boyd/papers/admm_distr_stats.html"),
    },
    "consensus": {
        "name": 'Conservative consensus · Robust evidence and reversion',
        "mode": 'Offline / two-sided confirmation · Default research starting point',
        "summary": 'Flag a trade only when Hampel or the local trend finds a large deviation and either raw levels or trend projections confirm reversion. This can protect persistent repricing while accommodating smooth drift.',
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Raw and trend-adjusted reversion",
                   'Raw reversion is jump_reversion. Trend reversion independently fits robust linear trends to the left and right and projects them to the target time. Similar side predictions form an average trend bridge. Take the largest residual scale across the left, right, and overall linear fits to avoid spuriously small scales from short side fits.',
                   r"p_L=\widehat z_L(t_i),\ p_R=\widehat z_R(t_i),\ b_i^{\rm tr}=(p_L+p_R)/2,\ \tau_i=\max(s_L,s_R,s_{\rm all}),\ \Delta_i^{\rm tr}=|p_R-p_L|,\ R_i^{\rm tr}=\{\Delta_i^{\rm tr}\le\max(a,v\tau_i)\}\land\{|y_i-b_i^{\rm tr}|>\max(a,q\tau_i)\}"),
                  ("4 · Exact Boolean rule and plotted baseline",
                   'H and L are the Hampel and local_linear candidates. This is a union with reversion confirmation, not a majority vote. Protect a shift only if both raw and trend views identify a large side-to-side change. When raw reversion holds, the raw bridge takes precedence; otherwise, a trend-only flag uses the trend bridge, keeping the plotted band consistent with the flag. Insufficient support on either side causes abstention.',
                   r"S_i^{\rm raw}=\{\Delta_i^{\rm raw}>\max(a,q\sigma_i^{\rm side})\},\quad S_i^{\rm tr}=\{\Delta_i^{\rm tr}>\max(a,q\tau_i)\},\quad F_i=(H_i\lor L_i)\land(R_i^{\rm raw}\lor R_i^{\rm tr})\land\neg(S_i^{\rm raw}\land S_i^{\rm tr})")),
        "example": 'time = [0,1,2,4,5,6], spread = [100,101,102,104,105,106] bp; target time = 3, spread = 130. Raw medians are 101 and 105; both trend projections are approximately 103. With q = 4.5 and a = 1, trend reversion is confirmed and L = True, so flag the target. A 103 bp target is not flagged.',
        "tradeoffs": 'The default prioritizes avoiding rejection of persistent changes, but can miss very long bursts. Components are correlated, so agreement does not provide independent confidence. Curvature, contaminated side extrapolations, and series edges can reduce support or accuracy.',
        "reference": ("Implementation and research discussion in docs/RESEARCH.md", "https://github.com/ezyhdxm/jump_filter/blob/main/docs/RESEARCH.md"),
    },
    "causal_ewma": {
        "name": 'Causal robust EWMA · Online comparator',
        "mode": 'Optional online comparator · Uses past information only',
        "summary": 'In an online setting, predict new trades from a historical robust level and confirm a new regime through consecutive same-direction deviations. When historical fitting does not require causality, start with the offline methods above.',
        "parameters": COMMON_PARAMETERS + ("threshold", "alpha", "persistence"),
        "steps": (("3 · Historical innovation and clipped EWMA update",
                   'MAD from past distinct cohorts provides σ; the center comes from the existing EWMA state. window/horizon constrain the history, and alpha is per cohort rather than per clock time. In the ordinary case, score each original trade before updating the state.',
                   r"e_c=z_c-m_{c-1},\quad B_c=\max(a,q\sigma_c),\quad F_i=\mathbf1\{|y_i-m_{c-1}|>B_c\},\quad m_c=m_{c-1}+\alpha\operatorname{clip}(e_c,-B_c,B_c)"),
                  ("4 · Provisional jumps and persistent regime confirmation",
                   'Accumulate the sign of cohort innovations outside the band. After persistence consecutive same-direction innovations, reset the center to the current cohort median and clear the supporting history. Score trades in the confirming cohort against its new median; earlier provisional flags are not revised retroactively. Insufficient subsequent support causes warm-up.',
                   r"\text{same-sign large }e_c\text{ for }P\text{ cohorts}\ \Longrightarrow\ m_c=z_c,\ \text{history}\leftarrow\varnothing,\quad\text{otherwise large trade innovations are provisional jumps}")),
        "example": 'Existing center = 100 bp, past σ = 1 bp, q = 4.5, a = 1, alpha = 0.2, new cohort median = 130: cutoff = 4.5 and deviation = 30, so flag it provisionally. Update the center to 100 + 0.2×4.5 = 100.9 bp. Confirmation at the third same-direction large innovation (persistence = 3) resets the center; the earlier two flags remain.',
        "tradeoffs": 'An online baseline for checking future leakage. The first few trades of genuine repricing may be provisionally rejected; historical markups can also contaminate the scale. Offline methods can use subsequent evidence when filtering historical data.',
        "reference": ("Implementation and causal-prefix discussion in docs/RESEARCH.md", "https://github.com/ezyhdxm/jump_filter/blob/main/docs/RESEARCH.md"),
    },
}

# CONFIGURATION LOGIC: The shared three-column fitting policy is explicit in both dashboards.
FITTING_STEPS = (
    ('Use in a downstream fit',
     "Strict selection uses jf_fit_eligible, equivalent to status = ok with no outlier flag on the original spread. select_fit_data(result, policy='hard') returns those records and adds weight 1. Keep evaluated, unflagged shifts; do not replace the original trade spread with the baseline as if it were new evidence. invalid, insufficient_history, ambiguous_transition, solver_not_converged, and provisional_jump are not automatically treated as clean.",
     r"\mathcal D_{\rm hard}=\{i:\texttt{jf\_fit\_eligible}_i\}=\{i:\texttt{jf\_status}_i=\texttt{ok}\ \land\ \neg\texttt{jf\_is\_outlier}_i\}"),
    ('Optional influence weights and coverage',
     "select_fit_data(result, policy='soft') defaults to evaluated ok/outlier records and applies jf_fit_weight = jf_weight: unflagged weight 1, flagged weight cutoff / |residual| capped at 1. These weights suggest influence, not probability or inverse variance. Provisional records are excluded by default. Coverage uses all supplied rows as the denominator, so an apparently clean subset with almost no support cannot conceal poor coverage.",
     r"w_i=\begin{cases}1,&\text{evaluated and unflagged}\\\min(1,B_i/\max(|y_i-b_i|,10^{-12})),&\text{flagged}\end{cases},\quad\text{coverage}=N_{\rm evaluated}/N_{\rm supplied},\quad\text{hard retention}=|\mathcal D_{\rm hard}|/N_{\rm supplied}"),
)

# CONFIGURATION LOGIC: Existing public helper remains a lightweight method summary lookup.
METHOD_HELP = {key: card["summary"] for key, card in METHOD_EXPLANATIONS.items()}
