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
    "window": "居中方法最多使用的邻居 timestamp 数；robust_trend 只限制残差 scale 邻域，baseline 拟合整段；causal 使用最多 window 个过去 cohorts。",
    "min_neighbors": "最低参考 timestamp 数；达不到就 abstain，不能当成已验证的正常交易。",
    "threshold": "MAD 标准化偏离阈值 q；调大减少标记，边界相等不标记。IQR 法不用此参数，敏感度由 iqr_multiplier 控制。",
    "abs_floor": "绝对偏离下限 a，单位与输入 spread 完全一致；比如 spread 为 bp，1.0 就是 1 bp。",
    "reversion_tolerance": "左右参考中心允许的差异 v；调大更容易认为交易之后回到了原水平。",
    "iqr_multiplier": "Tukey fence 的 IQR 倍数 k；增大扩大可接受区间。",
    "multiscale_votes": "三个时间/数量尺度中至少多少个确认异常；2 是多数确认，3 更保守。",
    "trend_penalty": "无量纲 TV 惩罚 λ；调大倾向更少水平跳变，也可能吞掉真实短暂市场变化。",
    "huber_delta": "标准化残差的 Huber 截断 δ；调小限制单个 cohort 对离线趋势的影响。",
    "max_iter": "离线趋势求解迭代上限；未收敛时保留诊断并 abstain。默认 2000。",
    "tolerance": "离线趋势求解的数值收敛容差，与金融异常的 score threshold 不同。",
    "alpha": "每个 timestamp cohort 的 EWMA 更新速度 α，并非按钟表时间调整。",
    "persistence": "连续同方向的大 cohort innovation 数；只用于在线比较方法。",
}

# CONFIGURATION LOGIC: Each tuple is a title, explanatory prose, and an independently rendered LaTeX formula.
COMMON_STEPS = (
    ("1 · Data and centered neighborhood / 数据与居中邻域",
     "每个 CUSIP 独立按真实时间排序。同一 timestamp 的交易取 median 得到 cohort spread z_j，原始每笔 y_i 分别打分。以下前后居中邻域用于 Hampel、IQR、local_linear、reversion、consensus、多尺度与 robust_trend 的局部残差 scale；排除目标 timestamp 整组。例外：robust_trend 的全段 baseline 包含目标但限制其影响；causal_ewma 的参考来自过去（确认新 regime 的当前 cohort 特例见方法公式）。gap > max_gap 独立分段，horizon 与 window 同时限制邻域；缺少一侧不会从另一侧补足。",
     r"z_j=\operatorname{median}\{y_r:t_r=t_j\},\quad L_i=\text{nearest }\lfloor W/2\rfloor\text{ earlier cohorts},\quad R_i=\text{nearest }(W-\lfloor W/2\rfloor)\text{ later cohorts},\quad\mathcal N_i=\{j\in L_i\cup R_i:|t_j-t_i|\le H\}"),
    ("2 · Scale, strict boundary and meaning / 尺度、边界与含义",
     "q = threshold，a = abs_floor。MAD 的 1.4826 是正态一致性换算系数，不要求成交残差服从正态。零 MAD 仍有 a/q 下限。大多数方法使用下面的 B_i；IQR 方法单独定义 fence。jf_score 是偏离尺度的倍数，不是异常概率。baseline 是诊断参考，并非观测到的 mid。",
     r"s(v;c)=\max\{1.4826\operatorname{median}|v-c|,a/q\},\quad B_i=\max(a,q\sigma_i),\quad \texttt{jf\_score}_i=|y_i-b_i|/\sigma_i,\quad |y_i-b_i|>B_i\text{ uses a strict inequality}"),
)

# CONFIGURATION LOGIC: A trading clock strips scheduled closures without inventing missing active-market observations.
CLOCK_STEPS = (
    ("Market clock and liquidity / 交易时钟与流动性",
     "time_basis='trading' 用配置的本地交易 session 压缩夜间、周末和显式 holidays；'wall' 使用真实日历时间。图显示实际 UTC，并在 session boundary 断开参考路径。居中时间权重、horizon、max_gap 与 gap diagnostics 使用选定时钟：scheduled 休市不算交易时钟 inactivity，但不假定隔夜没有经济 repricing；新开盘水平与前一侧不一致时 local_piecewise 可 abstain。Session 使用 [open,close)；配置之外的交易保留原始记录但不参与 trading 评分，需核查 calendar，并不直接等于 execution outlier。Calendar 是可编辑研究假设，不是完整美国债券交易日历；early close、节日可在浏览器上传权威 session schedule CSV，或用 API session_schedule（时区感知 open/close）。",
     r"\tau(t)=\int_{t_0}^{t}\mathbf1\{u\text{ is inside a configured open session}\}\,du,\quad d_{ij}=|\tau(t_i)-\tau(t_j)|\ \text{(trading)},\quad d_{ij}=|t_i-t_j|\ \text{(wall)}"),
    ("Local support and volatility / 局部支持与波动",
     "以同一个 time basis 展示实际所选参考数量、跨度、每小时 cohort density、距前 cohort gap，以及本方法局部波动尺度 jf_scale。同时显示真实 UTC 的 jf_wall_gap_minutes 与跨 session 的 jf_session_boundary。Density 用 N 而非 N−1 作分子；少于两个参考或 span = 0 时为缺失，不伪造无限 density。流动性增加时，同样 count window 覆盖更短时间；波动升高时本地 MAD/IQR 提高 band。一起看这些诊断才能区分缺支持、市场走势改变与单笔偏离。它们是诊断，不是交易成本模型或流动性风险概率。",
     r"N_i=|\mathcal N_i|,\quad \text{reference span}_i=\max_{j\in\mathcal N_i}\tau(t_j)-\min_{j\in\mathcal N_i}\tau(t_j),\quad \text{density}_i=N_i/(\text{span}_i/1\text{ hour}),\quad \text{local volatility}_i=\sigma_i"),
)

# CONFIGURATION LOGIC: Method cards use actual implementation formulas rather than generic algorithm names.
METHOD_EXPLANATIONS = {
    "hampel": {
        "name": "Local Hampel · 局部中位数与 MAD",
        "mode": "Offline / centered · 使用前后交易",
        "summary": "用同一个 bond 前后的稳健中心衡量每笔交易是否偏得太远，适合抓孤立的 markup/down 式异常点。",
        "parameters": COMMON_PARAMETERS + ("threshold",),
        "steps": (("3 · Median reference and MAD decision",
                   "邻域 cohort medians 的 median 是 baseline，围绕该 median 计算 MAD。目标 cohort 不参与中心或尺度估计。窗口足够时，单个极端 print 通常不能大幅拉动中心。",
                   r"b_i=\operatorname{median}_{j\in\mathcal N_i}z_j,\quad\sigma_i=s((z_j)_{j\in\mathcal N_i};b_i),\quad F_i=\mathbf1\{|y_i-b_i|>\max(a,q\sigma_i)\}"),),
        "example": "参考 spreads = [100,100,100,100,100,100] bp，目标 = 130 bp，q = 4.5，a = 1 bp：b = 100，MAD = 0，σ = 1/4.5 ≈ 0.222222 bp，B = 1 bp，score = 135，标记 outlier。目标 = 101 bp 恰好在边界，不标记。",
        "tradeoffs": "简洁、容易审计。真实水平跳变附近可能误标；连续异常占邻域多数时 median 与 MAD 会被污染。平滑趋势可以改用 local_linear。",
        "reference": ("Pearson et al. (2016), Generalized Hampel Filters", "https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf"),
    },
    "rolling_iqr": {
        "name": "Rolling IQR fences · 局部分位数边界",
        "mode": "Offline / centered · 使用前后交易",
        "summary": "用局部四分位距构造 Tukey 外围边界；不要求残差对称或正态，是 MAD 的另一种稳健尺度选择。",
        "parameters": COMMON_PARAMETERS + ("iqr_multiplier",),
        "steps": (("3 · Quantiles and the actual outlier fence",
                   "Q1、Q3 使用 NumPy 的 linear 插值经验分位数。中心取两四分位数的中点，使图中的对称 band 精确对应 Tukey fences；abs_floor 只会扩展 band。",
                   r"Q_1=Q_{0.25}(z_{\mathcal N_i}),\ Q_3=Q_{0.75}(z_{\mathcal N_i}),\ I=Q_3-Q_1,\ b_i=(Q_1+Q_3)/2,\ B_i=\max\{a,(k+1/2)I\},\ F_i=\mathbf1\{y_i<b_i-B_i\ \lor\ y_i>b_i+B_i\}"),
                  ("4 · Score versus decision",
                   "当 floor 不激活时，边界正好是 [Q1 − k IQR, Q3 + k IQR]。输出 scale 用 IQR 的正态一致性换算便于比较 score；标准化 fence 倍数是 (k+0.5) × 1.3489795。全局 threshold 对这个方法的边界、scale 和 score 都没有影响。",
                   r"[b_i-B_i,b_i+B_i]=[Q_1-kI,Q_3+kI]\ \text{if }(k+1/2)I\ge a,\quad c_Q=1.3489795003921634,\quad\sigma_i=\max\{I/c_Q,a/[(k+1/2)c_Q]\},\quad B_i=(k+1/2)c_Q\sigma_i")),
        "example": "参考 = [99,100,100,101,101,102] bp：Q1 = 100、Q3 = 101、IQR = 1；k = 3、a = 1 时 b = 100.5，B = 3.5，可接受区间 [97,104] bp。105 bp 标记；104 bp 不标记。σ ≈ 0.741301 bp。",
        "tradeoffs": "偏离阈值可直接看成分位数 fences；小样本、量化 spread 或异常群集会使分位数不稳定。真实市场跳变附近没有额外保护，k 并不是异常概率。",
        "reference": ("NIST Engineering Statistics Handbook, Box Plot / Tukey fences", "https://www.itl.nist.gov/div898/handbook/eda/section3/boxplot.htm"),
    },
    "local_linear": {
        "name": "Robust local linear · 稳健局部时间趋势",
        "mode": "Offline / centered · 使用前后交易",
        "summary": "先拟合局部趋势，再看目标 print 偏离趋势多少；正常的 spread drift 不应仅因偏离窗口 median 而被筛掉。",
        "parameters": COMMON_PARAMETERS + ("threshold",),
        "steps": (("3 · Actual elapsed-time design and distance weights",
                   "时间坐标按距目标的最大真实时间距离归一化，目标预测值就是截距 β0。初始 β = [邻居 median, 0]；tricube 距离权重有 0.05 下限，避免最远参考点失去数值支持。",
                   r"d_i=\max\{1\text{ ns},\max_{j\in\mathcal N_i}|t_j-t_i|\},\ u_j=(t_j-t_i)/d_i,\ X_j=(1,u_j),\ v_j=\max\{[1-\min(|u_j|,0.999)^3]^3,0.05\}"),
                  ("4 · Six Huber IRLS rounds and residual decision",
                   "每轮重新估计残差 MAD，然后做加权最小二乘。固定六轮是这个实现的工程选择；它借鉴 robust local regression，不声称精确复现 LOWESS。最终 σ 使用最终拟合残差的 median-centered MAD。",
                   r"r_j=z_j-X_j\beta,\ s=s(r;\operatorname{median}r),\ h_j=\min\{1,1.345s/\max(|r_j|,10^{-12})\},\ \beta\leftarrow\arg\min_\beta\sum_jv_jh_j(z_j-X_j\beta)^2;\quad b_i=\beta_0,\ \sigma_i=s(r^{\rm final};\operatorname{median}r^{\rm final}),\ F_i=\mathbf1\{|y_i-b_i|>B_i\}")),
        "example": "邻居时间 = [0,1,2,4,5,6]，spreads = [100,101,102,104,105,106] bp；目标 time = 3、spread = 130。局部线性预测 b ≈ 103 bp，最终残差 ≈ 0，q = 4.5、a = 1 时 B = 1 bp。目标残差 ≈ 27 bp、score ≈ 121.5，标记。103 bp 正常 print 不标记。",
        "tradeoffs": "适合平滑 drift 和不规则交易间隔；真实 abrupt jump、弯曲趋势和污染的短窗口会影响结果。侧向外推并没有市场结构保证。",
        "reference": ("Cleveland (1979), Robust Locally Weighted Regression", "https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf"),
    },
    "local_piecewise": {
        "name": "Two-sided local piecewise trend · 左右独立局部趋势",
        "mode": "Offline / independent side fits · 专门检查趋势拐点",
        "summary": "分别拟合之前和之后的走势并预测到目标时间。先涨后跌的正常拐点可以由两侧独立趋势解释，避免一条直线把峰顶误当异常。",
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Independent left/right robust linear predictions",
                   "左右各至少 max(2,floor(min_neighbors/2)) 个 cohort，分别使用 local_linear 的距离权重和六轮 Huber IRLS，在目标时间预测 pL 与 pR。左右斜率可以不同；同一个 target cohort 不参与任何一侧拟合。",
                   r"p_L=\widehat z_L(\tau(t_i)),\quad p_R=\widehat z_R(\tau(t_i)),\quad b_i=(p_L+p_R)/2,\quad\text{left slope and right slope need not agree}"),
                  ("4 · Agreement, local uncertainty and abstention",
                   "取左右各自 detrended residual MAD scale 的最大值，并有 a/q 下限；不使用一条跨越左右的共同直线残差 scale，因为它会把正常拐点当噪声。只有左右预测在 reversion_tolerance 允许范围内一致，才围绕共同参考评价目标。分歧时仍报告平均 baseline 和 score，但输出 ambiguous_transition、level-change candidate、权重 0；既不验证 clean，也不直接判 execution outlier。缺少任一侧支持同样 abstain。",
                   r"\sigma_i=\max(s_L,s_R,a/q),\quad\Delta_i=|p_R-p_L|,\quad A_i=\{\Delta_i\le\max(a,v\sigma_i)\},\quad F_i=A_i\land\{|y_i-b_i|>\max(a,q\sigma_i)\},\quad\neg A_i\Longrightarrow\texttt{ambiguous\_transition},\quad w_i=0,\ \texttt{jf\_fit\_eligible}=\texttt{False}")),
        "example": "time = [0,1,2,3,4,5,6]，spread = [100,101,102,103,102,101,100] bp。目标 time = 3：左侧走势外推约为 103，右侧走势向前投影约为 103，因此正常峰顶 103 不标记；若同一 time 的目标 print 是 130，左右仍预测 103，可把偏离约 27 bp 标记。",
        "tradeoffs": "连续的 slope change 比 abrupt level jump 更容易由两侧确认；不规则交易、污染短侧、剧烈曲率仍可使外推不可靠。左右预测分歧的 unresolved transition 会降低 fitting coverage，这种 abstention 必须计入统计。",
        "reference": ("Cleveland (1979), robust local regression (component inspiration)", "https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf"),
    },
    "jump_reversion": {
        "name": "Jump and reversion · 跳变后回归确认",
        "mode": "Offline / two-sided confirmation · 需要前后支持",
        "summary": "只有前后的中心都回到同一水平，才把中间的偏离视为疑似 execution effect；持续的新市场水平得到保护。",
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Independent before/after support",
                   "左右各至少 M 个 cohort。分别取 median，尺度仅来自各侧内部的离散度，不能把真实左右水平差当作随机噪声。",
                   r"M=\max\{2,\lfloor\texttt{min\_neighbors}/2\rfloor\},\ |L_i|,|R_i|\ge M,\ m_L=\operatorname{median}z_L,\ m_R=\operatorname{median}z_R,\ D=(z_L-m_L)\cup(z_R-m_R),\ b_i=(m_L+m_R)/2,\ \sigma_i=s(D;0)"),
                  ("4 · Reversion gate and persistent-shift guard",
                   "左右中心相差不超过 max(a,vσ) 才确认回归；目标本身还要超出偏离 band。若左右中心差异大于 max(a,qσ)，输出 level-change candidate 并保护该交易。不是要求紧接着下一笔必须回归。",
                   r"\Delta_i=|m_R-m_L|,\ A_i=\{\Delta_i\le\max(a,v\sigma_i)\},\ S_i=\{\Delta_i>\max(a,q\sigma_i)\},\ F_i=A_i\land\{|y_i-b_i|>B_i\}\land\neg S_i")),
        "example": "左 [100,100,100]、右 [100,100,100]、目标 130 bp：b = 100、Δ = 0，q = 4.5、a = 1、v = 2 时确认回归且偏离 30 > 1，标记。改成右 [130,130,130]：Δ = 30 > 1，持续水平变化得到保护。",
        "tradeoffs": "适合保护真正的 level shift。序列开头/结尾缺一侧时 abstain；短暂真实市场变化也可能像异常，长 burst 可能污染两侧。基于三列的工程规则，不是正式 changepoint 检验。",
        "reference": ("Killick et al. (2012), changepoint framework (context; this rule is not PELT)", "https://arxiv.org/abs/1101.1438"),
    },
    "multiscale": {
        "name": "Multiscale confirmation · 多尺度异常确认",
        "mode": "Offline / centered · 使用前后交易",
        "summary": "在短、中、长三个邻域重复判断，减少只在某个窗口下出现的偶然异常，目标是对 cluster 和窗口选择更稳健。",
        "parameters": COMMON_PARAMETERS + ("threshold", "multiscale_votes"),
        "steps": (("3 · Three count/time neighborhoods and Hampel votes",
                   "以基础 window W、horizon H 的 1、2、4 倍组成三个尺度。每个尺度都排除目标 timestamp cohort，达到 min_neighbors 后计算自己的 median、MAD 和严格偏离边界；最终需要 multiscale_votes 个尺度同时支持异常。不同尺度需要的总支持数不随窗口自动增加。",
                   r"f\in\{1,2,4\},\quad(W_f,H_f)=(fW,fH),\quad b_i^{(f)}=\operatorname{median}z_{\mathcal N_i^{(f)}},\quad\sigma_i^{(f)}=s(z_{\mathcal N_i^{(f)}};b_i^{(f)}),\quad F_i^{(f)}=\mathbf1\{|y_i-b_i^{(f)}|>\max(a,q\sigma_i^{(f)})\},\quad F_i=\mathbf1\{\sum_fF_i^{(f)}\ge V\}"),
                  ("4 · Plot the evidence that supports the decision",
                   "标记时，从支持异常的尺度中选 standardized score 最大者作为显示的 baseline、scale 和 band；未标记时，从未投异常票的可用尺度中选 score 最大者（若没有则选全部尺度中 score 最小者），让 band 与最终决定一致。缺少 V 个有支持的尺度会 abstain。这里没有额外 reversal/shift guard：真实跳变也可能跨尺度被标记。",
                   r"f^*=\arg\max_{f\in\mathcal A_i}|y_i-b_i^{(f)}|/\sigma_i^{(f)},\quad\mathcal A_i=\begin{cases}\{f:F_i^{(f)}=1\},&F_i=1\\\{f:F_i^{(f)}=0\text{ and supported}\},&F_i=0\end{cases},\quad(b_i,\sigma_i,B_i)=(b_i^{(f^*)},\sigma_i^{(f^*)},B_i^{(f^*)})")),
        "example": "三个有支持的尺度 center = [100,101,109]、scale = [1,1,1] bp，目标 = 110、q = 4.5、a = 1：scores = [10,9,1]，votes = [True,True,False]。V = 2 时标记，并显示第一尺度的 baseline = 100、B = 4.5；V = 3 时不标记。",
        "tradeoffs": "多尺度共享数据，投票不是独立证据，也不是统计显著性。要求更多 votes 会降低 sensitivity；稀疏数据缺支持时会 abstain。没有强制保护 repricing 的规则，需与 consensus / robust_trend 比较。",
        "reference": ("Pearson et al. (2016), generalized robust local filters (inspiration)", "https://acris.aalto.fi/ws/portalfiles/portal/13003265/art_10.1186_s13634_016_0383_6.pdf"),
    },
    "robust_trend": {
        "name": "Offline robust TV trend · 离线稳健分段趋势",
        "mode": "Offline / whole segment · 共同估计市场水平与稀疏偏离",
        "summary": "同时使用整个交易段的前后信息，把持续的水平变化留给趋势 b，把稀疏的大偏离交给 execution component h。可作为拟合前历史筛选的另一种参考，需要在真实数据上校准。",
        "parameters": COMMON_PARAMETERS + ("threshold", "trend_penalty", "huber_delta", "max_iter", "tolerance"),
        "steps": (("3 · Normalize cohort spreads and separate level from sparse shocks",
                   "μ 是整段 cohort median；s0 使用一阶差分的 median-centered MAD / √2，并有 a/q 下限。归一化后，λ 与 δ 都无量纲。目标函数联合求水平 b 和稀疏偏离 h；目标 cohort 参与稳健拟合，这个方法不是 leave-one-out。",
                   r"\mu=\operatorname{median}z,\quad s_0=\max\{1.4826022185\operatorname{MAD}(\Delta z)/\sqrt2,a/q\},\quad x_j=(z_j-\mu)/s_0,\quad\min_{b,h}\frac12\|x-b-h\|_2^2+\delta\|h\|_1+\lambda\sum_{j=1}^{n-1}|b_{j+1}-b_j|"),
                  ("4 · Equivalent Huber-TV objective and sparse component",
                   "固定 b 时 h 等于 soft threshold，所以等价于 Huber loss + 一阶 total variation。TV 惩罚相邻 cohort 的水平差；每条相邻边权重为 1，并非按真实时间距离缩放，因此 gap segmentation 仍很重要。这里拟合的是 piecewise-constant fused level，不是二阶线性 trend filtering。",
                   r"h_j=\operatorname{soft}(x_j-b_j,\delta),\quad \operatorname{soft}(r,\delta)=\operatorname{sign}(r)\max(|r|-\delta,0),\quad\min_b\sum_j\rho_\delta(x_j-b_j)+\lambda\|Db\|_1,\quad\rho_\delta(r)=\begin{cases}r^2/2,&|r|\le\delta\\\delta|r|-\delta^2/2,&|r|>\delta\end{cases}"),
                  ("5 · Back to raw units, local scale and per-trade flags",
                   "baseline = μ + s0 b。原始单笔 trade 用其 cohort 的 baseline，但 σ 来自周围 leave-cohort-out 的拟合残差 MAD（本方法系数为 1.4826022185，前面的 1.4826 是其近似）；这不是 latent mid 的置信区间。ADMM 以 max_iter / tolerance 控制收敛，未收敛段不强行打 clean/outlier 标签。",
                   r"\widehat m_j=\mu+s_0b_j,\quad e_j=z_j-\widehat m_j,\quad b_i=\widehat m_{c(i)},\quad\sigma_i=\max\{1.4826022185\operatorname{median}_{j\in\mathcal N_i}|e_j-\operatorname{median}_{k\in\mathcal N_i}e_k|,a/q\},\quad F_i=\mathbf1\{|y_i-b_i|>\max(a,q\sigma_i)\}"),
                  ("6 · Solver / ADMM update and convergence",
                   "令 r = x − b、d = Db，约束 b + r = x、Db − d = 0。ADMM 把 r、d 放在同一个可分离 proximal block，保持 two-block 结构；u、v 是 scaled duals。baseline 初值是 edge replication 的 3-cohort median，初始 ρ = 4；只在前 100 次的第 25、50、75、100 轮按 primal/dual ratio 在 [0.125,64] 范围内调整并重新缩放 duals。Huber proximal 下式用 η = x − b − u，soft 是上文定义。",
                   r"b^+=(I+D^\top D)^{-1}[x-r-u+D^\top(d-v)],\quad r^+=\operatorname{prox}_{\rho_\delta/\rho}(x-b^+-u),\quad d^+=\operatorname{soft}(Db^++v,\lambda/\rho),\quad u^+=u+b^++r^+-x,\quad v^+=v+Db^+-d^+;\quad \operatorname{prox}_{\rho_\delta/\rho}(\eta)=\begin{cases}\eta/(1+1/\rho),&|\eta|\le\delta(1+1/\rho)\\\eta-(\delta/\rho)\operatorname{sign}(\eta),&\text{otherwise}\end{cases}"),
                  ("7 · Explicit stopping criterion / 显式收敛判据",
                   "用归一化单位的 primal 与 dual norm 和各自绝对＋相对容差同时判断收敛；ε = tolerance。未达标且已用完 max_iter 时输出 solver_not_converged、零拟合权重与 abstention。求解器准确度与异常敏感度是两个不同设置。",
                   r"p=\|[b+r-x;Db-d]\|_2,\quad q_D=\rho\|r-r^{\rm old}-D^\top(d-d^{\rm old})\|_2,\quad p\le\epsilon[\sqrt{2n-1}+\max(\|[b;Db]\|_2,\|[r;d]\|_2,\|x\|_2)],\quad q_D\le\epsilon[\sqrt n+\rho\|u+D^\top v\|_2]")),
        "example": "整段 [100,100,100,130,100,100,100] bp，q = 4.5、a = 1、δ = 2.5、λ = 8：s0 = 2/9 bp，凸优化解为共同 baseline ≈ 100.092593 bp（数值容差可能有微小差异）。130 bp 残差 ≈ 29.907407 bp，local scale = 2/9、B = 1，标记；六个 100 bp print 不标记。此例需 window ≥ 6、min_neighbors = 6、horizon 覆盖整段。",
        "tradeoffs": "较大 λ 可能过度平滑真实跳变，较小 λ 可能让趋势追随 bad print。对段内部高度 A、长度 L 的大幅暂时 block，拟合该 block 的两条 TV 边约花 2λ|A|；维持 baseline 的 saturated Huber 约花 L(δ|A|−δ²/2)。这只是归一化单位的近似对比：长 burst 更容易被当成真 level，短暂真实变化可能被当成异常。稠密且方向一致的 execution bias 仅靠三列不可识别。",
        "reference": ("Boyd et al. (2011), Distributed Optimization and Statistical Learning via ADMM", "https://web.stanford.edu/~boyd/papers/admm_distr_stats.html"),
    },
    "consensus": {
        "name": "Conservative consensus · 稳健证据与回归确认",
        "mode": "Offline / two-sided confirmation · 默认研究入口",
        "summary": "Hampel 或局部趋势认为偏离大，同时原始水平或趋势投影确认之后回归，才标记；能保护持续 repricing，也适用于平滑 drift。",
        "parameters": COMMON_PARAMETERS + ("threshold", "reversion_tolerance"),
        "steps": (("3 · Raw and trend-adjusted reversion",
                   "raw reversal 就是 jump_reversion。trend reversal 独立拟合左右两侧的稳健线性趋势并外推至目标时间；两侧预测相近时，以其平均为 trend bridge。取左右以及整体线性残差 scale 的最大值，避免短侧拟合使尺度虚假变小。",
                   r"p_L=\widehat z_L(t_i),\ p_R=\widehat z_R(t_i),\ b_i^{\rm tr}=(p_L+p_R)/2,\ \tau_i=\max(s_L,s_R,s_{\rm all}),\ \Delta_i^{\rm tr}=|p_R-p_L|,\ R_i^{\rm tr}=\{\Delta_i^{\rm tr}\le\max(a,v\tau_i)\}\land\{|y_i-b_i^{\rm tr}|>\max(a,q\tau_i)\}"),
                  ("4 · Exact Boolean rule and plotted baseline",
                   "H、L 分别是 Hampel 与 local_linear 的候选；不是多数投票，而是回归确认的 union。只有 raw 与 trend 都认为左右发生大变化才保护 shift。raw reversal 成立时 raw bridge 优先；否则 trend-only 标记使用 trend bridge，从而图上的 band 与标记一致。缺足够任一侧支持就 abstain。",
                   r"S_i^{\rm raw}=\{\Delta_i^{\rm raw}>\max(a,q\sigma_i^{\rm side})\},\quad S_i^{\rm tr}=\{\Delta_i^{\rm tr}>\max(a,q\tau_i)\},\quad F_i=(H_i\lor L_i)\land(R_i^{\rm raw}\lor R_i^{\rm tr})\land\neg(S_i^{\rm raw}\land S_i^{\rm tr})")),
        "example": "time = [0,1,2,4,5,6]、spread = [100,101,102,104,105,106] bp；目标 time = 3、spread = 130。raw medians 为 101 与 105，trend projections 都约为 103；q = 4.5、a = 1 时 trend reversal 确认，L = True，因此标记。用 103 bp 目标则不标记。",
        "tradeoffs": "默认较重视避免筛掉持续变化，但可能漏掉很长 burst；组件相关，agreement 不提供独立置信度。曲率、受污染侧向外推和序列边缘都可能降低支持或准确度。",
        "reference": ("Implementation and research discussion in docs/RESEARCH.md", "https://github.com/ezyhdxm/jump_filter/blob/main/docs/RESEARCH.md"),
    },
    "causal_ewma": {
        "name": "Causal robust EWMA · 在线比较方法",
        "mode": "Optional online comparator · 只使用过去信息",
        "summary": "在线场景下，用过去的稳健水平预测新交易，并通过连续同方向偏离确认新 regime。拟合历史数据无需 causal 时，优先查看上述离线方法。",
        "parameters": COMMON_PARAMETERS + ("threshold", "alpha", "persistence"),
        "steps": (("3 · Historical innovation and clipped EWMA update",
                   "过去 distinct cohorts 的 MAD 提供 σ，中心来自已有 EWMA state。history 受 window/horizon 限制；alpha 以 cohort 为单位而非 clock time。正常时在更新前对每笔原始 trade 打分。",
                   r"e_c=z_c-m_{c-1},\quad B_c=\max(a,q\sigma_c),\quad F_i=\mathbf1\{|y_i-m_{c-1}|>B_c\},\quad m_c=m_{c-1}+\alpha\operatorname{clip}(e_c,-B_c,B_c)"),
                  ("4 · Provisional jumps and persistent regime confirmation",
                   "超过 band 的 cohort innovation 按符号累计；同方向连续 persistence 个后，以当前 cohort median 重置中心并清空 supporting history。确认 cohort 用自己的新 median 对每笔 trade 评分；历史 provisional flags 不追溯修改。之后支持不足会 warm up。",
                   r"\text{same-sign large }e_c\text{ for }P\text{ cohorts}\ \Longrightarrow\ m_c=z_c,\ \text{history}\leftarrow\varnothing,\quad\text{otherwise large trade innovations are provisional jumps}")),
        "example": "已有 center = 100 bp、past σ = 1 bp，q = 4.5、a = 1、alpha = 0.2，新 cohort median = 130：cutoff = 4.5，偏离 30，暂时标记；中心更新到 100 + 0.2×4.5 = 100.9 bp。同方向第三个大 innovation（persistence = 3）确认后中心重置，之前两次 flags 保留。",
        "tradeoffs": "适合检查 future leakage 的在线基准。真 repricing 的最初几笔可能被暂时筛掉；历史 markups 也会污染 scale。历史拟合用离线方法可利用后续证据修正这类判断。",
        "reference": ("Implementation and causal-prefix discussion in docs/RESEARCH.md", "https://github.com/ezyhdxm/jump_filter/blob/main/docs/RESEARCH.md"),
    },
}

# CONFIGURATION LOGIC: The shared three-column fitting policy is explicit in both dashboards.
FITTING_STEPS = (
    ("Use in a downstream fit / 如何交给后续拟合",
     "严格筛选使用 jf_fit_eligible，等价于 status = ok 且没有 outlier 标记的原始 spread。select_fit_data(result, policy='hard') 输出这些记录并添加权重 1。保留已评估且未标记的 shift；不要把 baseline 替换原始成交 spread 当成新证据。invalid、insufficient_history、ambiguous_transition、solver_not_converged 和 provisional_jump 不自动视为 clean。",
     r"\mathcal D_{\rm hard}=\{i:\texttt{jf\_fit\_eligible}_i\}=\{i:\texttt{jf\_status}_i=\texttt{ok}\ \land\ \neg\texttt{jf\_is\_outlier}_i\}"),
    ("Optional influence weights and coverage / 软权重与覆盖率",
     "select_fit_data(result, policy='soft') 默认只取已评估的 ok/outlier records 并以 jf_fit_weight = jf_weight 做软降权：未标记权重 1，标记权重为 cutoff / |residual| 上限 1。权重是影响力建议，不是概率，也不是 inverse variance。provisional 默认排除；coverage 用全部 supplied rows 作分母，避免只看到一个很干净却几乎没支持的数据子集。",
     r"w_i=\begin{cases}1,&\text{evaluated and unflagged}\\\min(1,B_i/\max(|y_i-b_i|,10^{-12})),&\text{flagged}\end{cases},\quad\text{coverage}=N_{\rm evaluated}/N_{\rm supplied},\quad\text{hard retention}=|\mathcal D_{\rm hard}|/N_{\rm supplied}"),
)

# CONFIGURATION LOGIC: Existing public helper remains a lightweight method summary lookup.
METHOD_HELP = {key: card["summary"] for key, card in METHOD_EXPLANATIONS.items()}
