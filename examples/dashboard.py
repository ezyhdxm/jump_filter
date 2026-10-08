# %% [markdown]
# # Jump Filter · Interactive bond trade review
#
# This notebook uses generated trades to demonstrate local spread outliers,
# genuine level changes and sparse trading. No private bond records are included.
# Install the dashboard extra: `python -m pip install -e ".[dashboard]"`.
#
# Pick a CUSIP and method, edit the sliders or exact inputs, then **Apply filter**.
# The statistical dashboard and CSV/settings exports use the last successful Apply.
# **Method & mathematics** immediately follows your selected method, even before Apply.
# It renders formulas, numerical examples, parameter effects and failure/coverage rules.
# Historical fit screening may use both earlier and later trades; Causal EWMA is an
# optional online comparator. Methods include Hampel, IQR fences, robust local linear,
# reversal, multiscale Hampel, two-sided local piecewise trend, offline Huber-TV
# level and conservative consensus.
# The trading-calendar controls remove configured closures from time distances;
# local support, volatility, reference density and active gaps remain auditable.

# %%
# SETUP LOGIC: Imports create no filtering result and read no private dataset.
from jump_filter import FilterConfig, make_demo, show_filter, select_fit_data

# %%
# SETUP LOGIC: The reproducible generator is a public synthetic demonstration.
data = make_demo(seed=42)

# CONFIGURATION LOGIC: Spread values are already basis points; the unit label applies no scaling.
settings = FilterConfig(method="consensus", window=31, horizon="3D", max_gap="1D",
                        min_neighbors=6, threshold=4.5, abs_floor=1.0,
                        time_basis="trading", session_timezone="America/New_York",
                        session_open="08:00", session_close="18:30")

# UI LOGIC: All algorithm settings can be changed through sliders and exact input boxes.
panel = show_filter(data, config=settings, cusip_col="CUSIP", time_col="time",
                    spread_col="spread", timezone="UTC", unit="bp")
panel.run()

# %% [markdown]
# ## Use your own dataframe
#
# Replace `data` above with your existing dataframe and supply its exact column names.
# For example, BondCliQ data can map `CUSIP`, `time`, and `BM_SPREAD`.
# If `BM_SPREAD` is in percentage points, use `unit="percentage points"`
# and choose `abs_floor` in the same units (1 bp = 0.01 percentage points).
# The engine preserves original rows and index; flags and reasons are added.
#
# Three columns cannot identify whether a flagged trade was retail, distressed,
# marked up/down, or charged a commission. Treat each flag as a review candidate.
# Centered Hampel, local trend and reversal methods use future observations.
# Causal robust EWMA uses only prior timestamps and marks unconfirmed jumps.
# Offline Huber-TV jointly fits the whole gap-separated segment, while its local
# residual scale excludes the target timestamp cohort. Solver failures abstain.
# Two-sided local piecewise trends can retain a continuous uptrend/downtrend corner;
# conflicting side predictions abstain as `ambiguous_transition` rather than
# automatically declaring the observation clean or an execution outlier.
# With `time_basis="trading"`, horizon/max_gap are cumulative configured open time.
# The weekday calendar is a research assumption. Specify `holidays` or pass an
# exact timezone-aware `session_schedule` to `filter_trades` for early closes.
# Pass the same table to `show_filter(..., session_schedule=calendar)` to use it
# in the notebook dashboard. Applied settings export its fingerprint and the
# authoritative schedule CSV for reproducibility.
#
# `panel.result` contains the applied annotated dataframe. The Export button saves
# complete annotations, bond statistics, settings and an offline interactive chart.
# **Compare methods for this bond** uses the applied settings; flag counts alone
# do not establish filtering accuracy.

# %%
# Input: panel.result rows jf_row_id=[0,1,2],jf_status=['ok','outlier','ambiguous_transition'],
# jf_is_outlier=[False,True,False],jf_weight=[1,.15,0],jf_fit_eligible=[True,False,False].
# Output: hard_fit jf_row_id=[0],jf_fit_weight=[1]; soft_fit jf_row_id=[0,1],jf_fit_weight=[1,.15].
# Trick: Applied original spreads are retained; provisional/ambiguous/unsupported rows are excluded by default.
# CORE LOGIC: STEP 1 — Select explicit hard and soft fitting populations from the applied review.
hard_fit = select_fit_data(panel.result, policy="hard")
soft_fit = select_fit_data(panel.result, policy="soft")

# REPORTING LOGIC: Hard fit keeps jf_fit_eligible with weight 1; soft fit exposes jf_fit_weight.
# Unsupported/invalid/provisional/ambiguous-transition/solver-failure rows are excluded.
# Soft weights limit influence; they are neither probabilities nor inverse variances.
print(f"Supplied: {len(data):,}; hard-fit candidates: {len(hard_fit):,}; soft-fit candidates: {len(soft_fit):,}")
