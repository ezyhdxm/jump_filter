# %% [markdown]
# # Jump Filter · Interactive bond trade review
#
# This notebook uses generated trades to demonstrate local spread outliers,
# genuine level changes and sparse trading. No private bond records are included.
# Install the dashboard extra: `python -m pip install -e ".[dashboard]"`.
#
# Pick a CUSIP and method, edit the sliders or exact inputs, then **Apply filter**.
# The statistical dashboard and CSV/settings exports use the last successful Apply.

# %%
# SETUP LOGIC: Imports create no filtering result and read no private dataset.
from jump_filter import FilterConfig, make_demo, show_filter

# %%
# SETUP LOGIC: The reproducible generator is a public synthetic demonstration.
data = make_demo(seed=42)

# CONFIGURATION LOGIC: Spread values are already basis points; the unit label applies no scaling.
settings = FilterConfig(method="consensus", window=31, horizon="3D", max_gap="1D",
                        min_neighbors=6, threshold=4.5, abs_floor=1.0)

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
#
# `panel.result` contains the applied annotated dataframe. The Export button saves
# complete annotations, bond statistics, settings and an offline interactive chart.
# **Compare methods for this bond** uses the applied settings; flag counts alone
# do not establish filtering accuracy.
