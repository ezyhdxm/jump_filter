"""Reproducible simulated trades; no customer or proprietary observations."""

# SETUP LOGIC: simulation dependencies.
import numpy as np
import pandas as pd


def make_demo(seed=42):
    """Eight scenarios in basis points; labels are simulation truth only."""
    # CONFIGURATION LOGIC: deterministic seed and explicitly synthetic identifiers.
    rng = np.random.default_rng(seed)
    count = 160
    scenarios = ("isolated_spikes", "short_bursts", "level_shift", "trend",
                 "sparse_gaps", "bid_ask_modes", "zero_mad", "one_sided_markup")
    records = []

    # Input: count=3,bond=2,scenario=level_shift ->
    # Output: positions=[0,1,2],mid=[130,130,130]; with count=82,
    # mid at positions [79,80,81]=[130,152,152].
    # Trick: true_mid is a simulation target; normal bid/ask trades can differ from it.
    # CORE LOGIC: STEP 1
    for bond, scenario in enumerate(scenarios):
        positions = np.arange(count)
        mid = np.full(count, 100.0 + 15 * bond)
        if scenario == "level_shift":
            mid += (positions >= 80) * 22
        if scenario == "trend":
            mid += positions * 0.25
        # Input: count=3,scenario=zero_mad,rng sampled noise=[.2,-.4,.1] ->
        # Output: noise=[0,0,0]; scenario=bid_ask_modes -> noise=[-1.8,1.6,-1.9].
        # Trick: simulated bid/ask bounce is legitimate noise, not a truth-labeled anomaly.
        # CORE LOGIC: STEP 2
        noise = rng.normal(0, 0.6, count)
        if scenario == "bid_ask_modes":
            noise += np.where(positions % 2, 2.0, -2.0)
        if scenario == "zero_mad":
            noise[:] = 0

        # Input: positions=[19,20,21],scenario=isolated_spikes,mid=100,noise=0 ->
        # Output: truth=[False,True,False],spread=[100,128,100].
        # Trick: bursts stress masking; directional markup does not imply identifiable retail status.
        # CORE LOGIC: STEP 3
        anomaly = np.zeros(count)
        locations = np.array([20, 50, 110, 135])
        anomaly[locations] = [28, -24, 30, -26]
        if scenario == "short_bursts":
            anomaly[50:54] = 24
        if scenario == "one_sided_markup":
            anomaly[locations] = 24
        spreads = mid + noise + anomaly
        truth = anomaly != 0

        # Input: increments=[5,10,15] minutes,start=2026-09-14 13:00 UTC ->
        # Output: times=[13:05,13:15,13:30] UTC on that date.
        # Trick: actual-time neighborhoods see gaps; transaction positions alone would miss them.
        # CORE LOGIC: STEP 4
        increments = rng.integers(2, 18, count)
        if scenario == "sparse_gaps":
            increments[::30] += 2 * 24 * 60
        elapsed = np.cumsum(increments)
        times = pd.Timestamp("2026-09-14 13:00", tz="UTC") + pd.to_timedelta(elapsed, unit="m")
        part = pd.DataFrame(dict(CUSIP=f"DEMO{bond + 1:05d}", time=times, spread=spreads,
                                 scenario=scenario, true_outlier=truth, true_mid=mid))
        records.append(part)

    # Input: records=[DataFrame({CUSIP:['DEMO00001'],spread:[100]}),
    # DataFrame({CUSIP:['DEMO00002'],spread:[115]})] ->
    # Output: DataFrame({CUSIP:['DEMO00001','DEMO00002'],spread:[100,115]}),index=[0,1].
    # CORE LOGIC: STEP 5
    return pd.concat(records, ignore_index=True)
