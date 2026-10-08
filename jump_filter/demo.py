"""Reproducible synthetic bonds with explicit weekday sessions and construction truth."""

# SETUP LOGIC: simulation dependencies have no external data or I/O side effects.
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: this illustrative schedule is not a claimed bond-market calendar.
DEMO_SCENARIOS = ("isolated_spikes", "short_bursts", "level_shift", "trend",
                  "sparse_gaps", "bid_ask_modes", "zero_mad", "one_sided_markup",
                  "turning_point", "turning_with_drop", "liquidity_shift")
SESSION_MINUTES = 630


def _demo_times(elapsed_minutes):
    """Invert active minutes onto configured 08:00–18:30 NY weekdays."""
    # Input: elapsed_minutes=[0,629,630,1260] ->
    # Output: session=[0,0,1,2],within_minutes=[0,629,0,0],
    # dates=['2026-09-14','2026-09-14','2026-09-15','2026-09-16'].
    # Trick: exact session duration moves to the next open, never to an exclusive close.
    # CORE LOGIC: STEP 1
    elapsed = np.asarray(elapsed_minutes, dtype=np.int64)
    sessions = elapsed // SESSION_MINUTES
    within = elapsed % SESSION_MINUTES
    dates = pd.bdate_range("2026-09-14", periods=int(sessions.max()) + 1)
    local = dates.take(sessions) + pd.Timedelta("8h") + pd.to_timedelta(within, unit="m")

    # Input: local=['2026-09-14 08:00','2026-09-14 18:29','2026-09-15 08:00','2026-09-16 08:00'] ->
    # Output: UTC=['2026-09-14 12:00Z','2026-09-14 22:29Z','2026-09-15 12:00Z','2026-09-16 12:00Z'].
    # Trick: weekdays omit only weekends; localize after hours to honor DST offsets.
    # CORE LOGIC: STEP 2
    return local.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def _demo_mid(scenario, positions, level):
    """Construct market changes independently of statistical outlier flags."""
    # Input: scenario='level_shift',positions=[79,80,81],level=130 ->
    # Output: mid=[130,152,152]; scenario='trend' -> mid=[149.75,150,150.25].
    # Trick: lasting repricing belongs to the fair path and never contamination truth.
    # CORE LOGIC: STEP 1
    mid = np.full(len(positions), float(level))
    if scenario == "level_shift":
        mid += (positions >= 80) * 22
    if scenario == "trend":
        mid += positions * 0.25

    # Input: scenario='turning_point',positions=[79,80,81],level=220 ->
    # Output: mid=[243.7,244,243.35]; scenario='turning_with_drop',level=235 ->
    # Output: mid=[258.7,247,246.35] (decimal floating-point values approximate).
    # Trick: unequal slopes make a true corner; the second case also drops 12bp permanently.
    # CORE LOGIC: STEP 2
    if scenario in ("turning_point", "turning_with_drop"):
        mid += 0.30 * np.minimum(positions, 80) - 0.65 * np.maximum(positions - 80, 0)
    if scenario == "turning_with_drop":
        mid -= (positions >= 80) * 12
    return mid


def _demo_noise(scenario, positions, rng):
    # Input: scenario='liquidity_shift',positions=[79,80,81] ->
    # Output: observation_noise_sigma=[0.4,2.8,2.8]; ordinary scenario ->[0.6,0.6,0.6].
    # Trick: increased clean scatter is a liquidity change, not bad-trade contamination.
    # CORE LOGIC: STEP 1
    sigma = np.full(len(positions), 0.6)
    if scenario == "liquidity_shift":
        sigma = np.where(positions < 80, 0.4, 2.8)
    if scenario == "zero_mad":
        sigma[:] = 0

    # Input: scenario='bid_ask_modes',positions=[0,1,2],normal noise=[0.2,-0.4,0.1] ->
    # Output: noise=[-1.8,1.6,-1.9]; scenario='zero_mad' ->noise=[0,0,0].
    # Trick: legitimate bid/ask bounce stays outside synthetic outlier labels.
    # CORE LOGIC: STEP 2
    noise = rng.normal(0, sigma, len(positions))
    if scenario == "bid_ask_modes":
        noise += np.where(positions % 2, 2.0, -2.0)
    return noise, sigma


def _demo_anomalies(scenario, count):
    # Input: scenario='isolated_spikes',count=160 ->
    # Output: anomaly positions[20,50,110,135] have[28,-24,30,-26],others0.
    # Trick: distortion labels are independent of every detector's decisions.
    # CORE LOGIC: STEP 1
    anomaly = np.zeros(count)
    locations = np.array([20, 50, 110, 135])
    anomaly[locations] = [28, -24, 30, -26]

    # Input: anomaly[50:54]=[-24,0,0,0],scenario='short_bursts' ->
    # Output: anomaly[50:54]=[24,24,24,24]; 'one_sided_markup' ->all4distortions=24.
    # Trick: execution premiums stress masking without asserting a retail cause.
    # CORE LOGIC: STEP 2
    if scenario == "short_bursts":
        anomaly[50:54] = 24
    if scenario == "one_sided_markup":
        anomaly[locations] = 24
    return anomaly


def _demo_arrivals(scenario, count, rng):
    # Input: scenario='sparse_gaps',count=3,sampled base increments=[5,10,15] min ->
    # Output: increments=[1895,10,15]; added gap is31.5 active hours.
    # Trick: within-open-session inactivity survives closure compression.
    # CORE LOGIC: STEP 1
    increments = rng.integers(2, 18, count)
    if scenario == "sparse_gaps":
        increments[::30] += 3 * SESSION_MINUTES

    # Input: scenario='liquidity_shift',count=4,rng=np.random.default_rng(7)
    # after step1 draws[17,12,12,16] -> Output: increments=[5,6,83,40],elapsed=[5,11,94,134].
    # Trick: half-open ranges make the later regime less active and noisier via _demo_noise.
    # CORE LOGIC: STEP 2
    if scenario == "liquidity_shift":
        split = count // 2
        increments[:split] = rng.integers(2, 8, split)
        increments[split:] = rng.integers(25, 95, count - split)
    return np.cumsum(increments), increments


def _demo_quantity(positions):
    # Input: positions=[0,1,2,3,4,5,6,7] ->
    # Output: Quantity=[100000,500000,1000000,2000000,5000000,750000,3000000,100000].
    # Trick: Size cycles independently of spread distortions; quantity is not an injected truth label.
    # CORE LOGIC: STEP 1
    sizes = np.array([100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 750_000, 3_000_000])
    return sizes[np.asarray(positions) % len(sizes)]


def make_demo(seed=42):
    """Eleven scenarios, 160 rows each, in bp; labels are synthetic truth only."""
    # CONFIGURATION LOGIC: reproducible generator and explicitly synthetic CUSIPs.
    rng = np.random.default_rng(seed)
    count = 160
    positions = np.arange(count)
    records = []

    # Input: bond=0,scenario='isolated_spikes',positions=[19,20,21],noise=[0,0,0] ->
    # Output: mid=[100,100,100],anomaly=[0,28,0],spread=[100,128,100].
    # Trick: each bond has an independent path and irregular open-session observations.
    # CORE LOGIC: STEP 1
    for bond, scenario in enumerate(DEMO_SCENARIOS):
        mid = _demo_mid(scenario, positions, 100.0 + 15 * bond)
        noise, sigma = _demo_noise(scenario, positions, rng)
        anomaly = _demo_anomalies(scenario, count)
        elapsed, increments = _demo_arrivals(scenario, count, rng)
        times = _demo_times(elapsed)

        # Input: scenario='turning_with_drop',positions=[79,80,81],anomaly=[0,0,0] ->
        # Output: true_outlier=[False,False,False],true_turning_point=[False,True,False],
        # true_regime_change=[False,True,False],Quantity=[1000000,2000000,5000000].
        # Trick: true corners and lasting drops are legitimate construction events.
        # CORE LOGIC: STEP 2
        turning = (positions == 80) & (scenario in ("turning_point", "turning_with_drop"))
        regime = (positions == 80) & (scenario in ("level_shift", "turning_with_drop"))
        part = pd.DataFrame(dict(CUSIP=f"DEMO{bond + 1:05d}", time=times, spread=mid + noise + anomaly,
                                 Quantity=_demo_quantity(positions),
                                 scenario=scenario, true_outlier=anomaly != 0, true_mid=mid,
                                 true_turning_point=turning, true_regime_change=regime,
                                 observation_noise_sigma=sigma, elapsed_trading_minutes=elapsed,
                                 interarrival_trading_minutes=increments))
        records.append(part)

    # Input: records contain11frames each160rows -> Output: one1760rowframe,
    # CUSIPs=['DEMO00001','DEMO00002','DEMO00003','DEMO00004','DEMO00005','DEMO00006',
    # 'DEMO00007','DEMO00008','DEMO00009','DEMO00010','DEMO00011'],index=0..1759.
    # Trick: provenance records configured sessions without claiming verified holidays.
    # CORE LOGIC: STEP 3
    result = pd.concat(records, ignore_index=True)
    result.attrs["simulation"] = dict(time_basis="configured_weekday_session", session_timezone="America/New_York",
                                      session_open="08:00", session_close="18:30", holidays=[], seed=seed)
    return result
