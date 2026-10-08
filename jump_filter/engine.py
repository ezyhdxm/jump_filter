"""CUSIP-isolated robust screening with explicit abstention and audit columns."""

# SETUP LOGIC: numerical imports; importing this module performs no I/O.
from collections import deque
from dataclasses import replace
from functools import lru_cache
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .config import FilterConfig
from .clock import trading_clock


def _parse_time(value, timezone):
    # INPUT VALIDATION LOGIC: normalize a timestamp, without guessing epoch units.
    if isinstance(value, (int, float, np.number)) or pd.isna(value):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _parse_times(values, timezone):
    # Input: aware datetime column ['2025-01-01T00:00Z',NaT] with microsecond units ->
    # Output: UTC nanosecond series ['2025-01-01T00:00Z',NaT] with positional index [0,1].
    # Trick: typed datetime arrays need no scalar reparsing; force ns for pandas 2/3 unit parity.
    # CORE LOGIC: STEP 1
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.reset_index(drop=True).dt.tz_convert("UTC").dt.as_unit("ns")
    if pd.api.types.is_datetime64_dtype(values.dtype):
        localized = values.reset_index(drop=True).dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        return localized.dt.tz_convert("UTC").dt.as_unit("ns")

    # Input: mixed values=['2025-01-01',123,'bad'],timezone='UTC' ->
    # Output: ['2025-01-01T00:00Z',NaT,NaT] with dtype datetime64[ns,UTC].
    # Trick: the scalar fallback preserves rejection of numeric epochs and mixed-zone/DST ambiguity.
    # CORE LOGIC: STEP 2
    return pd.Series([_parse_time(value, timezone) for value in values], dtype="datetime64[ns, UTC]")


def _resolve_backend(method, backend, size):
    # CONFIGURATION LOGIC: optional acceleration does not change filter controls or add mandatory imports.
    if backend not in ("auto", "python", "numba"):
        raise ValueError("backend must be 'auto', 'python' or 'numba'")
    if backend == "python" or (backend == "auto" and size < 10000):
        return None
    if method not in ("hampel", "local_linear", "jump_reversion", "local_piecewise", "consensus"):
        return None
    try:
        from . import accelerator
    except (ImportError, RuntimeError) as error:
        if backend == "numba":
            raise ImportError("the numba backend requires jump-filter[speed]") from error
        return None
    return accelerator


@lru_cache(maxsize=128)
def _duration_ns(value):
    # CACHEING LOGIC: immutable duration strings reuse their validated nanosecond conversion.
    return pd.Timedelta(value).value


def _scale(values, center, floor):
    # Input: values=[100,100,101], center=100, floor=1 -> Output: 1.0.
    # Trick: 1.4826 calibrates Gaussian MAD; floor prevents zero-MAD division.
    # CORE LOGIC: STEP 1
    return max(float(1.4826 * np.median(np.abs(values - center))), floor)


def _linear_reference(times, values, target, floor):
    """Local Huber IRLS; elapsed-time coordinates, not trade-row distances."""
    # Input: times=[0,1,2], values=[100,101,102], target=1 ->
    # Output: x=[-1,0,1], X=[[1,-1],[1,0],[1,1]], beta=[101,0].
    # Trick: scaling time keeps nanosecond timestamps out of the linear solve.
    # CORE LOGIC: STEP 1
    distance = np.maximum(np.max(np.abs(times - target)), 1)
    x = (times - target) / distance
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.array([float(np.median(values)), 0.0])
    weights = np.maximum((1 - np.minimum(np.abs(x), 0.999) ** 3) ** 3, 0.05)

    # Input: X=[[1,-1],[1,0],[1,1]], y=[100,101,102], beta=[101,0],
    # floor=1 -> Output: beta approximately [101,1], residual approximately [0,0,0].
    # Trick: residual Huber weights suppress spikes; fixed rounds bound runtime.
    # CORE LOGIC: STEP 2
    for _ in range(6):
        residual = values - design @ beta
        sigma = _scale(residual, float(np.median(residual)), floor)
        robust = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights * robust)
        beta = np.linalg.lstsq(design * root[:, None], values * root, rcond=None)[0]

    # Input: y=[100,101,102], fitted=[100,101,102], beta=[101,1],floor=1 ->
    # Output: reference=101.0, residual scale=1.0 (up to floating-point precision).
    # CORE LOGIC: STEP 3
    residual = values - design @ beta
    return float(beta[0]), _scale(residual, float(np.median(residual)), floor)


def _neighbors(times, position, config):
    # Input: times=[0,10,20,30,40], position=2,window=4,horizon=15ns ->
    # Output: left=[1], right=[3], reference=[1,3].
    # Trick: both elapsed-time and count bounds apply; target cohort is excluded.
    # CORE LOGIC: STEP 1
    target = times[position]
    horizon = _duration_ns(config.horizon)
    left_bound = np.searchsorted(times, target - horizon, side="left")
    right_bound = np.searchsorted(times, target + horizon, side="right")
    left_limit = config.window // 2
    right_limit = config.window - left_limit
    left = np.arange(max(left_bound, position - left_limit), position)
    right = np.arange(position + 1, min(right_bound, position + right_limit + 1))
    return left, right, np.concatenate((left, right))


def _reference_diagnostics(result, rows, reference_times):
    # Input: reference_times=[0,60000000000,120000000000]ns,rows=[4] ->
    # Output: row4 reference_span_minutes=2,reference_density_per_hour=90.
    # Trick: density counts distinct cohorts over their observed span, not scheduled open hours.
    # CORE LOGIC: STEP 1
    span = float(reference_times[-1] - reference_times[0]) / 60e9 if len(reference_times) > 1 else np.nan
    density = len(reference_times) * 60 / span if span > 0 else np.nan
    for row in rows:
        result["jf_reference_span_minutes"][row] = span
        result["jf_reference_density_per_hour"][row] = density


def _references(times, values, position, config):
    # Input: times=[0,1,2,3,4,5,6],values=[100,100,100,130,100,100,100],
    # position=3,window=6,horizon=1s,min_neighbors=6,threshold=4.5,abs_floor=1 ->
    # Output: left=[0,1,2],right=[4,5,6],refs=[100,100,100,100,100,100],
    # median=100,scale=2/9,linear=100,linear_scale=2/9 (numerically approximate).
    # Trick: abs_floor/threshold makes the ultimate raw cutoff at least abs_floor.
    # CORE LOGIC: STEP 1
    left, right, neighbors = _neighbors(times, position, config)
    if len(neighbors) < config.min_neighbors:
        return None
    refs = values[neighbors]
    floor = config.abs_floor / config.threshold
    median = float(np.median(refs))
    sigma = _scale(refs, median, floor)

    # Input: method=hampel,median=100,sigma=1 -> Output: linear=100,linear_scale=1;
    # method=local_linear,times=[0,1,2],refs=[100,101,102],target=1,floor=1 ->
    # Output: linear approximately101,linear_scale=1.
    # Trick: methods that do not use regression avoid unnecessary numerical fitting.
    # CORE LOGIC: STEP 2
    linear, linear_scale = median, sigma
    if config.method in ("local_linear", "consensus"):
        linear, linear_scale = _linear_reference(times[neighbors], refs, times[position], floor)

    # Input: left=[100,100,100],right=[100,100,100], min_neighbors=6 ->
    # Output: side_ok=True, left_mid=100,right_mid=100,side_scale=2/9.
    # Trick: each side needs independent timestamps, so one late print cannot confirm reversion.
    # CORE LOGIC: STEP 3
    minimum_side = max(2, config.min_neighbors // 2)
    side_ok = min(len(left), len(right)) >= minimum_side
    left_mid = float(np.median(values[left])) if len(left) else np.nan
    right_mid = float(np.median(values[right])) if len(right) else np.nan
    deviations = np.concatenate((values[left] - left_mid, values[right] - right_mid))
    side_scale = _scale(deviations, 0.0, floor)

    # Input: left times=[0,1,2],values=[100,101,102],right times=[4,5,6],
    # values=[104,105,106],target=3,floor=2/9,threshold=4.5,abs_floor=1,
    # reversion_tolerance=2,linear_scale=2/9 -> Output: projected centers=[103,103],
    # trend_scale=2/9,trend_reverted=True,trend_shift=False,trend_bridge=103
    # (regression quantities numerically approximate).
    # Trick: independent side extrapolations distinguish smooth trends from permanent steps.
    # CORE LOGIC: STEP 4
    left_projected, right_projected, trend_scale = left_mid, right_mid, side_scale
    if side_ok and config.method in ("consensus", "local_piecewise"):
        left_projected, left_sigma = _linear_reference(times[left], values[left], times[position], floor)
        right_projected, right_sigma = _linear_reference(times[right], values[right], times[position], floor)
        trend_scale = max(left_sigma, right_sigma, linear_scale if config.method == "consensus" else floor)
    trend_delta = abs(right_projected - left_projected)
    trend_reverted = bool(side_ok and trend_delta <= max(config.abs_floor, config.reversion_tolerance * trend_scale))
    trend_shift = bool(side_ok and trend_delta > max(config.abs_floor, config.threshold * trend_scale))
    trend_bridge = (left_projected + right_projected) / 2

    # Input: raw left=100,right=100,side_scale=1,trend_shift=False,threshold=4.5 ->
    # Output: consensus_shift=False.
    # Trick: median confirmation tolerates bursts; projected confirmation tolerates smooth trends.
    # CORE LOGIC: STEP 5
    consensus_shift = bool(side_ok and abs(right_mid - left_mid) > max(config.abs_floor, config.threshold * side_scale) and trend_shift)

    # Input: left_mid=100,right_mid=100,side_scale=2/9,tolerance=2,floor=1 ->
    # Output: reverted=True, shift=False, bridge=100, n_reference=6.
    # Trick: a persistent level difference is a regime candidate, never proof of a bad trade.
    # CORE LOGIC: STEP 6
    delta = abs(right_mid - left_mid)
    reverted = bool(side_ok and delta <= max(config.abs_floor, config.reversion_tolerance * side_scale))
    shift = bool(side_ok and delta > max(config.abs_floor, config.threshold * side_scale))
    bridge = (left_mid + right_mid) / 2 if side_ok else median
    return dict(median=median, sigma=sigma, linear=linear, linear_scale=linear_scale,
                bridge=bridge, side_scale=side_scale, reverted=reverted,
                shift=shift, n=len(neighbors), side_ok=side_ok,
                trend_bridge=trend_bridge, trend_scale=trend_scale,
                trend_reverted=trend_reverted, trend_shift=trend_shift,
                consensus_shift=consensus_shift)


def _retrospective_row(value, references, config):
    # Input: value=130,median=100,sigma=1,linear=100,linear_scale=1,bridge=100,
    # side_scale=1,reverted=True,trend_bridge=100,trend_scale=1,trend_reverted=True,
    # threshold=4.5,abs_floor=1 -> Output: hampel=True,linear=True,
    # reversal=True,trend_reversal=True.
    # CORE LOGIC: STEP 1
    r = references
    hampel = abs(value - r["median"]) > max(config.abs_floor, config.threshold * r["sigma"])
    linear = abs(value - r["linear"]) > max(config.abs_floor, config.threshold * r["linear_scale"])
    reversal = r["reverted"] and abs(value - r["bridge"]) > max(config.abs_floor, config.threshold * r["side_scale"])
    trend_reversal = r["trend_reverted"] and abs(value - r["trend_bridge"]) > max(config.abs_floor, config.threshold * r["trend_scale"])

    # Input: method=consensus,reversal=True,trend_reversal=False,hampel=True,
    # linear=True,reverted=True,bridge=100,side_scale=1,trend_bridge=103,trend_scale=1 ->
    # Output: use_raw=True,consensus_center=100,consensus_scale=1,center=100,
    # sigma=1,flagged=True.
    # Trick: choose the confirming baseline per trade so a flag agrees with its displayed boundary.
    # CORE LOGIC: STEP 2
    use_raw = reversal or (not trend_reversal and r["reverted"])
    consensus_center = r["bridge"] if use_raw else r["trend_bridge"]
    consensus_scale = r["side_scale"] if use_raw else r["trend_scale"]
    choices = {"hampel": (r["median"], r["sigma"], hampel),
               "local_linear": (r["linear"], r["linear_scale"], linear),
               "jump_reversion": (r["bridge"], r["side_scale"], reversal),
               "consensus": (consensus_center, consensus_scale, (reversal or trend_reversal) and (hampel or linear))}
    center, sigma, flagged = choices[config.method]

    # Input: method=consensus,flagged=True,shift=False,side_ok=True ->
    # Output: status='outlier',reason='reversion_confirmed_consensus',flagged=True.
    # Trick: protected shifts/edge abstention apply to confirmation methods; baselines remain comparative.
    # CORE LOGIC: STEP 3
    status = "outlier" if flagged else "ok"
    reason = "reversion_confirmed_consensus" if flagged else "within_local_band"
    if config.method in ("jump_reversion", "consensus") and not r["side_ok"]:
        status, reason, flagged = "insufficient_history", "needs_past_and_future_support", False
    elif config.method in ("jump_reversion", "consensus") and r["consensus_shift" if config.method == "consensus" else "shift"]:
        status, reason, flagged = "ok", "persistent_level_change_protected", False
    elif flagged and config.method != "consensus":
        reason = f"{config.method}_threshold_exceeded"

    # Input: value=130,center=100,sigma=1,threshold=4.5,flagged=True ->
    # Output: residual=30,score=30,cutoff=4.5,weight=0.15.
    # Trick: weight is an optional bounded influence suggestion, not an anomaly probability.
    # CORE LOGIC: STEP 4
    residual = value - center
    score = abs(residual) / sigma
    cutoff = max(config.abs_floor, config.threshold * sigma)
    weight = min(1.0, cutoff / max(abs(residual), 1e-12)) if flagged else 1.0
    weight = 0.0 if status == "insufficient_history" else weight
    return center, sigma, score, cutoff, flagged, status, reason, weight


def _assign(result, rows, values, references, config):
    # Input: rows=[2],values=[130],method=consensus,threshold=4.5,abs_floor=1;
    # refs={median:100,sigma:1,linear:100,linear_scale:1,bridge:100,side_scale:1,
    # reverted:True,trend_bridge:100,trend_scale:1,trend_reverted:True,
    # consensus_shift:False,shift:False,side_ok:True,n:6} -> Output: row 2 has
    # baseline=100,scale=1,score=30,threshold=4.5,residual=30,is_outlier=True,
    # status='outlier',reason='reversion_confirmed_consensus',weight=.15,
    # n_reference=6,regime_change=False (all column names have prefix jf_).
    # Trick: positional row IDs preserve duplicate dataframe index labels safely.
    # CORE LOGIC: STEP 1
    for row, value in zip(rows, values):
        center, sigma, score, cutoff, flag, status, reason, weight = _retrospective_row(value, references, config)
        fields = dict(jf_baseline=center, jf_scale=sigma, jf_score=score, jf_threshold=cutoff,
                      jf_residual=value - center, jf_is_outlier=flag, jf_status=status,
                      jf_reason=reason, jf_weight=weight, jf_n_reference=references["n"],
                      jf_regime_change=references["consensus_shift" if config.method == "consensus" else "shift"])
        for name, output in fields.items():
            result[name][row] = output


def _causal_cohort(result, rows, raw, state, history, stamp, config):
    # Input: history=[(0,100),(1,100)], stamp=2,horizon=1ns ->
    # Output: history=[(1,100)], support=1; below min_neighbors -> abstention.
    # Trick: strictly earlier cohorts are admitted; same-time values never train their own score.
    # CORE LOGIC: STEP 1
    horizon = _duration_ns(config.horizon)
    while history and (stamp - history[0][0] > horizon or len(history) > config.window):
        history.popleft()
    values = np.array([value for _, value in history], dtype=float)
    median_now = float(np.median(raw))
    support = len(values)
    floor = config.abs_floor / config.threshold

    # Input: history=[(0,100),(60000000000,101)],rows=[2] ->
    # Output: row2 reference_span_minutes=1,reference_density_per_hour=120.
    # CORE LOGIC: STEP 2
    _reference_diagnostics(result, rows, np.array([time for time, _ in history], dtype=np.int64))

    # Input: history medians=[100,100],current cohort=[130],min_neighbors=6 ->
    # Output: state center=100, support=2,status='insufficient_history',weight=0.
    # Trick: unscored records are not silently treated as fit-eligible normal trades.
    # CORE LOGIC: STEP 3
    if support < config.min_neighbors:
        state.update(center=float(np.median(np.append(values, median_now))), run=0, sign=0)
        for row in rows:
            result["jf_n_reference"][row] = support
        history.append((stamp, median_now))
        return

    # Input: center=100,past=[100,100,100],current median=130,threshold=4.5,
    # floor=2/9,persistence=3,run=0 -> Output: run=1, sign=1,confirmed=False.
    # Trick: persistence is consecutive cohort innovations in one direction, not row count.
    # CORE LOGIC: STEP 4
    center = state["center"]
    sigma = _scale(values, float(np.median(values)), floor)
    cutoff = max(config.abs_floor, config.threshold * sigma)
    innovation = median_now - center
    sign = int(np.sign(innovation)) if abs(innovation) > cutoff else 0
    state["run"] = state["run"] + 1 if sign and sign == state["sign"] else int(bool(sign))
    state["sign"] = sign
    confirmed = state["run"] >= config.persistence

    # Input: previous center=100,current cohort=[120,120],confirmed=True ->
    # Output: center=120,run=0,sign=0,history=[] (prior flags are unchanged).
    # Trick: confirmation resets regime history; only the current and later cohorts benefit.
    # CORE LOGIC: STEP 5
    if confirmed:
        center = median_now
        state.update(center=center, run=0, sign=0)
        history.clear()

    # Input: row spread=130,center=100,sigma=1,cutoff=4.5,confirmed=False ->
    # Output: flag=True,status='provisional_jump',score=30,weight=0.15.
    # CORE LOGIC: STEP 6
    for row, value in zip(rows, raw):
        residual = value - center
        flag = abs(residual) > cutoff
        fields = dict(jf_baseline=center, jf_scale=sigma, jf_residual=residual,
                      jf_score=abs(residual) / sigma, jf_threshold=cutoff, jf_is_outlier=flag,
                      jf_status="provisional_jump" if flag else "ok",
                      jf_reason="unconfirmed_causal_innovation" if flag else "within_causal_band",
                      jf_weight=min(1.0, cutoff / max(abs(residual), 1e-12)) if flag else 1.0,
                      jf_n_reference=support, jf_regime_change=confirmed)
        _publish_fields(result, row, fields)

    # Input: confirmed=False,state center=100,median_now=130,cutoff=4.5,alpha=.2,
    # stamp=10,history=[(9,100)] -> Output: next center=100.9,
    # history=[(9,100),(10,130)].
    # Trick: clipping limits one print's influence; cohort-time updates prevent duplicate amplification.
    # CORE LOGIC: STEP 7
    if not confirmed:
        state["center"] = center + config.alpha * np.clip(median_now - center, -cutoff, cutoff)
    history.append((stamp, median_now))


def _publish_fields(result, row, fields):
    # Input: row=2,fields={jf_score:30},result jf_score=[0,0,0] ->
    # Output: result jf_score=[0,0,30].
    # Trick: mutate only a private positional result buffer, never the source frame.
    # CORE LOGIC: STEP 1
    for name, value in fields.items():
        result[name][row] = value


def _plain_rows(result, rows, raw, center, sigma, cutoff, support, reason):
    # Input: rows=[2],raw=[130],center=100,sigma=1,cutoff=4.5,support=6,
    # reason='multiscale_threshold_exceeded' -> Output: row2 residual=30,score=30,
    # flag=True,status='outlier',weight=.15,n_reference=6.
    # Trick: every row retains its own spread even when the reference is shared by a cohort.
    # CORE LOGIC: STEP 1
    for row, value in zip(rows, raw):
        residual = float(value - center)
        flag = abs(residual) > cutoff
        fields = dict(jf_baseline=center, jf_scale=sigma, jf_residual=residual,
                      jf_score=abs(residual) / sigma, jf_threshold=cutoff, jf_is_outlier=flag,
                      jf_status="outlier" if flag else "ok", jf_n_reference=support,
                      jf_reason=reason if flag else "within_method_band",
                      jf_weight=min(1.0, cutoff / max(abs(residual), 1e-12)) if flag else 1.0)
        _publish_fields(result, row, fields)


def _piecewise_cohort(result, rows, raw, references, config):
    # Input: refs={side_ok:False,n:6},rows=[3] ->
    # Output: row3 status='insufficient_history',n_reference=6,weight=0.
    # Trick: both sides need enough distinct cohorts; one-sided extrapolation cannot confirm a turning point.
    # CORE LOGIC: STEP 1
    for row in rows:
        result["jf_n_reference"][row] = references["n"]
    if not references["side_ok"]:
        return

    # Input: left projected=105,right projected=105,side residual scales=1,
    # raw=[120],threshold=4.5 -> Output: baseline=105,scale=1,cutoff=4.5,flag=True.
    # Trick: independent slopes may have opposite signs; their predictions at the target must agree.
    # CORE LOGIC: STEP 2
    center, sigma = references["trend_bridge"], references["trend_scale"]
    cutoff = max(config.abs_floor, config.threshold * sigma)
    _plain_rows(result, rows, raw, center, sigma, cutoff, references["n"], "local_piecewise_residual_exceeded")

    # Input: projections=[100,120],tolerance=2,scale=1,raw=119 ->
    # Output: baseline=110,score=9,cutoff=4.5,status='ambiguous_transition',
    # flag=False,weight=0,regime_change=True (ineligible for fitting).
    # Trick: disagreement can mean a genuine jump or liquidity change; neither side certifies a clean target.
    # CORE LOGIC: STEP 3
    if not references["trend_reverted"]:
        for row in rows:
            _publish_fields(result, row, dict(jf_status="ambiguous_transition", jf_reason="independent_side_predictions_disagree",
                                             jf_is_outlier=False, jf_weight=0.0, jf_regime_change=True))


def _iqr_cohort(result, rows, raw, times, values, position, config):
    # SETUP LOGIC: additional numerical methods have no import-time dependency on the UI.
    from .offline import iqr_point

    # Input: times=[0,1,2,3,4],values=[100,100,130,100,100],position=2,
    # window=4,horizon=1s,min_neighbors=4 -> Output: neighbors=[0,1,3,4],support=4.
    # CORE LOGIC: STEP 1
    _, _, neighbors = _neighbors(times, position, config)
    support = len(neighbors)
    for row in rows:
        result["jf_n_reference"][row] = support
    if support < config.min_neighbors:
        return

    # Input: refs=[100,100,100,100],raw=[130],k=3,abs_floor=1 ->
    # Output: center=100,scale approximately1/4.721428251,cutoff=1,raw row flagged=True.
    # Trick: symmetric midhinge coordinates represent the exact Tukey fences, including skewed samples.
    # CORE LOGIC: STEP 2
    center, sigma, _, cutoff, _ = iqr_point(values[neighbors], float(raw[0]), config)
    _plain_rows(result, rows, raw, center, sigma, cutoff, support, "rolling_iqr_fence_exceeded")


def _multiscale_cohort(result, rows, raw, times, values, position, config):
    # Input: config window=4,horizon=1s,scale factors=[1,2,4] ->
    # Output: neighborhood windows=[4,8,16],horizons=[1s,2s,4s].
    # Trick: wider references remain inside the same gap segment and CUSIP.
    # CORE LOGIC: STEP 1
    references = []
    for factor in (1, 2, 4):
        expanded = replace(config, window=config.window * factor,
                           horizon=str(pd.Timedelta(config.horizon) * factor))
        _, _, neighbors = _neighbors(times, position, expanded)
        if len(neighbors) >= config.min_neighbors:
            center = float(np.median(values[neighbors]))
            sigma = _scale(values[neighbors], center, config.abs_floor / config.threshold)
            references.append((center, sigma, max(config.abs_floor, config.threshold * sigma), len(neighbors), neighbors))

    # Input: references support=[4,8,16],required votes=2 -> Output: 3 evaluable scales;
    # with support=[] -> Output: n_scales=0,weight=0,status='insufficient_history'.
    # Trick: abstain when too few scales have support; missing scales do not count as clean votes.
    # CORE LOGIC: STEP 2
    for row in rows:
        result["jf_n_scales"][row] = len(references)
        result["jf_n_reference"][row] = max((item[3] for item in references), default=0)
    if references:
        _reference_diagnostics(result, rows, times[max(references, key=lambda item: item[3])[4]])
    if len(references) < config.multiscale_votes:
        return

    # Input: value=130,references=[(100,1,4.5,6),(101,1,4.5,12),(130,1,4.5,24)],
    # required votes=2 -> Output: votes=2,flag=True,selected=(100,1,4.5,6),score=30.
    # Trick: the strongest confirming score defines the visible band; agreement is not independent evidence.
    # CORE LOGIC: STEP 3
    for row, value in zip(rows, raw):
        scores = np.array([abs(value - item[0]) / item[1] for item in references])
        confirmations = np.array([abs(value - item[0]) > item[2] for item in references])
        votes = int(confirmations.sum())
        flagged = votes >= config.multiscale_votes
        candidates = np.flatnonzero(confirmations if flagged else ~confirmations)
        selected = candidates[np.argmax(scores[candidates])] if len(candidates) else int(np.argmin(scores))
        center, sigma, cutoff, support, selected_neighbors = references[selected]

        # Input: votes=2,flagged=True,value=130,selected center=100,scale=1,cutoff=4.5 ->
        # Output: jf_n_votes=2,jf_is_outlier=True,jf_score=30,jf_weight=.15.
        # Trick: an unconfirmed minority vote remains an unflagged review case, not a forced outlier.
        # CORE LOGIC: STEP 4
        _plain_rows(result, [row], [value], center, sigma, cutoff, support, "multiscale_vote_confirmed")
        _reference_diagnostics(result, [row], times[selected_neighbors])
        result["jf_n_votes"][row] = votes
        if not flagged:
            _publish_fields(result, row, dict(jf_is_outlier=False, jf_status="ok", jf_weight=1.0,
                                             jf_reason="multiscale_consensus_not_reached" if votes else "within_method_band"))


def _trend_segment(result, cohorts, times, values, config):
    # SETUP LOGIC: Huber-TV is a joint offline estimator, unlike leave-cohort-out local references.
    from .offline import robust_trend_segment

    # Input: cohorts=3 timestamps,config min_neighbors=6 ->
    # Output: each row keeps status='insufficient_history',weight=0,solver state=None.
    # CORE LOGIC: STEP 1
    if len(values) <= config.min_neighbors:
        return
    centers, scales, diagnostics = robust_trend_segment(times, values, config)

    # Input: diagnostics={converged:False,iterations:1},rows=[0,1] ->
    # Output: solver_converged=[False,False],solver_iterations=[1,1],
    # statuses=['solver_not_converged','solver_not_converged'],weights=[0,0].
    # Trick: a finite approximate solver iterate is not silently accepted as validated fit evidence.
    # CORE LOGIC: STEP 2
    for _, cohort in cohorts:
        for row in cohort["row"].to_numpy(dtype=int):
            _publish_fields(result, row, dict(jf_solver_converged=diagnostics["converged"],
                                             jf_solver_iterations=diagnostics["iterations"]))
            if not diagnostics["converged"]:
                _publish_fields(result, row, dict(jf_status="solver_not_converged", jf_reason="huber_tv_solver_not_converged"))
    if not diagnostics["converged"]:
        return

    # Input: position=2,neighbors=[0,1,3,4],center=100,scale=1,raw=[130],threshold=4.5 ->
    # Output: support=4,cutoff=4.5,raw trade flagged=True when min_neighbors=4.
    # CORE LOGIC: STEP 3
    for position, (_, cohort) in enumerate(cohorts):
        rows = cohort["row"].to_numpy(dtype=int)
        _, _, neighbors = _neighbors(times, position, config)
        _reference_diagnostics(result, rows, times[neighbors])
        result["jf_n_reference"][rows] = len(neighbors)
        if len(neighbors) >= config.min_neighbors:
            cutoff = max(config.abs_floor, config.threshold * scales[position])
            _plain_rows(result, rows, cohort["value"].to_numpy(dtype=float), centers[position],
                        scales[position], cutoff, len(neighbors), "robust_trend_residual_exceeded")

        # Input: current center=120,previous center=100,scale=1,threshold=4.5,abs_floor=1 ->
        # Output: current cohort jf_regime_change=True.
        # Trick: only fitted transitions get markers; these are candidate level changes, not verified events.
        # CORE LOGIC: STEP 4
        if position and abs(centers[position] - centers[position - 1]) > max(config.abs_floor, config.threshold * max(scales[position], scales[position - 1])):
            for row in rows:
                result["jf_regime_change"][row] = True


def _segment(result, data, config):
    # Input: rows [(t=0,s=100),(t=0,s=102),(t=1,s=101)] ->
    # Output: times=[0,1],cohort medians=[101,101], row cohorts=[[0,1],[2]].
    # Trick: distinct-time medians keep duplicate reports from inflating reference support.
    # CORE LOGIC: STEP 1
    cohorts = list(data.groupby("stamp", sort=True))
    times = np.array([stamp for stamp, _ in cohorts], dtype=np.int64)
    values = np.array([part["value"].median() for _, part in cohorts], dtype=float)
    state = dict(center=values[0], run=0, sign=0)
    history = deque()

    # Input: method=robust_trend,cohort medians=[100,100,130,100,100],min_neighbors=4 ->
    # Output: the entire five-cohort segment is passed once to the joint solver.
    # Trick: global optimization is performed once per gap segment, never once per row.
    # CORE LOGIC: STEP 2
    if config.method == "robust_trend":
        _trend_segment(result, cohorts, times, values, config)
        return

    # Input: cohorts=[(0,rows=[0],value=[100]),(1,rows=[1],value=[101])].
    # Output: iteration0 rows=[0],raw=[100];iteration1 rows=[1],raw=[101].
    # Trick: raw values are scored individually; the reference series uses cohort medians.
    # CORE LOGIC: STEP 3
    for position, (stamp, part) in enumerate(cohorts):
        rows = part["row"].to_numpy(dtype=int)
        raw = part["value"].to_numpy(dtype=float)

        # Input: times=[0,60000000000,120000000000],position=1,window=2 ->
        # Output: neighbors=[0,2],row1 reference_span_minutes=2,density=60/hour.
        # Trick: causal and multiscale methods replace this centered audit with their actual reference set.
        # CORE LOGIC: STEP 4
        _, _, neighbors = _neighbors(times, position, config)
        _reference_diagnostics(result, rows, times[neighbors])
        result["jf_n_reference"][rows] = len(neighbors)

        # Input: times=[0,1,2],cohort row IDs=[[0],[1],[2]],values=[100,100,100],
        # method=causal_ewma,min_neighbors=2,horizon=1s,threshold=4.5,abs_floor=1 ->
        # Output: row statuses=['insufficient_history','insufficient_history','ok'],
        # row reference counts=[0,1,2],row baselines=[NaN,NaN,100].
        # CORE LOGIC: STEP 5
        if config.method == "causal_ewma":
            _causal_cohort(result, rows, raw, state, history, stamp, config)
        elif config.method == "rolling_iqr":
            _iqr_cohort(result, rows, raw, times, values, position, config)
        elif config.method == "multiscale":
            _multiscale_cohort(result, rows, raw, times, values, position, config)
        else:
            _local_cohort(result, rows, raw, times, values, position, config)


def _array_segment(result, rows, raw, stamps, config):
    # Input: rows=[0,1,2],stamps=[0,0,1],raw=[100,102,101] ->
    # Output: boundaries=[0,2,3],times=[0,1],values=[101,101].
    # Trick: one vectorized median reduction replaces a DataFrame per timestamp while excluding full cohorts.
    # CORE LOGIC: STEP 1
    starts = np.r_[0, np.flatnonzero(stamps[1:] != stamps[:-1]) + 1]
    boundaries = np.r_[starts, len(stamps)]
    times = stamps[starts]
    codes = np.repeat(np.arange(len(starts)), np.diff(boundaries))
    values = pd.Series(raw).groupby(codes, sort=False).median().to_numpy(dtype=float)
    state, history = dict(center=values[0], run=0, sign=0), deque()

    # Input: method=robust_trend,min_neighbors=6,rows=[0,1],raw=[100,101],stamps=[0,1] ->
    # Output: row statuses=['insufficient_history','insufficient_history'],weights=[0,0],
    # baselines=[NaN,NaN],reference counts=[0,0]; the two-cohort segment cannot support the solver.
    # Trick: the joint optimization path stays unchanged; local methods use array slices below.
    # CORE LOGIC: STEP 2
    if config.method == "robust_trend":
        data = pd.DataFrame(dict(row=rows, value=raw, stamp=stamps))
        _trend_segment(result, list(data.groupby("stamp", sort=True)), times, values, config)
        return

    # Input: boundaries=[0,2,3],position=0 -> Output: selected rows=[0,1],raw=[100,102].
    # Trick: slices are views of sorted private arrays; original frame/index objects are not mutated.
    # CORE LOGIC: STEP 3
    for position, stamp in enumerate(times):
        selected = slice(boundaries[position], boundaries[position + 1])
        cohort_rows, cohort_raw = rows[selected], raw[selected]
        _, _, neighbors = _neighbors(times, position, config)
        _reference_diagnostics(result, cohort_rows, times[neighbors])
        result["jf_n_reference"][cohort_rows] = len(neighbors)

        # Input: method=hampel,times=[0,1,2,3,4]ns,values=[100,100,130,100,100],position=2,
        # rows=[2],raw=[130],window=4,horizon='1s',min_neighbors=4,threshold=4.5,abs_floor=1 ->
        # Output: row2 baseline=100,scale=2/9,score=135,cutoff=1,residual=30,
        # status='outlier',reason='hampel_threshold_exceeded',weight=1/30,reference count=4.
        # Trick: only storage/preparation changed; each fallback dispatch uses the established numerical helper.
        # CORE LOGIC: STEP 4
        if config.method == "causal_ewma":
            _causal_cohort(result, cohort_rows, cohort_raw, state, history, stamp, config)
        elif config.method == "rolling_iqr":
            _iqr_cohort(result, cohort_rows, cohort_raw, times, values, position, config)
        elif config.method == "multiscale":
            _multiscale_cohort(result, cohort_rows, cohort_raw, times, values, position, config)
        else:
            _local_cohort(result, cohort_rows, cohort_raw, times, values, position, config)


def _bulk_groups(result, data, utc_times, config, accelerator):
    # Input: sorted IDs=[A,A,B],stamps=[0,1,5],rows=[2,0,1] ->
    # Output: rows=[2,0,1],same_id=[False,True,False],gaps=[NaN,1/60e9,NaN] minutes.
    # Trick: gaps are computed only within a bond; shuffled/duplicate source indexes remain positional.
    # CORE LOGIC: STEP 1
    rows = data["row"].to_numpy(dtype=int)
    stamps = data["stamp"].to_numpy(dtype=np.int64)
    raw = data["value"].to_numpy(dtype=float)
    ids = data["id"].to_numpy()
    same_id = np.r_[False, ids[1:] == ids[:-1]]
    selected_gaps = np.r_[np.nan, np.diff(stamps).astype(float) / 60e9]
    wall_gaps = np.r_[np.nan, np.diff(utc_times[rows]).astype(float) / 60e9]
    selected_gaps[~same_id], wall_gaps[~same_id] = np.nan, np.nan

    # Input: selected gaps=[NaN,1],wall gaps=[NaN,3691],max_gap=10min ->
    # Output: boundary=[False,True],segment starts=[0],segment ends=[2].
    # Trick: a market closure is audited separately; only the configured clock gap splits references.
    # CORE LOGIC: STEP 2
    result["jf_gap_minutes"][rows] = selected_gaps
    result["jf_wall_gap_minutes"][rows] = wall_gaps
    result["jf_session_boundary"][rows] = np.isfinite(wall_gaps) & (wall_gaps > selected_gaps + 1e-9)
    segment_start = ~same_id | (selected_gaps > pd.Timedelta(config.max_gap).value / 60e9)
    boundaries = np.r_[np.flatnonzero(segment_start), len(rows)]

    # Input: backend=python,segment boundaries=[0,2,3] ->
    # Output: independent array segments rows[0:2] and rows[2:3] enter the reference helpers.
    # Trick: unsupported methods keep the same solver/reference algorithm with cheaper preparation.
    # CORE LOGIC: STEP 3
    if accelerator is None:
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            _array_segment(result, rows[begin:end], raw[begin:end], stamps[begin:end], config)
        return

    # Input: stamps=[0,0,1,0],segment starts=[True,False,False,True] ->
    # Output: cohort starts=[0,2,3],row cohorts=[0,0,1,2],cohort medians=[101,103,110] for raw=[100,102,103,110].
    # Trick: identical clock values on different bonds/segments are never combined into one cohort.
    # CORE LOGIC: STEP 4
    cohort_start = segment_start | np.r_[False, stamps[1:] != stamps[:-1]]
    cohort_rows = np.flatnonzero(cohort_start)
    row_cohorts = np.cumsum(cohort_start) - 1
    values = pd.Series(raw).groupby(row_cohorts, sort=False).median().to_numpy(dtype=float)
    times = stamps[cohort_rows]
    segment_cohorts = row_cohorts[boundaries[:-1]]
    segment_ends = np.r_[segment_cohorts[1:], len(times)]
    lengths = segment_ends - segment_cohorts

    # Input: times=[0,1,5]ns,values=raw=[100,101,110],row_cohorts=[0,1,2],segment cohort
    # starts=[0,2],ends=[2,3],method=hampel,window=4,horizon='1s',min_neighbors=4 ->
    # Output: per-cohort starts=[0,0,2],ends=[2,2,3]; audit shape=(3,14),column6=[1,1,0],
    # columns0..4/11..12 allNaN,columns5/7..10/13 all0 (every row has insufficient support).
    # Trick: amortize compilation/dispatch across all CUSIPs instead of crossing Python per bond/cohort.
    # CORE LOGIC: STEP 5
    starts = np.repeat(segment_cohorts, lengths)
    ends = np.repeat(segment_ends, lengths)
    audit = accelerator.local_kernel(times, values, starts, ends, row_cohorts, raw,
        accelerator.LOCAL_METHODS.index(config.method), config.window, pd.Timedelta(config.horizon).value,
        config.min_neighbors, config.threshold, config.abs_floor, config.reversion_tolerance)

    # Input: row IDs=[2,0],audit baselines=[100,101],status codes=[2,1] ->
    # Output: source row2 baseline=100,status='outlier';source row0 baseline=101,status='ok'.
    # Trick: private integer transport codes restore every original string/boolean audit field explicitly.
    # CORE LOGIC: STEP 6
    fields = ("jf_baseline", "jf_scale", "jf_score", "jf_threshold", "jf_residual", "jf_weight",
              "jf_n_reference", "jf_is_outlier", "jf_regime_change")
    for position, name in enumerate(fields):
        result[name][rows] = audit[:, position]
    result["jf_status"][rows] = accelerator.STATUS_NAMES[audit[:, 9].astype(int)]
    result["jf_reason"][rows] = accelerator.REASON_NAMES[audit[:, 10].astype(int)]
    result["jf_reference_span_minutes"][rows] = audit[:, 11]
    result["jf_reference_density_per_hour"][rows] = audit[:, 12]

    # Input: row cohorts=[0,1,1,2],retry markers=[0,1,0,0] ->
    # Output: cohort1's two raw trades are recomputed once by the original Python/SVD helpers.
    # Trick: rounding-near decisions use the established exact solver without changing cutoffs or audit reasons.
    # CORE LOGIC: STEP 7
    retry_cohorts = np.unique(row_cohorts[audit[:, 13] > 0])
    row_ends = np.r_[cohort_rows[1:], len(rows)]
    for position in retry_cohorts:
        selected = slice(cohort_rows[position], row_ends[position])
        begin, end = starts[position], ends[position]
        result["jf_regime_change"][rows[selected]] = False
        _local_cohort(result, rows[selected], raw[selected], times[begin:end], values[begin:end], position - begin, config)


def _local_cohort(result, rows, raw, times, values, position, config):
    # Input: values=[100,100,130,100,100],position=2,min_neighbors=4,
    # method=hampel,window=4,horizon=1s,threshold=4.5,abs_floor=1 ->
    # Output: target row baseline=100,flag=True; insufficient references leave weight0.
    # CORE LOGIC: STEP 1
    refs = _references(times, values, position, config)
    if refs is None:
        return
    if config.method == "local_piecewise":
        _piecewise_cohort(result, rows, raw, refs, config)
    else:
        _assign(result, rows, raw, refs, config)


def _compiled_groups(result, data, utc_times, config, accelerator, backend):
    # Input: backend='auto',compiler raises a NumbaError before publishing scores ->
    # Output: all segments are scored by Python and actual backend=None;
    # backend='numba' with that error -> Output: the compiler error is raised.
    # Trick: only recognized compiler errors trigger fallback; numerical/domain errors remain visible.
    # CORE LOGIC: STEP 1
    try:
        _bulk_groups(result, data, utc_times, config, accelerator)
    except (accelerator.COMPILATION_ERROR, ImportError):
        if backend == "numba":
            raise
        _bulk_groups(result, data, utc_times, config, None)
        return None
    return accelerator


def _empty_result(size, config):
    # SETUP LOGIC: separate validity, scoring, and flagging states explicitly.
    result = {name: np.full(size, np.nan) for name in (
        "jf_score", "jf_baseline", "jf_scale", "jf_residual", "jf_threshold",
        "jf_gap_minutes", "jf_wall_gap_minutes", "jf_reference_span_minutes", "jf_reference_density_per_hour")}
    result.update(jf_row_id=np.arange(size), jf_is_outlier=np.zeros(size, dtype=bool),
                  jf_regime_change=np.zeros(size, dtype=bool), jf_weight=np.zeros(size),
                  jf_session_boundary=np.zeros(size, dtype=bool),
                  jf_n_votes=np.zeros(size, dtype=int), jf_n_scales=np.zeros(size, dtype=int),
                  jf_solver_iterations=np.zeros(size, dtype=int), jf_solver_converged=np.full(size, None, dtype=object),
                  jf_n_reference=np.zeros(size, dtype=int), jf_method=np.full(size, config.method, dtype=object),
                  jf_status=np.full(size, "insufficient_history", dtype=object),
                  jf_reason=np.full(size, "insufficient_distinct_timestamp_support", dtype=object))
    return result


def _group_gaps(result, group, utc_times):
    # Input: rows=[0,1],wall timestamps Fri18:29NY/Mon08:00NY,
    # trading clock timestamps=[37740000000000,37800000000000]ns ->
    # Output: clock gap minutes=[NaN,1],wall gap minutes=[NaN,3691],boundary=[False,True].
    # Trick: original UTC age and compressed age are both retained; a closure does not prove an unchanged market level.
    # CORE LOGIC: STEP 1
    rows = group["row"].to_numpy(dtype=int)
    selected_gaps = group["stamp"].diff().to_numpy() / 60e9
    wall_gaps = pd.Series(utc_times[rows]).diff().to_numpy() / 60e9
    result["jf_gap_minutes"][rows] = selected_gaps
    result["jf_wall_gap_minutes"][rows] = wall_gaps
    result["jf_session_boundary"][rows] = np.isfinite(wall_gaps) & (wall_gaps > selected_gaps + 1e-9)


def filter_trades(frame, config=None, *, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC", session_schedule=None, backend="auto"):
    """Return a copy with jf_* audit fields, preserving every source row and index.

    Naive timestamps use ``timezone``; numeric epochs and ambiguous DST times
    are invalid. No field is interpreted as a true mid or known transaction fee.
    """
    # CONFIGURATION LOGIC: validate mappings, timezone and reserved output names.
    config = config or FilterConfig()
    if not isinstance(frame, pd.DataFrame) or not isinstance(config, FilterConfig):
        raise TypeError("frame must be a DataFrame and config a FilterConfig")
    mappings = (cusip_col, time_col, spread_col)
    if len(set(mappings)) != 3 or not frame.columns.is_unique:
        raise ValueError("three distinct mappings and unique column names required")
    missing = set(mappings) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if any(str(column).startswith("jf_") for column in frame.columns):
        raise ValueError("jf_ prefix is reserved; pass original source columns")
    ZoneInfo(timezone)
    if backend not in ("auto", "python", "numba"):
        raise ValueError("backend must be 'auto', 'python' or 'numba'")

    # Input: CUSIP=['001',None],time=['2026-01-01','bad'],spread=[100,inf] ->
    # Output: IDs=['001',<NA>],parsed=[2026-01-01 00:00 UTC,NaT],valid=[True,False].
    # Trick: IDs retain exact string values; NaN/inf, blank IDs and invalid times are quarantined.
    # CORE LOGIC: STEP 1
    identifiers = frame[cusip_col].astype("string")
    numeric = pd.to_numeric(frame[spread_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    parsed = _parse_times(frame[time_col], timezone)
    valid = identifiers.notna().to_numpy() & identifiers.str.strip().ne("").fillna(False).to_numpy(dtype=bool)
    valid &= parsed.notna().to_numpy() & np.isfinite(numeric)
    result = _empty_result(len(frame), config)
    result["jf_status"][~valid] = "invalid_input"
    result["jf_reason"][~valid] = "missing_id_invalid_time_or_nonfinite_spread"

    # Input: valid Friday18:29 and Monday08:00NY,configured trading sessions08–18:30 ->
    # Output: active=[True,True],clock gap=60 seconds; Saturday print is outside_session.
    # Trick: closures are removed only by the supplied calendar; no trades are interpolated into closed hours.
    # CORE LOGIC: STEP 2
    clock, active, calendar = trading_clock(parsed, config, session_schedule)
    outside = valid & ~active
    result["jf_status"][outside] = "outside_session"
    result["jf_reason"][outside] = "outside_declared_trading_intervals"
    valid &= active

    # Input: valid rows [(A,t=2,s=101,row=0),(A,t=1,s=100,row=2)] ->
    # Output: working rows [(A,t=1,s=100,row=2),(A,t=2,s=101,row=0)].
    # Trick: sorting is internal only; integer row IDs restore exact source order.
    # CORE LOGIC: STEP 3
    data = pd.DataFrame(dict(id=frame[cusip_col].to_numpy(), stamp=clock,
                             value=numeric, row=np.arange(len(frame))))
    data = data.loc[valid].sort_values(["id", "stamp", "row"], kind="stable")
    utc_times = parsed.astype("int64").to_numpy()
    accelerator = _resolve_backend(config.method, backend, len(data))

    # Input: A timestamps=[0,1,100],max_gap=10ns -> Output: segments [[0,1],[100]].
    # Trick: gaps reset both centered neighborhoods and causal state; CUSIPs never share references.
    # CORE LOGIC: STEP 4
    if len(data):
        if accelerator is None:
            _bulk_groups(result, data, utc_times, config, None)
        else:
            accelerator = _compiled_groups(result, data, utc_times, config, accelerator, backend)

    # Input: statuses=['ok','outlier','insufficient_history','solver_not_converged'],
    # flags=[False,True,False,False] -> Output: fit_eligible=[True,False,False,False].
    # Trick: unflagged abstentions or failed optimizations are not clean training labels.
    # CORE LOGIC: STEP 5
    result["jf_fit_eligible"] = (result["jf_status"] == "ok") & ~result["jf_is_outlier"]

    # OUTPUT ASSEMBLY LOGIC: publish positional fields and reproducible metadata.
    output = frame.copy(deep=True)
    for name, values in result.items():
        output[name] = values
    output["jf_time"] = parsed.array
    output["jf_clock_time"] = pd.array(clock, dtype="Int64")
    output.loc[~valid, "jf_clock_time"] = pd.NA
    output.attrs["jump_filter"] = dict(config=config.to_dict(), cusip_col=cusip_col,
                                      time_col=time_col, spread_col=spread_col, timezone=timezone,
                                      future_observations=config.method != "causal_ewma", calendar=calendar,
                                      backend="numba" if accelerator is not None else "python")
    return output


def summarize(annotated, *, cusip_col="CUSIP"):
    """Counts cover every source row; flagged rate uses only evaluated rows."""
    # Input: A statuses=['ok','outlier','invalid_input'],flags=[False,True,False] ->
    # Output: _evaluated=[True,True,False],_invalid=[False,False,True],
    # _insufficient=[False,False,False],_accepted=[True,False,False],one A group.
    # Trick: insufficient_history/invalid_input stay outside the evaluated denominator.
    # CORE LOGIC: STEP 1
    data = annotated.copy()
    data["_evaluated"] = data["jf_status"].isin(["ok", "outlier", "provisional_jump"])
    data["_invalid"] = data["jf_status"].eq("invalid_input")
    data["_insufficient"] = data["jf_status"].eq("insufficient_history")
    data["_accepted"] = data["_evaluated"] & ~data["jf_is_outlier"]
    data["_fit_eligible"] = data["jf_fit_eligible"]
    data["_failed"] = data["jf_status"].eq("solver_not_converged")
    data["_outside"] = data["jf_status"].eq("outside_session")
    data["_ambiguous"] = data["jf_status"].eq("ambiguous_transition")
    groups = data.groupby(cusip_col, dropna=False, sort=False)

    # Input: A rows=3,evaluated=2,flagged=1,accepted=1 ->
    # Output: flagged_rate=.5,coverage=2/3,accepted=1.
    # CORE LOGIC: STEP 2
    summary = groups.agg(rows=("jf_row_id", "size"), evaluated=("_evaluated", "sum"),
                         flagged=("jf_is_outlier", "sum"), accepted=("_accepted", "sum"),
                         invalid=("_invalid", "sum"), insufficient=("_insufficient", "sum"),
                         fit_eligible=("_fit_eligible", "sum"), solver_failed=("_failed", "sum"),
                         outside_session=("_outside", "sum"), ambiguous_transition=("_ambiguous", "sum"),
                         regime_candidates=("jf_regime_change", "sum"), median_score=("jf_score", "median"),
                         median_reference_count=("jf_n_reference", "median"))
    summary["flagged_rate"] = summary["flagged"].div(summary["evaluated"].replace(0, np.nan))
    summary["coverage"] = summary["evaluated"].div(summary["rows"])
    return summary.reset_index()


def select_fit_data(annotated, *, policy="hard", include_provisional=False):
    """Return fitting candidates with explicit ``jf_fit_weight``; never mutate input.

    Hard exclusion uses accepted rows. Soft screening also keeps evaluable
    anomalies with bounded influence weights. These weights are not inverse
    variances and do not solve identification of the economic mid.
    """
    # CONFIGURATION LOGIC: require engine annotations and an explicit downstream policy.
    if policy not in ("hard", "soft"):
        raise ValueError("policy must be 'hard' or 'soft'")
    if not {"jf_status", "jf_is_outlier", "jf_weight", "jf_fit_eligible"} <= set(annotated.columns):
        raise ValueError("pass an annotated frame from filter_trades")

    # Input: statuses=['ok','outlier','insufficient_history'],flags=[False,True,False],
    # weights=[1,.15,0],policy=soft -> Output: selected mask=[True,True,False],
    # fitting weights=[1,.15]. With policy=hard -> selected=[True,False,False],weights=[1].
    # Trick: future-dependent filtering is suitable for historical estimation; causal provisional jumps stay excluded by default.
    # CORE LOGIC: STEP 1
    states = ["ok", "outlier"] + (["provisional_jump"] if include_provisional else [])
    mask = annotated["jf_fit_eligible"] if policy == "hard" else annotated["jf_status"].isin(states) & annotated["jf_weight"].gt(0)
    selected = annotated.loc[mask].copy()
    selected["jf_fit_weight"] = 1.0 if policy == "hard" else selected["jf_weight"]
    return selected
