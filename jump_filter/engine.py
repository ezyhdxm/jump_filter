"""CUSIP-isolated robust screening with explicit abstention and audit columns."""

# SETUP LOGIC: numerical imports; importing this module performs no I/O.
from collections import deque
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .config import FilterConfig


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
    horizon = pd.Timedelta(config.horizon).value
    left_bound = np.searchsorted(times, target - horizon, side="left")
    right_bound = np.searchsorted(times, target + horizon, side="right")
    left_limit = config.window // 2
    right_limit = config.window - left_limit
    left = np.arange(max(left_bound, position - left_limit), position)
    right = np.arange(position + 1, min(right_bound, position + right_limit + 1))
    return left, right, np.concatenate((left, right))


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
    linear, linear_scale = _linear_reference(times[neighbors], refs, times[position], floor)

    # Input: left=[100,100,100],right=[100,100,100], min_neighbors=6 ->
    # Output: side_ok=True, left_mid=100,right_mid=100,side_scale=2/9.
    # Trick: each side needs independent timestamps, so one late print cannot confirm reversion.
    # CORE LOGIC: STEP 2
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
    # CORE LOGIC: STEP 3
    left_projected, right_projected, trend_scale = left_mid, right_mid, side_scale
    if side_ok:
        left_projected, left_sigma = _linear_reference(times[left], values[left], times[position], floor)
        right_projected, right_sigma = _linear_reference(times[right], values[right], times[position], floor)
        trend_scale = max(left_sigma, right_sigma, linear_scale)
    trend_delta = abs(right_projected - left_projected)
    trend_reverted = bool(side_ok and trend_delta <= max(config.abs_floor, config.reversion_tolerance * trend_scale))
    trend_shift = bool(side_ok and trend_delta > max(config.abs_floor, config.threshold * trend_scale))
    trend_bridge = (left_projected + right_projected) / 2

    # Input: raw left=100,right=100,side_scale=1,trend_shift=False,threshold=4.5 ->
    # Output: consensus_shift=False.
    # Trick: median confirmation tolerates bursts; projected confirmation tolerates smooth trends.
    # CORE LOGIC: STEP 4
    consensus_shift = bool(side_ok and abs(right_mid - left_mid) > max(config.abs_floor, config.threshold * side_scale) and trend_shift)

    # Input: left_mid=100,right_mid=100,side_scale=2/9,tolerance=2,floor=1 ->
    # Output: reverted=True, shift=False, bridge=100, n_reference=6.
    # Trick: a persistent level difference is a regime candidate, never proof of a bad trade.
    # CORE LOGIC: STEP 5
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
    horizon = pd.Timedelta(config.horizon).value
    while history and (stamp - history[0][0] > horizon or len(history) > config.window):
        history.popleft()
    values = np.array([value for _, value in history], dtype=float)
    median_now = float(np.median(raw))
    support = len(values)
    floor = config.abs_floor / config.threshold

    # Input: history medians=[100,100],current cohort=[130],min_neighbors=6 ->
    # Output: state center=100, support=2,status='insufficient_history',weight=0.
    # Trick: unscored records are not silently treated as fit-eligible normal trades.
    # CORE LOGIC: STEP 2
    if support < config.min_neighbors:
        state.update(center=float(np.median(np.append(values, median_now))), run=0, sign=0)
        for row in rows:
            result["jf_n_reference"][row] = support
        history.append((stamp, median_now))
        return

    # Input: center=100,past=[100,100,100],current median=130,threshold=4.5,
    # floor=2/9,persistence=3,run=0 -> Output: run=1, sign=1,confirmed=False.
    # Trick: persistence is consecutive cohort innovations in one direction, not row count.
    # CORE LOGIC: STEP 3
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
    # CORE LOGIC: STEP 4
    if confirmed:
        center = median_now
        state.update(center=center, run=0, sign=0)
        history.clear()

    # Input: row spread=130,center=100,sigma=1,cutoff=4.5,confirmed=False ->
    # Output: flag=True,status='provisional_jump',score=30,weight=0.15.
    # CORE LOGIC: STEP 5
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
    # CORE LOGIC: STEP 6
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

    # Input: times=[0,1,2],cohort row IDs=[[0],[1],[2]],values=[100,100,100],
    # method=causal_ewma,min_neighbors=2,horizon=1s,threshold=4.5,abs_floor=1 ->
    # Output: row statuses=['insufficient_history','insufficient_history','ok'],
    # row reference counts=[0,1,2],row baselines=[NaN,NaN,100].
    # CORE LOGIC: STEP 2
    for position, (stamp, part) in enumerate(cohorts):
        rows = part["row"].to_numpy(dtype=int)
        raw = part["value"].to_numpy(dtype=float)
        if config.method == "causal_ewma":
            _causal_cohort(result, rows, raw, state, history, stamp, config)
        else:
            refs = _references(times, values, position, config)
            if refs is not None:
                _assign(result, rows, raw, refs, config)


def _empty_result(size, config):
    # SETUP LOGIC: separate validity, scoring, and flagging states explicitly.
    result = {name: np.full(size, np.nan) for name in (
        "jf_score", "jf_baseline", "jf_scale", "jf_residual", "jf_threshold")}
    result.update(jf_row_id=np.arange(size), jf_is_outlier=np.zeros(size, dtype=bool),
                  jf_regime_change=np.zeros(size, dtype=bool), jf_weight=np.zeros(size),
                  jf_n_reference=np.zeros(size, dtype=int), jf_method=np.full(size, config.method, dtype=object),
                  jf_status=np.full(size, "insufficient_history", dtype=object),
                  jf_reason=np.full(size, "insufficient_distinct_timestamp_support", dtype=object))
    return result


def filter_trades(frame, config=None, *, cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC"):
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

    # Input: CUSIP=['001',None],time=['2026-01-01','bad'],spread=[100,inf] ->
    # Output: IDs=['001',<NA>],parsed=[2026-01-01 00:00 UTC,NaT],valid=[True,False].
    # Trick: IDs retain exact string values; NaN/inf, blank IDs and invalid times are quarantined.
    # CORE LOGIC: STEP 1
    identifiers = frame[cusip_col].astype("string")
    numeric = pd.to_numeric(frame[spread_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    parsed = pd.Series([_parse_time(value, timezone) for value in frame[time_col]], dtype="datetime64[ns, UTC]")
    valid = identifiers.notna().to_numpy() & identifiers.str.strip().ne("").fillna(False).to_numpy(dtype=bool)
    valid &= parsed.notna().to_numpy() & np.isfinite(numeric)
    result = _empty_result(len(frame), config)
    result["jf_status"][~valid] = "invalid_input"
    result["jf_reason"][~valid] = "missing_id_invalid_time_or_nonfinite_spread"

    # Input: valid rows [(A,t=2,s=101,row=0),(A,t=1,s=100,row=2)] ->
    # Output: working rows [(A,t=1,s=100,row=2),(A,t=2,s=101,row=0)].
    # Trick: sorting is internal only; integer row IDs restore exact source order.
    # CORE LOGIC: STEP 2
    data = pd.DataFrame(dict(id=frame[cusip_col].to_numpy(), stamp=parsed.astype("int64").to_numpy(),
                             value=numeric, row=np.arange(len(frame))))
    data = data.loc[valid].sort_values(["id", "stamp", "row"], kind="stable")
    gap = pd.Timedelta(config.max_gap).value

    # Input: A timestamps=[0,1,100],max_gap=10ns -> Output: segments [[0,1],[100]].
    # Trick: gaps reset both centered neighborhoods and causal state; CUSIPs never share references.
    # CORE LOGIC: STEP 3
    for _, group in data.groupby("id", sort=False):
        segments = group["stamp"].diff().gt(gap).cumsum()
        for _, segment in group.groupby(segments, sort=False):
            _segment(result, segment, config)

    # OUTPUT ASSEMBLY LOGIC: publish positional fields and reproducible metadata.
    output = frame.copy(deep=True)
    for name, values in result.items():
        output[name] = values
    output["jf_time"] = parsed.array
    output.attrs["jump_filter"] = dict(config=config.to_dict(), cusip_col=cusip_col,
                                      time_col=time_col, spread_col=spread_col, timezone=timezone,
                                      future_observations=config.method != "causal_ewma")
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
    groups = data.groupby(cusip_col, dropna=False, sort=False)

    # Input: A rows=3,evaluated=2,flagged=1,accepted=1 ->
    # Output: flagged_rate=.5,coverage=2/3,accepted=1.
    # CORE LOGIC: STEP 2
    summary = groups.agg(rows=("jf_row_id", "size"), evaluated=("_evaluated", "sum"),
                         flagged=("jf_is_outlier", "sum"), accepted=("_accepted", "sum"),
                         invalid=("_invalid", "sum"), insufficient=("_insufficient", "sum"),
                         regime_candidates=("jf_regime_change", "sum"), median_score=("jf_score", "median"),
                         median_reference_count=("jf_n_reference", "median"))
    summary["flagged_rate"] = summary["flagged"].div(summary["evaluated"].replace(0, np.nan))
    summary["coverage"] = summary["evaluated"].div(summary["rows"])
    return summary.reset_index()
