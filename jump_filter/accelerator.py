"""Optional compiled local screening, with bounded references and audit parity.

Only numerical arrays enter the compiled kernel. Trading-calendar validation,
original row order, timestamp-cohort medians and output publication remain in
the public engine. No fast-math or parallel reductions are enabled.
"""

# SETUP LOGIC: the optional compiler is imported only when this module is selected.
import numpy as np
from numba import njit
from numba.core.errors import NumbaError

# CONFIGURATION LOGIC: method and audit codes are private transport values.
LOCAL_METHODS = ("hampel", "local_linear", "jump_reversion", "local_piecewise", "consensus")
COMPILATION_ERROR = NumbaError
STATUS_NAMES = np.array(["insufficient_history", "ok", "outlier", "ambiguous_transition"], dtype=object)
REASON_NAMES = np.array([
    "insufficient_distinct_timestamp_support", "within_local_band",
    "reversion_confirmed_consensus", "hampel_threshold_exceeded",
    "local_linear_threshold_exceeded", "jump_reversion_threshold_exceeded",
    "needs_past_and_future_support", "persistent_level_change_protected",
    "within_method_band", "local_piecewise_residual_exceeded",
    "independent_side_predictions_disagree",
], dtype=object)


@njit(cache=True)
def _mad(values, center, floor):
    # Input: values=[100,100,101], center=100, floor=1 -> Output: 1.0.
    # Trick: retain the reference engine's 1.4826 MAD calibration and floor.
    # CORE LOGIC: STEP 1
    return max(1.4826 * np.median(np.abs(values - center)), floor)


@njit(cache=True)
def _weighted_line(x, values, weights):
    # Input: x=[-1,0,1],values=[100,101,102],weights=[1,1,1] ->
    # Output: total=3,x_mean=0,y_mean=101,variance=2,covariance=2.
    # Trick: centering before the weighted solve avoids cancellation in its determinant.
    # CORE LOGIC: STEP 1
    total = np.sum(weights)
    x_mean = np.sum(weights * x) / total
    y_mean = np.sum(weights * values) / total
    centered = x - x_mean
    variance = np.sum(weights * centered * centered)
    covariance = np.sum(weights * centered * (values - y_mean))

    # Input: variance=2,covariance=2,x_mean=0,y_mean=101 -> Output: [101,1].
    # Trick: nearly rank-deficient designs retain the original SVD least-squares solve.
    # CORE LOGIC: STEP 2
    if variance <= 1e-12 * total:
        design = np.column_stack((np.ones(len(x)), x))
        root = np.sqrt(weights)
        return np.linalg.lstsq(design * root.reshape((-1, 1)), values * root, rcond=-1.0)[0]
    slope = covariance / variance
    return np.array([y_mean - slope * x_mean, slope])


@njit(cache=True)
def _line(times, values, target, floor):
    # Input: times=[0,1,2],values=[100,101,102],target=1 ->
    # Output: x=[-1,0,1],weights=[.05,1,.05],beta=[101,0].
    # Trick: actual elapsed time and tricube weights are identical to the Python reference.
    # CORE LOGIC: STEP 1
    distance = max(np.max(np.abs(times - target)), 1)
    x = (times - target) / distance
    weights = np.maximum((1 - np.minimum(np.abs(x), 0.999) ** 3) ** 3, 0.05)
    beta = np.array([np.median(values), 0.0])

    # Input: x=[-1,0,1],values=[100,101,102],floor=1 ->
    # Output: beta approximately [101,1] after six Huber IRLS rounds.
    # Trick: no convergence shortcut is used; both backends perform the same six rounds.
    # CORE LOGIC: STEP 2
    for _ in range(6):
        residual = values - (beta[0] + beta[1] * x)
        sigma = _mad(residual, np.median(residual), floor)
        robust = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
        beta = _weighted_line(x, values, weights * robust)

    # Input: values=[100,101,102],fitted=[100,101,102],floor=1 ->
    # Output: reference approximately101,scale=1.
    # Trick: floating-point regression values may differ at rounding precision, not model definition.
    # CORE LOGIC: STEP 3
    residual = values - (beta[0] + beta[1] * x)
    return beta[0], _mad(residual, np.median(residual), floor)


@njit(cache=True)
def _local_reference(times, values, position, left_start, right_end, method, minimum, threshold, floor, tolerance, abs_floor):
    # Input: values=[100,100,130,100,100],position=2,left_start=0,right_end=5 ->
    # Output: left=[100,100],right=[100,100],refs=[100,100,100,100],median=100,scale=floor.
    # Trick: slices exclude the complete target timestamp cohort from every reference.
    # CORE LOGIC: STEP 1
    left = values[left_start:position]
    right = values[position + 1:right_end]
    refs = np.concatenate((left, right))
    median = np.median(refs)
    sigma = _mad(refs, median, floor)
    linear, linear_sigma = median, sigma
    if method == 1 or method == 4:
        reference_times = np.concatenate((times[left_start:position], times[position + 1:right_end]))
        linear, linear_sigma = _line(reference_times, refs, times[position], floor)

    # Input: left=[100,100],right=[100,100],minimum=4 ->
    # Output: side_ok=True,left_mid=100,right_mid=100,side_scale=floor.
    # Trick: a side needs max(2,min_neighbors//2) distinct timestamps, independently.
    # CORE LOGIC: STEP 2
    side_ok = min(len(left), len(right)) >= max(2, minimum // 2)
    left_mid = np.median(left) if len(left) else np.nan
    right_mid = np.median(right) if len(right) else np.nan
    deviations = np.concatenate((left - left_mid, right - right_mid))
    side_scale = _mad(deviations, 0.0, floor)
    left_projected, right_projected, trend_scale = left_mid, right_mid, side_scale

    # Input: left times=[0,1],values=[100,101],right times=[3,4],values=[103,104],target=2 ->
    # Output: projections approximately[102,102],trend_scale=floor.
    # Trick: each side extrapolates independently, retaining support for genuine turning points.
    # CORE LOGIC: STEP 3
    if side_ok and (method == 3 or method == 4):
        left_projected, left_sigma = _line(times[left_start:position], left, times[position], floor)
        right_projected, right_sigma = _line(times[position + 1:right_end], right, times[position], floor)
        trend_scale = max(left_sigma, right_sigma, linear_sigma if method == 4 else floor)
    trend_delta = abs(right_projected - left_projected)
    trend_reverted = side_ok and trend_delta <= max(abs_floor, tolerance * trend_scale)
    trend_shift = side_ok and trend_delta > max(abs_floor, threshold * trend_scale)

    # Input: raw/projected left=100,right=100,median/linear=100,all scales=2/9,
    # threshold=4.5,abs_floor=1,tolerance=2,side_ok=True -> Output:
    # (100,2/9,100,2/9,100,2/9,True,False,True,100,2/9,True,False,0,0).
    # Trick: returning primitive values keeps the compiler independent of Python audit dictionaries.
    # CORE LOGIC: STEP 4
    delta = abs(right_mid - left_mid)
    reverted = side_ok and delta <= max(abs_floor, tolerance * side_scale)
    shift = side_ok and delta > max(abs_floor, threshold * side_scale)
    bridge = (left_mid + right_mid) / 2 if side_ok else median
    trend_bridge = (left_projected + right_projected) / 2
    return median, sigma, linear, linear_sigma, bridge, side_scale, reverted, shift, side_ok, trend_bridge, trend_scale, trend_reverted, shift and trend_shift, delta, trend_delta


@njit(cache=True)
def _decision(value, refs, method, threshold, abs_floor):
    # Input: value=130,all centers=100,all scales=1,reverted=True,trend_reverted=True ->
    # Output: hampel=True,linear=True,reversal=True,trend_reversal=True.
    # Trick: use strict '>' at the visible cutoff, exactly as in the Python engine.
    # CORE LOGIC: STEP 1
    median, sigma, linear, linear_sigma, bridge, side_scale, reverted, shift, side_ok, trend_bridge, trend_scale, trend_reverted, consensus_shift = refs[:13]
    hampel = abs(value - median) > max(abs_floor, threshold * sigma)
    line_flag = abs(value - linear) > max(abs_floor, threshold * linear_sigma)
    reversal = reverted and abs(value - bridge) > max(abs_floor, threshold * side_scale)
    trend_reversal = trend_reverted and abs(value - trend_bridge) > max(abs_floor, threshold * trend_scale)
    center, scale, flag = median, sigma, hampel
    status, reason = 1, 1

    # Input: method=local_linear,line center=101,scale=1,line_flag=True ->
    # Output: center=101,scale=1,flag=True; method=jump_reversion uses raw bridge instead.
    # CORE LOGIC: STEP 2
    if method == 1:
        center, scale, flag = linear, linear_sigma, line_flag
    elif method == 2:
        center, scale, flag = bridge, side_scale, reversal
    elif method == 3:
        center, scale = trend_bridge, trend_scale
        flag = abs(value - center) > max(abs_floor, threshold * scale)
        reason = 8

    # Input: method=consensus,reversal=True,trend_reversal=False,bridge=100 ->
    # Output: center=100,scale=side_scale,flag=(hampel or line_flag).
    # Trick: the confirming raw or projected reference determines the displayed band per trade.
    # CORE LOGIC: STEP 3
    if method == 4:
        use_raw = reversal or (not trend_reversal and reverted)
        center = bridge if use_raw else trend_bridge
        scale = side_scale if use_raw else trend_scale
        flag = (reversal or trend_reversal) and (hampel or line_flag)
    if flag:
        status = 2
        reason = 2 if method == 4 else (9 if method == 3 else 3 + method)

    # Input: method=consensus,side_ok=False -> Output: status=0,reason=6,flag=False.
    # Input: method=local_piecewise,side_ok=True,trend_reverted=False ->
    # Output: status=3,reason=10,flag=False,regime=True.
    # Trick: abstention and protected transitions override anomaly flags, never silently admit fitting rows.
    # CORE LOGIC: STEP 4
    regime = consensus_shift if method == 4 else (False if method == 3 else shift)
    if (method == 2 or method == 4) and not side_ok:
        status, reason, flag = 0, 6, False
    elif (method == 2 or method == 4) and regime:
        status, reason, flag = 1, 7, False
    elif method == 3 and not trend_reverted:
        status, reason, flag, regime = 3, 10, False, True

    # Input: value=130,center=100,scale=1,threshold=4.5,abs_floor=1,
    # flag=True,regime=False,status=2,reason=2 ->
    # Output: (100,1,30,4.5,30,.15,1.0,0.0,2.0,2.0).
    # Trick: ambiguous and insufficient cases keep weight zero regardless of their numerical residual.
    # CORE LOGIC: STEP 5
    residual = value - center
    cutoff = max(abs_floor, threshold * scale)
    weight = min(1.0, cutoff / max(abs(residual), 1e-12)) if flag else 1.0
    if status == 0 or status == 3:
        weight = 0.0
    return center, scale, abs(residual) / scale, cutoff, residual, weight, 1.0 if flag else 0.0, 1.0 if regime else 0.0, float(status), float(reason)


@njit(cache=True)
def _needs_reference_retry(value, refs, method, threshold, abs_floor, tolerance):
    # Input: value=101,reference=100,cutoff=1 -> Output: True;
    # value=130,reference=100,cutoff=1 -> Output: False.
    # Trick: repair only near floating-point decision boundaries; never relax the strict anomaly cutoff.
    # CORE LOGIC: STEP 1
    rounding = 128 * np.finfo(np.float64).eps * max(1.0, abs(value), abs(refs[0]), abs(refs[2]), abs(refs[9]))
    centers = (refs[0], refs[2], refs[4], refs[9])
    scales = (refs[1], refs[3], refs[5], refs[10])
    for index in range(4):
        if abs(abs(value - centers[index]) - max(abs_floor, threshold * scales[index])) <= rounding:
            return True

    # Input: side delta=2,side scale=1,tolerance=2 -> Output: True;
    # side delta=0,side scale=1,tolerance=2,threshold=4.5,abs_floor=1 -> Output: False.
    # Trick: reversion and regime-protection comparisons also receive exact reference recomputation.
    # CORE LOGIC: STEP 2
    if method >= 2 and refs[8]:
        for delta, scale in ((refs[13], refs[5]), (refs[14], refs[10])):
            if abs(delta - max(abs_floor, tolerance * scale)) <= rounding:
                return True
            if abs(delta - max(abs_floor, threshold * scale)) <= rounding:
                return True
    return False


@njit(cache=True)
def local_kernel(times, values, starts, ends, row_cohorts, raw, method, window, horizon, minimum, threshold, abs_floor, tolerance):
    # Input: raw has3rows,cohorts=[0,1,2] -> Output: audit shape=(3,14),
    # baseline/scale/score/cutoff/residual/span/density=NaN,weight/support/flag/regime/status/reason=0.
    # Trick: fixed numeric transport avoids per-row Python dictionaries; publication restores named audit fields.
    # CORE LOGIC: STEP 1
    audit = np.zeros((len(raw), 14), dtype=np.float64)
    audit[:, :5] = np.nan
    audit[:, 11:13] = np.nan
    left_limit = window // 2
    right_limit = window - left_limit
    floor = abs_floor / threshold
    row_start = 0

    # Input: times=[0,10,20,30,40],position=2,window=4,horizon=15 ->
    # Output: left_start=1,right_end=4,support=2,row interval includes only target-cohort rows.
    # Trick: segment bounds prohibit cross-CUSIP/gap references; horizon endpoints are inclusive.
    # CORE LOGIC: STEP 2
    for position in range(len(times)):
        target = times[position]
        left_start = max(starts[position], position - left_limit)
        right_end = min(ends[position], position + right_limit + 1)
        while left_start < position and target - times[left_start] > horizon:
            left_start += 1
        while right_end > position + 1 and times[right_end - 1] - target > horizon:
            right_end -= 1
        support = right_end - left_start - 1
        row_end = row_start

        # Input: row_cohorts=[0,1,1,2],position=1,row_start=1 -> Output: row_end=3.
        # Trick: cohort IDs follow sorted source rows; every simultaneous trade retains its own value.
        # CORE LOGIC: STEP 3
        while row_end < len(raw) and row_cohorts[row_end] == position:
            row_end += 1
        audit[row_start:row_end, 6] = support
        if support > 1:
            first = left_start if left_start < position else position + 1
            last = right_end - 1 if right_end > position + 1 else position - 1
            span = (times[last] - times[first]) / 60e9
            audit[row_start:row_end, 11] = span
            audit[row_start:row_end, 12] = support * 60 / span if span > 0 else np.nan

        # Input: support=4,minimum=4,method=hampel,refs all100 ->
        # Output: references computed; support=3 leaves all rows insufficient with reference diagnostics retained.
        # CORE LOGIC: STEP 4
        if support >= minimum:
            refs = _local_reference(times, values, position, left_start, right_end, method, minimum, threshold, floor, tolerance, abs_floor)
            side_ok = refs[8]
            if method != 3 or side_ok:
                for row in range(row_start, row_end):
                    decision = _decision(raw[row], refs, method, threshold, abs_floor)
                    _write_decision(audit, row, decision)
                    audit[row, 13] = 1.0 if _needs_reference_retry(raw[row], refs, method, threshold, abs_floor, tolerance) else 0.0
        row_start = row_end
    return audit


@njit(cache=True)
def _write_decision(audit, row, decision):
    # Input: decision=[100,1,30,4.5,30,.15,True,False,2,2],row=0 ->
    # Output: audit row0 columns0..5=[100,1,30,4.5,30,.15],7..10=[1,0,2,2].
    # Trick: support at column6 and span/density at11/12 are published by the actual reference selector.
    # CORE LOGIC: STEP 1
    for field in range(6):
        audit[row, field] = decision[field]
    for field in range(6, 10):
        audit[row, field + 1] = decision[field]
