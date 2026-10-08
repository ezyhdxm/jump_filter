"""Retrospective IQR fences and convex Huber total-variation screening.

The trend fit is a joint, bounded-influence estimator, not a held-out prediction
or an identified mid price. The caller supplies sorted distinct-time cohort
medians and splits market gaps before fitting one segment.
"""

# SETUP LOGIC: imports have no file, display, or solver side effects.
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: exact normal-quantile calibration, in dimensionless units.
NORMAL_IQR = 1.3489795003921634
NORMAL_MAD = 1.482602218505602


def iqr_point(references, value, config):
    """Return midhinge, Gaussian-calibrated scale, score, raw cutoff, flag."""
    # Input: references=[98,99,100,101,102],value=110,iqr_multiplier=3,abs_floor=1 ->
    # Output: q1=99,q3=101,center=100,width=2,cutoff=7.
    # Trick: NumPy's linear sample quantiles are explicit; equality at a fence is retained.
    # CORE LOGIC: STEP 1
    q1, q3 = np.quantile(np.asarray(references, dtype=float), [0.25, 0.75], method="linear")
    center = float((q1 + q3) / 2)
    width = float(q3 - q1)
    effective_threshold = (config.iqr_multiplier + 0.5) * NORMAL_IQR
    cutoff = max(config.abs_floor, (config.iqr_multiplier + 0.5) * width)

    # Input: center=100,width=2,value=110,cutoff=7,effective_threshold=4.7214282514 ->
    # Output: scale approximately1.4826022185,score approximately6.744897502,flag=True.
    # Trick: the IQR multiplier controls flags; config.threshold is irrelevant for this method.
    # CORE LOGIC: STEP 2
    scale = max(width / NORMAL_IQR, config.abs_floor / effective_threshold)
    score = abs(float(value) - center) / scale
    return center, scale, score, float(cutoff), bool(abs(float(value) - center) > cutoff)


def _difference_adjoint(values, size):
    """Apply D transpose for (D b)[i] = b[i + 1] - b[i]."""
    # Input: values=[2,5],size=3 -> Output: [-2,-3,5].
    # Trick: zero padding supplies the two endpoint signs; size=1 has no differences.
    # CORE LOGIC: STEP 1
    padded = np.pad(np.asarray(values, dtype=float), (1, 1))
    return padded[:-1] - padded[1:]


def _factor_tv_system(size):
    """Factor the positive-definite tridiagonal matrix I + D transpose D."""
    # Input: size=3 -> Output: initial diagonal=[2,3,2].
    # Trick: endpoints have one neighbor; size=1 instead uses the identity matrix.
    # CORE LOGIC: STEP 1
    diagonal = np.full(size, 3.0)
    diagonal[[0, -1]] = 2.0
    if size == 1:
        diagonal[0] = 1.0

    # Input: diagonal=[2,3,2],off-diagonal=-1 -> Output: factors=[2,2.5,1.6].
    # Trick: Thomas elimination overwrites a private factor array, never the source signal.
    # CORE LOGIC: STEP 2
    for index in range(1, size):
        diagonal[index] -= 1.0 / diagonal[index - 1]
    return diagonal


def _solve_tv_system(factors, right_hand_side):
    """Solve (I + D transpose D) b = right_hand_side in linear storage."""
    # Input: factors=[2,2.5,1.6],right_hand_side=[1,0,1] ->
    # Output: forward vector=[1,0.5,1.2].
    # Trick: off-diagonal=-1 determines the plus sign in the forward substitution.
    # CORE LOGIC: STEP 1
    forward = np.asarray(right_hand_side, dtype=float).copy()
    for index in range(1, len(forward)):
        forward[index] += forward[index - 1] / factors[index - 1]

    # Input: factors=[2,2.5,1.6],forward=[1,0.5,1.2] ->
    # Output: solution=[0.75,0.5,0.75], up to floating-point precision.
    # Trick: backwards traversal reuses the factorization for every ADMM iteration.
    # CORE LOGIC: STEP 2
    solution = np.empty_like(forward)
    solution[-1] = forward[-1] / factors[-1]
    for index in range(len(forward) - 2, -1, -1):
        solution[index] = (forward[index] + solution[index + 1]) / factors[index]
    return solution


def _soft_threshold(values, cutoff):
    # Input: values=[-5,-1,0,1,5],cutoff=2 -> Output: [-3,0,0,0,3].
    # Trick: equality at the cutoff maps to zero; this is the exact l1 proximal operator.
    # CORE LOGIC: STEP 1
    return np.sign(values) * np.maximum(np.abs(values) - cutoff, 0.0)


def _huber_prox(values, delta, rho=1.0):
    # Input: values=[-8,-4,0,4,8],delta=2.5 -> Output: [-5.5,-2,0,2,5.5].
    # Trick: prox of Huber/rho changes branches at delta*(1+1/rho), continuously.
    # CORE LOGIC: STEP 1
    return np.where(np.abs(values) <= delta * (1 + 1 / rho),
                    values / (1 + 1 / rho), values - delta / rho * np.sign(values))


def _admm_update(signal, state, factors, delta, penalty, rho=1.0):
    """One two-block ADMM round, with residual and differences in one block."""
    # Input: signal=[0,10,0],state b=[0,0,0],r=[0,10,0],d=[0,0],u=[0,0,0],v=[0,0],
    # factors=[2,2.5,1.6],delta=2.5,penalty=8 -> Output: b=[0,0,0],r=[0,7.5,0],d=[0,0].
    # Trick: constraint signs are b+r=signal and Db-d=0; common rho cancels from b's solve.
    # CORE LOGIC: STEP 1
    _, residual, difference, residual_dual, difference_dual = state
    rhs = signal - residual - residual_dual + _difference_adjoint(difference - difference_dual, len(signal))
    baseline = _solve_tv_system(factors, rhs)
    next_residual = _huber_prox(signal - baseline - residual_dual, delta, rho)
    next_difference = _soft_threshold(np.diff(baseline) + difference_dual, penalty / rho)

    # Input: b=[0,0,0],r=[0,7.5,0],signal=[0,10,0],d=[0,0],u=[0,0,0],v=[0,0] ->
    # Output: u=[0,-2.5,0],v=[0,0],next state has the same b,r,d listed above.
    # Trick: joint r,d proximal updates are separable, so this remains two-block ADMM.
    # CORE LOGIC: STEP 2
    next_residual_dual = residual_dual + baseline + next_residual - signal
    next_difference_dual = difference_dual + np.diff(baseline) - next_difference
    return baseline, next_residual, next_difference, next_residual_dual, next_difference_dual


def _admm_diagnostics(signal, previous, state, tolerance, rho=1.0):
    """Primal and dual residual norms with absolute-plus-relative tolerances."""
    # Input: signal=[0,10,0],previous r=[0,10,0],d=[0,0],current b=[0,0,0],
    # r=[0,7.5,0],d=[0,0],u=[0,-2.5,0],v=[0,0] -> Output: primal=2.5,dual=2.5.
    # Trick: the dual residual is A transpose B times the joint proximal-block change.
    # CORE LOGIC: STEP 1
    baseline, residual, difference, residual_dual, difference_dual = state
    equality = baseline + residual - signal
    variation = np.diff(baseline) - difference
    primal = float(np.hypot(np.linalg.norm(equality), np.linalg.norm(variation)))
    dual_vector = residual - previous[1] - _difference_adjoint(difference - previous[2], len(signal))
    dual = float(rho * np.linalg.norm(dual_vector))

    # Input: same current state as step1,tolerance=0.00001 -> Output approximately:
    # primal_tolerance=0.00012236068,dual_tolerance=0.00004232051,converged=False.
    # Trick: tolerances live in normalized units and remain unchanged by spread-unit conversion.
    # CORE LOGIC: STEP 2
    norm_a = np.hypot(np.linalg.norm(baseline), np.linalg.norm(np.diff(baseline)))
    norm_b = np.hypot(np.linalg.norm(residual), np.linalg.norm(difference))
    primal_limit = tolerance * (np.sqrt(2 * len(signal) - 1) + max(norm_a, norm_b, np.linalg.norm(signal)))
    dual_limit = tolerance * (np.sqrt(len(signal)) + rho * np.linalg.norm(residual_dual + _difference_adjoint(difference_dual, len(signal))))
    return dict(primal_residual=primal, dual_residual=dual, primal_tolerance=float(primal_limit),
                dual_tolerance=float(dual_limit), converged=bool(primal <= primal_limit and dual <= dual_limit))


def _balance_admm_penalty(state, diagnostics, rho):
    # Input: rho=1,primal=1,dual=0.01,state u=[2],v=[] ->
    # Output: rho=2,next u=[1],next v=[],with b,r,d unchanged.
    # Trick: only early bounded updates are allowed; rescale duals to preserve their unscaled values.
    # CORE LOGIC: STEP 1
    primal, dual = diagnostics["primal_residual"], diagnostics["dual_residual"]
    ratio = 2.0 if primal > 10 * dual else 0.5 if dual > 10 * primal else 1.0
    next_rho = float(np.clip(rho * ratio, 0.125, 64.0))
    baseline, residual, difference, residual_dual, difference_dual = state
    multiplier = rho / next_rho
    return (baseline, residual, difference, residual_dual * multiplier, difference_dual * multiplier), next_rho


def _huber_tv(signal, penalty, delta, max_iter, tolerance):
    """Minimize sum Huber_delta(signal-b) + penalty * sum abs(diff(b))."""
    # Input: signal=[0,10,0] -> Output: factors=[2,2.5,1.6],
    # state b=[0,0,0],r=[0,10,0],d=[0,0],u=[0,0,0],v=[0,0],rho=4.
    # Trick: replicated-edge median3 is only initialization; the converged convex objective is unchanged.
    # CORE LOGIC: STEP 1
    factors = _factor_tv_system(len(signal))
    zero = np.zeros(len(signal), dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(signal, (1, 1), mode="edge"), 3)
    warm = np.median(windows, axis=1)
    state = (warm, signal - warm, np.diff(warm), zero.copy(), np.zeros(len(signal) - 1))
    rho = 4.0

    # Input: signal=[0,10,0],penalty=8,delta=2.5,max_iter=500,tolerance=1e-5 ->
    # Output: baseline approximately[1.25,1.25,1.25],converged=True (within stopping tolerance).
    # Trick: rho may balance at rounds25,50,75,100 only, then stays fixed; limits remain auditable.
    # CORE LOGIC: STEP 2
    for iteration in range(1, max_iter + 1):
        previous = state
        state = _admm_update(signal, previous, factors, delta, penalty, rho)
        diagnostics = _admm_diagnostics(signal, previous, state, tolerance, rho)
        if diagnostics["converged"]:
            break
        if iteration <= 100 and iteration % 25 == 0:
            state, rho = _balance_admm_penalty(state, diagnostics, rho)

    # Input: signal=[0,10,0],baseline=[1.25,1.25,1.25],delta=2.5,penalty=8 ->
    # Output: sparse_component=[0,6.25,0],objective=20.3125 (for the exact optimum).
    # Trick: evaluate the original feasible Huber objective, not the ADMM split-variable objective.
    # CORE LOGIC: STEP 3
    baseline = state[0]
    residual = signal - baseline
    sparse = _soft_threshold(residual, delta)
    loss = 0.5 * np.square(residual - sparse) + delta * np.abs(sparse)
    objective = float(np.sum(loss) + penalty * np.sum(np.abs(np.diff(baseline))))
    diagnostics.update(iterations=iteration, objective=objective, sparse_component=sparse, solver_rho=rho)
    return baseline, diagnostics


def _local_residual_scale(times, residuals, position, config):
    """Return held-out local residual MAD and distinct-cohort reference count."""
    # Input: times=[0,10,20,30,40],position=2,window=4,horizon=15ns ->
    # Output: neighbors=[1,3],count=2.
    # Trick: both elapsed horizon and distinct timestamp counts bound the window; target is excluded.
    # CORE LOGIC: STEP 1
    target = times[position]
    horizon = pd.Timedelta(config.horizon).value
    left = np.searchsorted(times, target - horizon, side="left")
    right = np.searchsorted(times, target + horizon, side="right")
    half = config.window // 2
    before = np.arange(max(left, position - half), position)
    after = np.arange(position + 1, min(right, position + config.window - half + 1))
    neighbors = np.concatenate((before, after))

    # Input: residuals=[0,0,30,0,0],neighbors=[1,3],abs_floor=1,threshold=4.5 ->
    # Output: scale=2/9,count=2; no neighbors also returns the floor and count=0.
    # Trick: centering residual MAD estimates scatter separately from TV baseline bias.
    # CORE LOGIC: STEP 2
    floor = config.abs_floor / config.threshold
    if len(neighbors) == 0:
        return floor, 0
    reference = residuals[neighbors]
    deviation = float(NORMAL_MAD * np.median(np.abs(reference - np.median(reference))))
    return max(deviation, floor), len(neighbors)


def robust_trend_segment(times, values, config):
    """Return baseline, local scales, and auditable joint-fit diagnostics.

    The normalized objective is 0.5||z-b-a||^2 + delta||a||_1 +
    lambda||D b||_1. D is the unweighted first difference over distinct
    timestamp cohorts, so it favors piecewise-constant spread paths. Time
    controls local scale support and caller-side gap splitting, not TV weight.
    """
    # INPUT VALIDATION LOGIC: accept only one sorted, finite, distinct-time segment.
    times = np.asarray(times, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError("times and values must be equally sized one-dimensional arrays")
    if not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError("values must be finite and times strictly increasing")
    if len(values) == 0:
        raise ValueError("robust_trend_segment requires at least one cohort")

    # Input: values=[100,100,100,100,130,100,100,100,100],abs_floor=1,threshold=4.5 ->
    # Output: location=100,noise=2/9,z=[0,0,0,0,135,0,0,0,0].
    # Trick: differenced MAD/sqrt2 estimates Gaussian level noise; trends/jumps can inflate it.
    # CORE LOGIC: STEP 1
    location = float(np.median(values))
    difference = np.diff(values)
    deviation = float(np.median(np.abs(difference - np.median(difference)))) if len(difference) else 0.0
    noise = max(NORMAL_MAD * deviation / np.sqrt(2.0), config.abs_floor / config.threshold)
    signal = (values - location) / noise

    # Input: z=[0,0,0,0,135,0,0,0,0],lambda=8,delta=2.5,max_iter=500,tolerance=1e-5 ->
    # Output: fitted z approximately[0.3125]*9,baseline approximately[100.0694444444]*9.
    # Trick: the target has bounded self-influence in this joint estimator; this is not leave-one-out.
    # CORE LOGIC: STEP 2
    normalized, diagnostics = _huber_tv(signal, config.trend_penalty, config.huber_delta,
                                       config.max_iter, config.tolerance)
    baseline = location + noise * normalized
    residuals = values - baseline

    # Input: times=hourly9 cohorts,values and baseline as step2,window=31,horizon=3D ->
    # Output: scales approximately[2/9]*9,n_reference=[8]*9,
    # sparse_component approximately[0,0,0,0,29.375,0,0,0,0] in input spread units.
    # Trick: objective stays dimensionless; returned baselines, scales, and sparse values use input units.
    # CORE LOGIC: STEP 3
    scale_and_count = [_local_residual_scale(times, residuals, index, config) for index in range(len(values))]
    scales = np.array([item[0] for item in scale_and_count], dtype=float)
    diagnostics["n_reference"] = np.array([item[1] for item in scale_and_count], dtype=int)
    diagnostics["normalization_scale"] = float(noise)
    diagnostics["normalization_location"] = location
    diagnostics["sparse_component"] = noise * diagnostics["sparse_component"]
    return baseline, scales, diagnostics
