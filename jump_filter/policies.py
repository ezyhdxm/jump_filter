"""Optional fitting decisions, kept separate from the chosen algorithm's evidence."""

# SETUP LOGIC: policies reuse prepared arrays and never read external data.
from dataclasses import replace
import numpy as np
import pandas as pd


def _quantity_values(frame, quantity_col, multiplier):
    # Input: size=['500','bad',0,-1,1000],multiplier=1000 ->
    # Output: quantity=[500000,NaN,NaN,NaN,1000000],valid=[True,False,False,False,True].
    # Trick: unknown and nonpositive quantities cannot establish a trade's size.
    # CORE LOGIC: STEP 1
    if quantity_col is None:
        return np.full(len(frame), np.nan), np.zeros(len(frame), dtype=bool)
    values = pd.to_numeric(frame[quantity_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        quantity = values * multiplier
    valid = np.isfinite(quantity) & (quantity > 0)
    return np.where(valid, quantity, np.nan), valid


def _bracket_support(data, size, config):
    # Input: sorted ids=[A,A,A,A,B],stamps=[0,0,1,2,1]ns,rows=[0,1,2,3,4] ->
    # Output: cohort stamps=[0,1,2,1],cohort ids=[A,A,A,B],row cohorts=[0,0,1,2,3].
    # Trick: all same-time trades share one cohort and cannot bracket each other.
    # CORE LOGIC: STEP 1
    result = np.zeros(size, dtype=bool)
    if data.empty:
        return result
    stamps = data["stamp"].to_numpy(dtype=np.int64)
    ids = data["id"].to_numpy()
    starts = np.r_[True, (ids[1:] != ids[:-1]) | (stamps[1:] != stamps[:-1])]
    cohort_ids, times = ids[starts], stamps[starts]
    row_cohorts = np.cumsum(starts) - 1

    # Input: cohort ids=[A,A,A,B],times=[0,1,2,1]ns,horizon=max_gap=1s ->
    # Output: past=[False,True,True,False],future=[True,True,False,False].
    # Trick: nearest strict neighbors suffice for one-per-side; both must be inside the
    # same segment and horizon. window>=3 always allows at least one cohort per side.
    # CORE LOGIC: STEP 2
    limit = min(pd.Timedelta(config.horizon).value, pd.Timedelta(config.max_gap).value)
    same = cohort_ids[1:] == cohort_ids[:-1]
    close = same & (np.diff(times) <= limit)
    past, future = np.r_[False, close], np.r_[close, False]
    result[data["row"].to_numpy(dtype=int)] = (past & future)[row_cohorts]
    return result


def _reference_result(method, result, data, utc_times, config, backend):
    # SETUP LOGIC: a runtime import avoids coupling the engine's module initialization.
    from . import engine

    # Input: method=local_piecewise,one bond times=[0,1,2,3,4,5,6]ns,
    # spreads=[100,100,100,130,100,100,100],window=6,min_neighbors=6,horizon=1s ->
    # Output: target3 baseline=100,scale approximately2/9,cutoff=1,status='outlier'.
    # Trick: no policy can feed its removals back into the support reference.
    # CORE LOGIC: STEP 1
    if config.method == method:
        return result
    support_config = replace(config, method=method)
    support = engine._empty_result(len(utc_times), support_config)
    accelerator = engine._resolve_backend(method, backend, len(data))
    if accelerator is None:
        engine._bulk_groups(support, data, utc_times, support_config, None)
    else:
        engine._compiled_groups(support, data, utc_times, support_config, accelerator, backend)
    return support


def _support_evidence(result, data, utc_times, config, backend):
    # Input: 91 hourly spreads=100-4*abs(arange(91)-45),target45=100,
    # window31,min_neighbors6 -> Output: piecewise baseline100,scale approximately2/9;
    # its two-sided support preserves the real peak rather than averaging its shape away.
    # Trick: only insufficient side counts permit a bracketed full-line fallback;
    # disagreement between independent side projections always remains uncertain.
    # CORE LOGIC: STEP 1
    piecewise = _reference_result("local_piecewise", result, data, utc_times, config, backend)
    bracket = _bracket_support(data, len(utc_times), config)
    fallback = (piecewise["jf_status"] == "insufficient_history") & bracket
    linear = None
    if fallback.any():
        linear = _reference_result("local_linear", result, data, utc_times, config, backend)

    # Input: piecewise baselines=[100,NaN],statuses=['ok','insufficient_history'],
    # linear baselines=[98,101],fallback=[False,True] ->
    # Output: selected baseline=[100,101],support methods=['local_piecewise','local_linear_fallback'].
    # Trick: merging is positional and never changes the chosen method's original reference.
    # CORE LOGIC: STEP 2
    fields = ("jf_baseline", "jf_residual", "jf_threshold", "jf_n_reference", "jf_status", "jf_regime_change")
    support = {name: piecewise[name].copy() for name in fields}
    method = np.full(len(utc_times), "local_piecewise", dtype=object)
    if linear is not None:
        for name in fields:
            support[name][fallback] = linear[name][fallback]
        method[fallback] = "local_linear_fallback"

    # Input: support baseline=[100,NaN],residual=[12,NaN],threshold=[4.5,NaN],
    # units_per_bp=.01 -> Output: trend=[100,NaN],residual=[12,NaN],
    # threshold=[4.5,NaN],distance_bps=[1200,NaN].
    # Trick: all existing spread columns keep source units; only the new distance uses bp.
    # CORE LOGIC: STEP 3
    result["jf_support_trend"] = support["jf_baseline"].copy()
    result["jf_support_residual"] = support["jf_residual"].copy()
    result["jf_support_threshold"] = support["jf_threshold"].copy()
    result["jf_support_distance_bps"] = np.abs(support["jf_residual"]) / config.spread_units_per_bp
    evaluated = np.isin(support["jf_status"], ("ok", "outlier"))
    finite = np.isfinite(support["jf_baseline"]) & np.isfinite(support["jf_threshold"])
    sufficient = evaluated & finite & (support["jf_n_reference"] >= config.min_neighbors)

    # Input: sufficient=[True,True,False],bracket=[True,True,False],
    # transition=[False,True,False] -> Output: reliable=[True,False,False].
    # Trick: projected side disagreement protects jumps; the fallback line additionally
    # protects its raw regime changes. Neither source certifies a transition's mid.
    # CORE LOGIC: STEP 4
    transition = support["jf_regime_change"] | (result["jf_reason"] == "persistent_level_change_protected")
    reliable = sufficient & bracket & ~transition
    reason = np.full(len(utc_times), "insufficient_support", dtype=object)
    reason[(support["jf_n_reference"] >= config.min_neighbors) & ~bracket] = "no_past_and_future_bracket"
    reason[finite & bracket & transition] = "protected_transition"
    reason[reliable] = "reliable_two_sided_support"
    result["jf_support_reliable"], result["jf_support_reason"] = reliable, reason

    # Input: methods=['local_piecewise','local_linear_fallback'],finite=[True,False]
    # -> Output: published support methods=['local_piecewise','none'].
    # Trick: methods identify computed reference evidence even when a transition is protected.
    # CORE LOGIC: STEP 5
    method[~finite] = "none"
    result["jf_support_method"] = method


def _cap_exceeds(result, config):
    # Input: trend=100.00000000000003,value=90,units_per_bp=1,cap=10 ->
    # Output: distance approximately10.00000000000003bp,roundoff approximately
    # 1.14e-13bp,exceeds=False;value=89.9999999999 -> Output: exceeds=True.
    # Trick: eight source-value ULPs cover local-fit roundoff at the strict boundary;
    # the tolerance scales with source representation and the explicit bp conversion.
    # CORE LOGIC: STEP 1
    baseline = result["jf_support_trend"]
    value = baseline + result["jf_support_residual"]
    magnitude = np.maximum(np.abs(baseline), np.abs(value))
    magnitude = np.maximum(magnitude, config.max_deviation_bps * config.spread_units_per_bp)
    roundoff = 8 * np.spacing(magnitude) / config.spread_units_per_bp
    result["jf_support_distance_tolerance_bps"] = roundoff
    return result["jf_support_distance_bps"] > config.max_deviation_bps + roundoff


def apply_policies(result, frame, data, utc_times, config, quantity_col, backend):
    """Apply explicit decision rules after scoring, preserving original evidence."""
    # Input: algorithm status=['ok','insufficient_history'],flags=[False,False],
    # size=[1000000,500000] -> Output: copied algorithm status/flags unchanged,
    # quantity=[1000000,500000],quantity_valid=[True,True].
    # Trick: snapshots remain positional, so duplicate source index labels are harmless.
    # CORE LOGIC: STEP 1
    for name in ("status", "reason", "is_outlier", "fit_eligible", "weight"):
        result[f"jf_algorithm_{name}"] = result[f"jf_{name}"].copy()
    quantity, known = _quantity_values(frame, quantity_col, config.quantity_multiplier)
    result["jf_quantity"], result["jf_quantity_valid"] = quantity, known
    result["jf_quantity_notional"] = quantity.copy()
    _support_evidence(result, data, utc_times, config, backend)

    # Input: status=['ok','insufficient_history','provisional_jump'],original
    # residual=[12,12,12],cutoff=[5,5,5],n_reference=[6,6,0],min_neighbors=6,
    # support reliable=[True,False,False],support residual=[12,NaN,NaN],
    # support cutoff=[5,NaN,NaN] -> Output: weak=[False,True,True],
    # suspicious=[True,False,False].
    # Trick: an unreliable bridge cannot positively certify suspicion; a reliable
    # independent reference takes precedence over the chosen method's weak diagnostic.
    # CORE LOGIC: STEP 2
    status = result["jf_algorithm_status"]
    weak = np.isin(status, ("insufficient_history", "ambiguous_transition", "provisional_jump"))
    original_supported = result["jf_n_reference"] >= config.min_neighbors
    original_exceeds = np.isfinite(result["jf_threshold"]) & (np.abs(result["jf_residual"]) > result["jf_threshold"])
    support_exceeds = np.abs(result["jf_support_residual"]) > result["jf_support_threshold"]
    causal_fallback = (status == "provisional_jump") & ~result["jf_support_reliable"] & original_supported & original_exceeds
    suspicious = (result["jf_support_reliable"] & support_exceeds) | causal_fallback
    small = known & (quantity < config.quantity_threshold)

    # Input: reliable=[True,False,False],causal_fallback=[False,True,False] ->
    # Output: suspicion source=['two_sided_support_trend','causal_history','none'].
    # Trick: reliable trend evidence is recorded even when it clears a weak candidate;
    # only a supported causal provisional jump can use the history-only fallback.
    # CORE LOGIC: STEP 3
    suspicion_source = np.full(len(frame), "none", dtype=object)
    suspicion_source[result["jf_support_reliable"]] = "two_sided_support_trend"
    suspicion_source[causal_fallback] = "causal_history"
    result["jf_policy_suspicious_source"] = suspicion_source

    # Input: weak=[True,True,True],known=[True,True,False],small=[True,False,False],
    # suspicious=[True,True,True],distance=[10,10.01,20]bp,reliable=[True,True,False],
    # both rules on -> Output: quantity_reject=[True,False,False],
    # max_deviation=[False,True,False],retained_uncertain=[False,False,False].
    # Trick: both boundaries are strict: exactly 1MM is large and exactly 10bp stays.
    # Invalid/outside-session/failed-solver states cannot receive either override.
    # CORE LOGIC: STEP 4
    actionable = np.isin(status, ("ok", "outlier", "insufficient_history", "ambiguous_transition", "provisional_jump"))
    quantity_reject = config.quantity_rule & weak & small & suspicious
    deviation = config.max_deviation_rule & actionable & result["jf_support_reliable"] & _cap_exceeds(result, config)
    retained = config.quantity_rule & weak & known & ~quantity_reject & ~deviation
    rejected = quantity_reject | deviation

    # Input: quantity_reject=[True,False],deviation=[False,False],retained=[False,True]
    # -> Output: reasons=['small_quantity_suspicious','quantity_uncertain_retained'].
    # Trick: the absolute cap takes precedence when both optional rules reject a row.
    # CORE LOGIC: STEP 5
    reason = np.full(len(frame), "no_override", dtype=object)
    reason[config.quantity_rule & weak & ~known] = "quantity_unknown_no_override"
    reason[retained] = "quantity_uncertain_retained"
    reason[quantity_reject] = "small_quantity_suspicious"
    reason[deviation] = "support_trend_max_deviation_exceeded"
    result.update(jf_policy_small_quantity=small, jf_policy_suspicious=suspicious,
                  jf_policy_quantity_rejected=quantity_reject, jf_policy_max_deviation=deviation,
                  jf_policy_retained_uncertain=retained, jf_policy_reason=reason)

    # Input: algorithm statuses=['insufficient_history','provisional_jump','invalid_input'],
    # quantity_reject=[True,False,False],retained=[False,True,False] ->
    # Output: final statuses=['policy_outlier','retained_uncertain','invalid_input'],
    # flags=[True,False,False],fit_eligible=[False,True,False],weights=[0,1,0].
    # Trick: explicit retained uncertainty is a fitting choice, not a clean statistical
    # label. Mandatory policy exclusions also receive zero soft-fitting weight.
    # CORE LOGIC: STEP 6
    result["jf_status"][rejected], result["jf_status"][retained] = "policy_outlier", "retained_uncertain"
    result["jf_is_outlier"][rejected], result["jf_is_outlier"][retained] = True, False
    result["jf_fit_eligible"][rejected], result["jf_fit_eligible"][retained] = False, True
    result["jf_weight"][rejected], result["jf_weight"][retained] = 0.0, 1.0
    changed = rejected | retained
    result["jf_reason"][changed] = reason[changed]
