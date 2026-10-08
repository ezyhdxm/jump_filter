# Optional fitting rules

These rules express an explicit fitting policy on top of a selected statistical method. Both switches start **off**. Enabling them preserves the original algorithm evidence and records any change to the fitting decision; a retained uncertain trade is not relabeled statistically clean.

## Quantity-aware uncertainty

Choose any source column with `quantity_col`. Normalize its value as

\[
Q_i=\operatorname{numeric}(\text{quantity}_i)\times\text{quantity_multiplier}.
\]

The default multiplier is 1 and the small-trade boundary is 1,000,000 in original amount units. Use multiplier 1,000 for columns reported in thousands, or 1,000,000 for columns reported in millions. A finite positive amount is required. Missing, unparseable, zero, negative, and infinite quantities do not imply a small or large trade and leave the quantity policy inactive for that row.

The uncertainty set contains the algorithm statuses `insufficient_history`, `ambiguous_transition`, and `provisional_jump`. With a reliable independent local trend, a row is suspicious only if its deviation exceeds that trend's statistical cutoff. This independent evidence takes precedence over an unstable diagnostic bridge from an unsupported algorithm row. Without a reliable trend, the only fallback is a finite, history-supported `causal_ewma` provisional-jump deviation exceeding its algorithm cutoff. An unsupported or ambiguous diagnostic bridge alone does not establish suspicion. Size alone is not suspicious evidence.

For an uncertain row with a known valid quantity:

\[
\text{quantity rejection}_i
=\mathbf1\{Q_i<1{,}000{,}000\}\,
 \mathbf1\{\text{suspicious}_i\}.
\]

Otherwise, retain the row with status `retained_uncertain`. This includes a large trade with inadequate statistical support, and a small trade without positive suspicious evidence. The original unsupported or ambiguous status remains available in the algorithm audit. A quantity exactly equal to the configured boundary is not small.

An algorithm-confirmed `outlier` remains rejected at any quantity. An originally accepted observation remains accepted by the quantity rule. Invalid inputs, outside-session records, and failed solvers are not rescued by size. A separately enabled maximum-distance rule can still reject a quantity-retained observation.

For example, default consensus receives nine minute observations with spreads `[100,100,60,100,100,100,100,100,100]`. The third trade has eight total reference cohorts but only two earlier ones, so consensus abstains. With quantity 500,000 it is suspicious and rejected by the quantity rule. With quantity 2,000,000 it is retained uncertain. If a reliable 100 bp support trend is available and the 10 bp distance rule is also enabled, the 40 bp deviation rejects either size.

## Maximum distance from a support trend

The reference is an independent leave-target-cohort-out **shape-aware local trend**, using the selected method's count, time, gap, and calendar controls. It first fits robust linear trends separately to the earlier and later neighborhoods. With enough references on each side and agreeing projected levels at the target, their average forms the support trend. This can follow a continuous up-then-down corner without treating its curvature as trading noise. Available fits from the selected method are reused. Other methods retain their own plotted statistical band; this additional trend is separately audited.

Each primary side fit requires `max(2, min_neighbors // 2)` distinct reference timestamps. The two predictions must agree within the local-piecewise tolerance; disagreement makes support unreliable and does **not** fall back to a line spanning the transition. Only when side counts are insufficient may a full-neighborhood robust line supply a fallback, requiring enough total references, at least one earlier and one later reference, finite estimates, and no detected level transition. Both constructions remain in the same gap-separated neighborhood. The engine does not invent a support trend across a long inactivity gap. These checks do not identify an economically observed mid price.

Let \(u\) be input spread units per basis point. Then

\[
d_i^{\rm bp}=\frac{|y_i-\widehat m_i^{\rm support}|}{u},
\qquad
\text{distance rejection}_i
=\mathbf1\{\text{reliable support}_i\}
 \mathbf1\{d_i^{\rm bp}>10\}.
\]

Defaults use `spread_units_per_bp=1` for observations already in bp. Percentage-point spreads require 0.01; decimal-rate spreads require 0.0001. Thus the same 10 bp boundary is 10, 0.10, or 0.001 respectively in those input units. A display label does not perform a conversion. Exactly 10 bp is retained by this rule; greater than 10 bp is rejected regardless of quantity, even when a wide statistical band accepts the observation.

Without a reliable reference, the distance rule abstains and the other decisions continue to apply. Both optional rules can use later trades through the centered support trend. Enabling them with `causal_ewma` therefore makes the **final policy decision** potentially noncausal while preserving the original causal algorithm audit.

### Why a retained point can appear outside the limit

The dashed limits are defined only where the trade has reliable support. A neighboring line segment cannot certify a point inside a support hole. Quantity screening may retain a sparse or transition trade even when its cap has **not been assessed**. A finite distance to an unreliable diagnostic bridge also does not constitute a passed cap check. For example, a genuine 100-to-130 bp level step can produce a 115 bp midpoint and a 15 bp diagnostic residual: rejecting it against that unsupported midpoint would remove legitimate repricing.

Every policy review now publishes `jf_policy_cap_assessed`, `jf_policy_cap_status` and `jf_policy_cap_reason`. Outcomes distinguish `disabled`, `not_actionable`, `no_reliable_support`, `within_limit` and `exceeded`. The charts distinguish uncertain trades **within the cap** from uncertain trades with an **unchecked cap**, and show the support reason in the hover. The dashboard and per-CUSIP summaries report assessed caps, cap abstentions, retained unchecked trades and strict verification exclusions separately.

### Require a verified cap check for fitting

Enable **Require a verified cap check for fitting** (`require_cap_support=True`) together with `max_deviation_rule=True` when your fitting sample must satisfy the distance criterion. This additional switch defaults off to preserve the original uncertainty policy. In strict mode, unverified support prevents fitting at every quantity; a detected cap violation still becomes a policy outlier. Formerly unflagged unverified records receive `unverified_support`, `jf_fit_eligible=False` and weight zero. A confirmed original algorithm outlier retains its flag and status, but also gets zero weight if its cap is unverified. Neither hard nor soft fitting includes these rows. The original evidence remains in `jf_algorithm_*`.

Writing \(H_i\) for the original/policy fitting choice, \(A_i\) for a reliable cap assessment and \(D\) for the configured bp limit, the optional verified sample obeys

\[
\mathcal D_{\rm verified}
=\{i:H_i=1,\ A_i=1,\ d_i^{\rm bp}\le D+\epsilon_i\},
\]

where \(\epsilon_i\) is the recorded allowance for eight source-value ULPs. Thus a retained fitting row must have a verified distance, rather than merely an absent violation flag. No trend is invented across a gap. Coverage can fall at segment endpoints, sparse/liquidity periods and protected transitions; these records remain auditable as unverified, without inferring bad execution.

## API and dashboard

```python
# SETUP LOGIC: The rules use the same public engine and explicit source mapping.
from jump_filter import FilterConfig, filter_trades

# CONFIGURATION LOGIC: Spreads are bp; quantity values are original amounts.
controls = FilterConfig(method="local_linear", time_basis="trading",
                        quantity_rule=True, quantity_threshold=1_000_000,
                        quantity_multiplier=1, max_deviation_rule=True,
                        max_deviation_bps=10, spread_units_per_bp=1)

# Input: df has nine A trades at NY times 2026-09-14 09:00..09:08,
# spread=[100,100,60,100,100,100,100,100,100] bp and trade_notional=2000000 for each.
# Output: all nine original rows remain; only position 2 is flagged and excluded from hard fitting.
# Trick: Large size does not rescue a supported algorithm outlier; the 40 bp support deviation also exceeds 10 bp.
# CORE LOGIC: STEP 1
review = filter_trades(df, controls, quantity_col="trade_notional")
```

The notebook and browser expose independent enable switches, a quantity-column selector, quantity-unit controls, and synchronized sliders with exact inputs for both boundaries. Changes remain pending until Apply. Method comparisons and exports use the applied quantity mapping and settings.

For a fitting sample with verified cap coverage, add `require_cap_support=True` to the configuration above. With both distance controls enabled, every returned hard-fitting row has `jf_policy_cap_assessed=True` and a distance at or below the configured limit, allowing the recorded roundoff tolerance.

`jf_fit_eligible` is the final hard-fitting decision. Original algorithm status and final retained/excluded counts are reported separately. Explicit uncertain retention can improve fitting coverage, but must not increase the reported fraction of statistically evaluated trades. Use the policy reason and support diagnostics to audit each override.

## Why the shaded band can break

The plotted band is the selected method's statistical deviation boundary around its diagnostic reference. It deliberately stops at declared session closures, long drawing gaps, and missing numerical references. A gap does not imply that the engine deleted trades or that closed time counted as active trading time.

The earlier renderer used Plotly's `tonexty` fill between interrupted upper and lower lines. Plotly closes interrupted fill paths by connecting their endpoints, which can create misleading shading across a break. Each valid reference run is now a separately closed polygon; missing-reference holes remain visible even when chart sampling omits the unsupported trade. See the [Plotly scatter fill reference](https://plotly.com/python/reference/scatter/#scatter-fill).

The band is not a confidence interval for a latent mid and does not replace the optional quantity or distance policy. A policy can reject an observation that lies inside the selected method's band; its exact reason remains visible in the hover and exported audit.
