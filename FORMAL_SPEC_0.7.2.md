# Formal implementation contract — pydrlpdid 0.7.2

This file records the implementation choices that are fixed by the formal
estimator in the accompanying article.

## Common horizon stack

Every estimator exposed by `DRLPDID` is evaluated on the same
horizon-specific set of supported reference dates. A reference date is
retained only if it contains at least one eligible event and one eligible
clean control. The base period is `h=-1`. Composition is horizon specific and
is not silently replaced by a fixed-event population.

This contract does not redefine the literature benchmark
`LPDID(target_estimand="ra")`. The latter follows Dube et al.'s
regression-adjustment sample and may use control-only calendar cells when
estimating the clean-control outcome regression. The formal common-stack RA
member is `DRLPDID(estimation_method="ra")` and is labeled `DRLPDID-RA`.

## Absorbing adoption

An event is first treatment at the reference date. Controls are either
not-yet-treated through the horizon outcome or never treated, as selected by
`control_group`.

## Treatment switching

An event is an observed `0 -> 1` transition with an observed transition-free
history of length `L=stabilization_window`. Post-treatment eligibility requires
treatment to remain one through the reported horizon. Controls are local
all-stayers with no transition over the corresponding history and outcome
window. Stable-one controls have a causal interpretation under the
conditional-mean stabilization restriction stated in the article; the package
cannot test that identifying restriction.

The formal switching API uses a conservative observed-history sample: a unit
already treated in its first observed period is removed because its initial
spell is left-censored. This restriction is an implementation convention, not
an additional identification theorem.

## Nuisance adjustment

The public API supplies one list of predetermined covariates to both local
nuisance models. Calendar-time indicators are always included. No propensity
clipping, fallback estimator, or automatic change of the requested horizon
window is used.

For numerical identification, nuisance matrices are reduced to a deterministic
full-rank basis in original column order. Outcome-regression columns must be
identified among clean controls. The retained and discarded IPT columns are
recorded in the result metadata. If the retained outcome basis is nested in the
retained IPT balancing span, the IPT and DRLPDID-IPT point estimates must agree
within a numerical tolerance certified by the measured retained-basis
imbalance, the fitted outcome coefficients, and floating-point roundoff;
otherwise estimation stops.

## Inference

Analytic influence functions are aggregated by physical panel unit. Multiplier
inference uses unit-level Rademacher draws. The simultaneous family contains
only estimated horizons with finite, strictly positive standard errors; the
normalized base and unsupported horizons are excluded. A scalar `post_window`
is reported only if every prespecified horizon is supported.

The cluster sandwich follows the article without a parameter-count
degrees-of-freedom multiplier. Consequently, inference is invariant to
algebraically redundant nuisance blocks: when nested bases make IPT and
DRLPDID-IPT the same sample estimator, their influence-function standard
errors also agree up to numerical tolerance.

## Scope of double robustness

The double-robustness property concerns the local comparison-outcome regression
and local event propensity model, conditional on the causal identification
assumptions. It does not protect against anticipation, failure of conditional
parallel trends, invalid treatment histories, or failure of the stabilization
restriction.
