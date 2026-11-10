# Formal implementation contract — pydrlpdid 0.8.0rc1

## Estimand and common support

At each horizon, RA, IPW, IPT, DR-IPW, and DR-IPT estimate an ATT on the same
supported local stack. A retained reference date must contain at least one
eligible event and one eligible clean control. The normalized base is `h=-1`.
Unsupported horizons are returned as `NaN`; support and composition are never
silently changed.

`post_aggregation="mean_horizons"` averages horizon-specific estimates and
their cluster influence functions. `post_aggregation="pooled"` estimates the
average long-difference outcome once on a common post-window stack. These are
different estimands when their samples differ, so no default is inferred.

## Treatment histories

For absorbing adoption, an event is first treatment at the reference date.
Controls are not-yet-treated through the outcome horizon or never treated,
according to `control_group`.

For switching, an event is an observed `0 -> 1` transition with `L` observed,
transition-free prior periods and treatment sustained through the requested
horizon. Controls are transition-free local stayers; stable-one controls rely
on the conditional-mean stabilization restriction in the article.

The default `history_policy="observed"` never converts an unobserved initial
state into a transition. A left-censored initially treated unit may be rejected,
dropped, or admitted to the stable-one control pool only after `L` observed
transition-free periods. `history_policy="diff_diff"` is a named compatibility
convention and is not the formal default.

## Nuisance models

The common interface `covariates=[...]` uses one predetermined basis for PS and
OR. Advanced users may supply distinct `ps_covariates` and `or_covariates`.
Calendar effects are included. The default OR has common slopes;
`calendar_interacted` is an explicit richer specification.

All nuisance matrices undergo deterministic original-order rank reduction.
OR columns must be identified among clean controls. No clipping or fallback
estimator is silently activated. Convergence, balance, propensity range,
control odds, effective control sample size, retained columns, and discarded
columns are recorded where applicable.

If the retained OR basis is nested in the retained IPT balancing span, IPT and
DR-IPT must agree within the tolerance implied by observed residual imbalance
and floating-point error. Failure of that certified identity stops estimation.

Generated lagged outcome changes are predetermined for post-treatment targets.
For placebo leads, any generated change that is an algebraic component of
`Y[t+h]-Y[t-1]` is excluded and recorded. A mechanically fitted placebo is not
treated as evidence for parallel trends.

## Inference

Influence functions are aggregated by physical panel unit. The default cluster
sandwich has no parameter-count correction, preserving invariance to redundant
moment representations. Optional `small_sample_correction="cr1"` scales the
complete cluster influence representation; pointwise, scalar, and multiplier
inference therefore remain coherent.

Multiplier inference uses unit-level Rademacher draws. Simultaneous bands
exclude the normalized base and unsupported horizons. Each scalar receives
its own marginal multiplier interval and p-value from its cluster influence
function.

Singular or excessively ill-conditioned stacked Jacobians are errors. Cluster
inference never falls back silently to a nonclustered covariance estimator.

## Scope of robustness

Double robustness concerns the local PS and OR nuisance models conditional on
the causal design. It does not repair anticipation, invalid treatment paths,
failure of conditional parallel trends, poor overlap, or failure of the
switching stabilization restriction. Placebo estimates diagnose restrictions;
they are not constrained to zero by implementation.
