# Formal implementation contract — pydrlpdid 0.8.1

## Public estimator surfaces

`LPDID` reproduces the constructor, fit vocabulary, samples, point estimates,
and analytic standard-error conventions of `diff-diff` 3.12 for the certified
features and is the package's only public literature-benchmark API. Its
defaults are `pre_window=2`, `post_window=0`,
`control_group="clean"`, `reweight=False`, `history_policy="diff_diff"`, and
`left_censoring="keep_as_control_after_L"`. `DRLPDID` uses the same public
window and design vocabulary, including defaults `pre_window=2`,
`post_window=0`, `control_group="clean"`, `history_policy="diff_diff"`, and
`left_censoring="keep_as_control_after_L"`. Passing both windows as `None`
requests the full supported grid. It retains the five RA/IPW/IPT/DR-IPW/DR-IPT
estimators and does not reinterpret the binary `reweight` flag as a
semiparametric estimator choice.

`LPDID` supports `no_composition`, PMD, outcome-level and
outcome-change lags, absorbed factors, custom pooled pre/post windows, and
`rank_deficient_action`. `DRLPDID` supports the common `ylags`, `dylags`,
`absorb`, `rank_deficient_action`, and `df_convention="cluster"` vocabulary,
but it does not silently claim PMD or no-composition support where the
semiparametric score has not been separately certified. Survey-design
inference is not certified in 0.8.1 and
raises an explicit `NotImplementedError`.

## Estimand and common support

At each horizon, RA, IPW, IPT, DR-IPW, and DR-IPT evaluate the ATT score on the
same supported target stack. A retained reference date must contain at least
one eligible event and one eligible clean control. The normalized base is
`h=-1`.

`unsupported_event_policy="drop"` estimates the supported-event ATT and records
the requested, retained, and excluded event rows and dates. Setting it to
`"error"` fails closed when any requested event lacks a clean control. The
package never treats a generalized inverse or a model prediction as empirical
overlap.

Target support and outcome-regression training support are distinct. The Dube
et al. RA convention uses `or_training_support="all_clean_controls"`: controls
from dates without an event may identify a common covariate slope, while the
ATT remains evaluated on supported event dates. The strict alternative
`"event_dates"` trains the OR only on the target stack. Propensity-score and DR
scores remain on event dates. The enlarged OR experiment is not yet certified
for DR-IPW or DR-IPT, so those methods reject an explicit
`"all_clean_controls"` request.

The public event-time surface uses `pre_window` and integer `post_window`.
When either is `None`, its largest data-supported value is inferred. The
legacy `horizons=(min_h,max_h)` spelling remains a compatibility alias and
cannot be combined with the new window arguments.

Both estimators expose `pre_aggregation` and `post_aggregation`. Each accepts
`pooled`, `equal_horizon`, `event_weighted`, `all`, or a sequence of those
names. The legacy name `mean_horizons` maps to `equal_horizon`.

`pooled` estimates the average long-difference outcome once on a complete
common pre/post stack, following the pooled-regression construction in Dube et
al. `equal_horizon` forms

\[
\widehat\theta_{\mathrm{EH}}=|\mathcal H|^{-1}
\sum_{h\in\mathcal H}\widehat\theta_h.
\]

`event_weighted` forms

\[
\widehat\theta_{\mathrm{EW}}=
\sum_{h\in\mathcal H}\frac{N_{1h}}{\sum_{j\in\mathcal H}N_{1j}}
\widehat\theta_h,
\]

where `N_1h` is the number of eligible treated event rows supporting horizon
`h`. This is the package's cross-estimator comparison target for methods whose
overall ATT equally weights supported treated event-horizon observations. It
does not imply equality across methods with different samples, support rules,
or within-horizon estimands. In particular, `LPDID(reweight=False)` retains
the variance-weighted horizon estimand; cross-method ATT comparison normally
requires `reweight=True`. These aggregation rules do not by themselves fix
cohort composition across horizons. `LPDID(no_composition=True)` is the
separate fixed-post-composition option; the analogous semiparametric score is
not yet certified for `DRLPDID`.

For both linear summaries, the same weights combine the horizon-specific
cluster influence functions; cross-horizon covariance is therefore retained.
Analytic inference conditions on the realized treatment design. The paired
cluster bootstrap recomputes effects, event counts, and weights in each
resample. Every result records `aggregation`, `horizons`, `weights`, and
`weight_basis`. `result.aggregates` returns all requested rows;
`result.aggregate(window, aggregation)` selects one. `result.pooled` remains
the pooled-only compatibility table, and `result.att` uses the first requested
post method (`pooled` by default and under `all`).

## Treatment histories

For absorbing adoption, an event is first treatment at the reference date.
Controls are not-yet-treated through the outcome horizon or never treated,
according to `control_group`.

For switching, an event is an observed `0 -> 1` transition with `L` observed,
transition-free prior periods and treatment sustained through the requested
horizon. Controls are transition-free local stayers; stable-one controls rely
on the conditional-mean stabilization restriction in the article.

`non_absorbing="first_entry"` implements the first-entry target, while
`non_absorbing="effect_stabilization"` requires `stabilization_window=L`.

The public compatibility default `history_policy="diff_diff"` follows the
literature API. The explicit strict policy `history_policy="observed"` never
converts an unobserved initial state into a transition. Under that policy, a
left-censored initially treated unit may be rejected, dropped, or admitted to
the stable-one control pool only after `L` observed transition-free periods.
The selected boundary convention is always recorded.

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
DR-IPT point estimates and uncorrected influence functions must agree within
the tolerance implied by observed residual imbalance and floating-point error.
Failure of the point-estimate identity stops estimation. Method-specific CR1
standard errors can differ because IPT and DR-IPT estimate systems with
different effective parameter counts; this finite-sample convention does not
alter their common uncorrected influence function.

Generated lagged outcomes and changes are predetermined for post-treatment
targets. For placebo leads, any generated change or target-period outcome
level that is an algebraic component of `Y[t+h]-Y[t-1]` is excluded and
recorded. A mechanically fitted placebo is not treated as evidence for
parallel trends. `LPDID` retains the `diff-diff` outcome-change convention;
its mechanically zero placebo is explicitly a convention, not a diagnostic
result. `DRLPDID(method="ra")` retains the same benchmark convention;
propensity and doubly robust methods omit and record overlapping controls.

First-entry treated events must remain treated at every date from entry through
the requested post horizon. First-entry controls must not have entered treatment
at any date through that horizon. Pooled pre and post regressions require every
target outcome in the requested window; partial unit-specific averages are not
permitted.

## Inference

Influence functions are first formed at the physical panel-unit level and may
then be summed within a user-supplied coarser cluster. A coarser cluster is
valid only when every panel unit belongs to exactly one cluster over the full
panel; missing cluster labels or units that move between clusters are rejected.
For DRLPDID, the default is `small_sample_correction="cr1"`; the only
alternative is `"none"`. The covariance is the stacked M-estimation sandwich:
the bread is the inverse Jacobian of the complete fitted moment system, and the
meat is the cross-product of cluster-summed empirical scores. CR1 uses the
effective parameter count of the method actually fitted, exactly as the
benchmark engine does for its corresponding regression or stacked-RA system.
Compact Jacobian, score, cluster, variance, and parameter-count diagnostics are
recorded. Analytic pointwise and scalar inference uses
Student `t` tails with `G-1` degrees of freedom, where `G` is the realized
cluster count for the target. The normalized base period has no standard
error, p-value, or confidence interval.

Multiplier inference uses unit-level mean-zero, unit-variance draws. Simultaneous bands
exclude the normalized base and unsupported horizons. Each scalar receives
its own marginal multiplier interval and p-value from its cluster influence
function.

The multiplier distribution is user-selectable as Rademacher, Mammen, or
Webb; no data-dependent automatic switch is made. Paired cluster bootstrap is
a separate full-reestimation procedure. It records requested and successful
draws and stops when the prespecified minimum success rate is not met. It is
experimental in `0.8.1`; fewer than 99 draws are accepted only for smoke
testing, and at least 999 are recommended for reported percentile intervals.

The certified cluster is the panel unit, a one-to-one relabeling, or a nested
coarser grouping. Direct coarser-cluster inference is defined by summing the
unit influence functions within the grouping before computing covariance,
multiplier inference, or paired cluster resampling. Paired resampling relabels
panel units when a coarse cluster is drawn more than once.
Observation-level HC inference is not part of the public panel API.

Singular or excessively ill-conditioned stacked Jacobians are errors. Cluster
inference never falls back silently to a nonclustered covariance estimator.

## Scope of robustness

Double robustness concerns the local PS and OR nuisance models conditional on
the causal design. It does not repair anticipation, invalid treatment paths,
failure of conditional parallel trends, poor overlap, or failure of the
switching stabilization restriction. Placebo estimates diagnose restrictions;
they are not constrained to zero by implementation.
