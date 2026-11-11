# Validation report — pydrlpdid 0.8.0rc6

Date: 2026-11-11

## Scope

This release candidate aligns the public DRLPDID vocabulary with the public
LPDID benchmark without disguising their distinct estimating equations.
DRLPDID has no `reweight` flag: RA, IPW, IPT, DR-IPW, and DR-IPT target the
eligible-entry ATT on the supported semiparametric stack. Only DR-IPW and
DR-IPT have the double-robustness property.

## Cluster sandwich correction

`inference="cluster"` now has an explicit stacked M-estimation contract. For
each horizon or pooled scalar:

1. observation-level nuisance-and-ATT moments are evaluated at the fitted
   parameters;
2. scores are summed within the requested inference cluster;
3. the bread is the inverse Jacobian of the mean moment system;
4. the meat is the cross-product of cluster-summed scores divided by `n^2`;
5. the target covariance is the bread-meat-bread projection;
6. optional CR1 scales the variance by
   `G/(G-1) * (n-1)/(n-K)`, where `K` is the effective parameter count of the
   method actually fitted;
7. analytic p-values and confidence intervals use Student `t` with `G-1`
   degrees of freedom.

The earlier artificial common full-DR value of `K` was removed. Consequently,
nested IPT and DR-IPT retain identical point estimates and uncorrected
influence functions, while their method-specific CR1 standard errors may
differ because their fitted systems have different dimensions.

Compact sandwich diagnostics record the covariance type, uncorrected
variance, score count, cluster count, Jacobian rank, condition number, CR1
factor, and parameter count.

## Public API changes

- DRLPDID defaults: `pre_window=2`, `post_window=0`,
  `control_group="clean"`, `history_policy="diff_diff"`, and
  `left_censoring="keep_as_control_after_L"`.
- `pre_window=None, post_window=None` explicitly requests the full supported
  event-time grid.
- Added public `df_convention="cluster"` and
  `rank_deficient_action={"warn","error","silent"}`.
- Added fit vocabulary `ylags`, `dylags`, `absorb`, `only_event`, and
  `only_pooled`.
- Retained visible advanced nuisance overrides `ps_covariates` and
  `or_covariates`.
- Default pre and post scalar rows are fitted on complete pooled common
  samples. `post_aggregation="mean_horizons"` remains an explicit alternative.
- Generated placebo-overlapping outcome changes and target-period outcome
  levels are omitted and recorded.

Legacy `design`, `horizons`, `aggregate_window`, and
`outcome_change_lags` spellings remain explicit keyword-only arguments solely
to reproduce older notebooks. They are not hidden estimator modes.

## Tests

The complete suite passed with diff-diff 3.12 installed as an external oracle:

```
91 passed
```

The suite covers:

- exact LPDID point-estimate, standard-error, p-value, and pooled-scalar parity
  with diff-diff 3.12 across absorbing, first-entry, and effect-stabilization
  designs;
- VW, RW, and RA benchmark estimands;
- missing outcomes, PMD, no composition, custom pools, outcome lags, absorbed
  factors, coarser clustering, and rank deficiency;
- a direct manual bread-meat-bread identity for the cluster sandwich;
- exact RA sandwich agreement between the LPDID and DRLPDID implementations
  when they receive the same local stack and correction;
- method-specific CR1 scaling;
- cluster aggregation, multiplier weights, pooled inference, bootstrap
  diagnostics, nuisance-basis separation, and treatment-history rules;
- IPT/DR-IPT point and uncorrected influence-function identities under nested
  nuisance bases.

## Build and installation

Wheel:

`dist/pydrlpdid-0.8.0rc6-py3-none-any.whl`

SHA-256:

`b0a6a01b2b8ab32ab6f2dbb588df0e2404eab378f7f1ffa925b6882162bd2e8e`

The wheel was installed and verified in the local Python 3.10 environment
`pylpdid-env`. The installed package reports version `0.8.0rc6`, and the basic
LPDID/DRLPDID example executes successfully.

## Remaining scientific limits

- `cluster_bootstrap` remains experimental; it is not a substitute for a
  completed outer coverage Monte Carlo.
- PMD and no-composition are not claimed for DRLPDID until their
  semiparametric scores are separately formalized and validated.
- The literature-compatible treatment-history boundary convention is a
  default, not an identifying assumption. Empirical work should request
  `history_policy="observed"` when an unobserved initial state must not be
  converted into a transition.
- Equality of the RA sandwich on an identical stack does not imply equality of
  the public LPDID and DRLPDID-RA estimates when their certified support rules
  retain different calendar cells.
- No definitive Monte Carlo was run in this API-and-inference revision.
