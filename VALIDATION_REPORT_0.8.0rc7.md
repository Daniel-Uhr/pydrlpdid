# Validation report — pydrlpdid 0.8.0rc7

## Scope

This release candidate adds a common, explicit aggregation layer to the
public `LPDID` and `DRLPDID` estimators. It does not change the horizon-level
estimating equations.

Available pre/post summaries are:

1. `pooled`: one joint pooled-outcome regression on the complete common
   window stack;
2. `equal_horizon`: equal average of separately estimated horizon effects;
3. `event_weighted`: horizon average weighted by eligible treated event rows;
4. `all`: return the three named summaries.

`mean_horizons` remains an alias for `equal_horizon`. `result.att` uses the
first requested post method and therefore remains pooled under defaults and
under `all`. All rows are exposed in `result.aggregates`; `result.pooled`
continues to return only the joint pooled regressions.

## Inferential contract

Linear aggregates use the declared horizon weights to combine the complete
cluster influence functions. This preserves cross-horizon covariance. The
analytic sandwich treats event counts and adoption timing as the realized
design. Paired cluster-bootstrap fits recompute the event counts and weights
inside each resample.

`event_weighted` is a common target-weighting option, not a method-equivalence
claim. Cross-estimator comparisons still require the same horizon window,
support, composition, within-horizon estimand, and population weights.
`LPDID(reweight=False)` remains variance weighted within horizon; use
`reweight=True` when the intended comparison is an equally weighted ATT.

## Automated validation

The complete suite was run with the source tree and the local `diff-diff`
3.12 oracle target on the import path:

```text
94 passed, 25 warnings
```

After the final pooled-view guard, the self-contained suite was rerun without
the temporary external oracle target: `61 passed, 1 skipped, 18 warnings`.
The one skip is the intentionally optional `diff-diff` parity module.

The warnings are expected design/convergence diagnostics, deliberately small
bootstrap smoke tests, and upstream `diff-diff` overlap warnings. Existing
event-path, pooled-regression, switch-in, absorbed-factor, PMD, CR1, and
coarser-cluster parity tests remain green.

New contract tests verify for both public estimators that:

- `all` returns exactly three pre and three post summaries;
- `result.pooled` remains a two-row compatibility view;
- `result.att` remains tied to the declared primary aggregation;
- stored event weights sum to one and are non-uniform under unequal support;
- the event-weighted estimate equals the manual weighted horizon average;
- the corresponding standard error is obtained from the combined cluster
  influence function;
- `mean_horizons` resolves to the canonical `equal_horizon` name.

## Wheel and Python 3.10 installation

Built artifact:

```text
dist/pydrlpdid-0.8.0rc7-py3-none-any.whl
```

SHA-256:

```text
8d22578fb9805423a80b01990b68bc8fd66cf04237edd87e89aec96b6a88f395
```

The wheel replaced rc6 in the local Python 3.10 environment
`pylpdid-env`. The independent installation check loaded the package from
`site-packages`, reported version `0.8.0rc7`, and confirmed that both public
constructors expose `pre_aggregation`.

## Remaining scientific qualification

This is the final planned public-API adjustment for the 0.8.0 release line.
It does not replace the still-recommended large outer Monte Carlo for coverage
of experimental paired cluster bootstrap procedures. Fixed post-treatment
composition is separately available for `LPDID(no_composition=True)`; an
analogous semiparametric fixed-composition score is not claimed for
`DRLPDID` without additional formal validation.
