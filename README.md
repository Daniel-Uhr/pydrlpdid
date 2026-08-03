# pydrlpdid 0.7.2

`pydrlpdid` implements the local semiparametric estimators developed in
*Doubly Robust Local Projections Difference-in-Differences: Staggered
Adoption and Treatment Switching*.

Version 0.7.2 deliberately exposes a small formal API. The five estimator
identities in `DRLPDID` use the same horizon-specific supported event stack:

- `ra`: regression adjustment;
- `ipw`: logit inverse-probability weighting;
- `ipt`: inverse-probability tilting;
- `dr-ipw`: logit-based doubly robust estimation;
- `dr-ipt`: improved IPT--WLS doubly robust estimation.

Only `dr-ipw` and `dr-ipt` carry the double-robustness property: consistency
requires a correct local comparison-outcome regression or a correct local
treatment-entry propensity model, conditional on the identifying assumptions.

## Installation

From the directory containing the wheel:

```bash
python -m pip install --upgrade pydrlpdid-0.7.2-py3-none-any.whl
```

For an editable installation from this source tree:

```bash
python -m pip install -e .
```

## Absorbing staggered adoption

```python
from pydrlpdid import DRLPDID

result = DRLPDID(
    estimation_method="dr-ipt",
    design="absorbing",
    control_group="not_yet_treated",
    horizons=(-10, 19),
    post_window=(0, 19),
    inference="multiplier",
    n_bootstrap=999,
    seed=123,
).fit(
    data=df,
    outcome="asmrs",
    unit="state",
    time="year",
    first_treat="first_treat",
    covariates=["pcinc", "asmrh", "cases"],
)
```

## Sustained switch-ins

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    design="switching",
    stabilization_window=20,
    horizons=(-10, 15),
    post_window=(0, 10),
    inference="multiplier",
    n_bootstrap=999,
    seed=123,
).fit(
    data=df,
    outcome="growth",
    unit="country",
    time="year",
    treatment="democracy",
    covariates=["lag1y", "lag2y", "lag3y", "lag4y"],
)
```

The switching design is fixed to observed `0 -> 1` events with an
L-period transition-free history, treatment sustained through each reported
post-event horizon, and transition-free local stayers as controls. Stable-one
stayers are admitted under the paper's effect-stabilization restriction.
Units already treated in their first observed period are removed because the
initial spell is left-censored; this conservative implementation convention is
recorded in the formal contract.

## Inference and output

`result.event_study` reports analytic panel-unit-cluster-robust standard
errors and pointwise confidence intervals:

```text
horizon, estimate, se, t_stat, p_value, ci_lower, ci_upper
```

With `inference="multiplier"`, it additionally reports multiplier pointwise
intervals in separate columns and simultaneous sup-t bands:

```text
multiplier_ci_lower, multiplier_ci_upper, multiplier_p_value,
sim_ci_lower, sim_ci_upper
```

The analytic cluster-robust interval is never overwritten. The base horizon
`h=-1` is normalized to zero.

The cluster sandwich follows the influence-function formula in the article
without a parameter-count degrees-of-freedom multiplier. Consequently, when
nested bases make IPT and DRLPDID-IPT the same sample estimator, their analytic
standard errors also coincide up to numerical tolerance.

If `post_window=(lo, hi)` is supplied, `result.scalars` contains the
equal-weight average over exactly those horizons and its joint
cluster-robust standard error. The package does not silently replace a
missing requested horizon with a shorter window.

## Formal safeguards

- all estimator identities use the same supported reference-date stack;
- propensity odds are untrimmed and no overlap regularization is hidden;
- IPT acceptance requires the balancing moments, not only an optimizer flag;
- nuisance designs use a deterministic full-rank basis, recorded in diagnostics;
- the IPT--DRIPT identity is checked when nuisance bases are nested;
- nuisance failures do not trigger a different estimator;
- the event population follows the as-observed horizon-specific composition.

`LPDID` remains available solely as the literature benchmark API. In
particular, `LPDID(target_estimand="ra")` is the regression-adjustment
estimator of Dube et al. and retains its native clean-control regression
sample, including control-only calendar cells. It is distinct from
`DRLPDID(estimation_method="ra")`, which is labeled `DRLPDID-RA` and uses the
formal common supported stack.

`LPDID` also accepts `inference="multiplier"` so that pointwise
cluster-robust intervals and simultaneous bands for the literature RA
benchmark are produced from its own cluster influence functions.

## Article replication files

- `examples/monte_carlo_absorbing_N500.py`: absorbing-adoption Monte Carlo
  with `N=500` and 500 replications by default;
- `notebooks/Application_1_Bacon_pydrlpdid_v0.7.2_LOCAL_H19.ipynb`:
  complete no-fault-divorce application;
- `notebooks/Application_2_Dube_and_Formal_Switching_pydrlpdid_v0.7.2.ipynb`:
  complete Dube-compatible and formal sustained-switch-in application;
- `notebooks/Monte_Carlo_DRLPDID_pydrlpdid_v0.7.2_FINAL.ipynb`:
  complete article Monte Carlo with atomic, version-signed checkpoints.

The notebooks must be run from the 0.7.2 source root. They write all tables,
figures, diagnostics, and manifests to application-specific output folders.
