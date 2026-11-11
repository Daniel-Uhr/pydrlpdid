# pydrlpdid 0.8.0rc3

`pydrlpdid` estimates local-projection Difference-in-Differences effects for
staggered absorbing adoption and sustained treatment switch-ins. The formal
API contains RA, IPW, IPT, DR-IPW, and DR-IPT estimators on one common
horizon-specific comparison stack.

## Install

Install the current release candidate directly from GitHub:

```bash
python -m pip install "git+https://github.com/Daniel-Uhr/pydrlpdid.git"
```

On Windows, `py -3.10` can be used to select Python 3.10 explicitly:

```powershell
py -3.10 -m pip install "git+https://github.com/Daniel-Uhr/pydrlpdid.git"
```

Alternatively, clone the project and install it from the repository folder:

```bash
git clone https://github.com/Daniel-Uhr/pydrlpdid.git
cd pydrlpdid
python -m pip install .
```

For an editable development installation with the test dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

`pydrlpdid` requires Python 3.9 or later. No machine-specific path is required.

## Minimal use

```python
from pydrlpdid import DRLPDID

result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=5,
    post_window=10,
    control_group="clean",
    cluster="id",
    inference="cluster",
).fit(
    df,
    outcome="y",
    unit="id",
    time="year",
    treatment="treated",
    covariates=["x1", "x2"],
)

print(result.summary())
event_study = result.event_study
pooled = result.pooled
att = result.att
p_value = result.p_value
```

When `pre_window` and `post_window` are omitted, all data-supported horizons
are evaluated. The default post-treatment scalar is the pooled ATT over
`h=0,...,post_window`; the pre row is the mean of supported placebo horizons,
excluding the normalized base `h=-1`. To request a different post-treatment
window or the mean of horizon-specific estimates, name it explicitly:

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=5,
    post_window=10,
    aggregate_window=(0, 5),
    post_aggregation="mean_horizons",  # or "pooled"
    inference="multiplier",
    seed=123,
).fit(
    df, outcome="y", unit="id", time="year",
    treatment="treated", covariates=["x1", "x2"],
)
```

`first_treat="first_treat"` remains accepted for panels that already contain
the adoption cohort (use 0 for never-treated units). Supplying both
`first_treat` and `treatment` triggers a consistency check.

## Sustained switch-ins

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=2,
    post_window=5,
    non_absorbing="effect_stabilization",
    stabilization_window=2,
    control_group="clean",
    history_policy="observed",
    inference="cluster",
).fit(
    df,
    outcome="y",
    unit="id",
    time="year",
    treatment="treatment",
    covariates=["x1", "x2"],
)
```

For the first-entry estimand:

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=2,
    post_window=5,
    non_absorbing="first_entry",
    control_group="clean",
    inference="cluster",
).fit(
    df, outcome="y", unit="id", time="year",
    treatment="treatment", covariates=["x1", "x2"],
)
```

An initial treated level is not an observed switch. Choose the treatment of
left-censored spells explicitly with `left_censoring="error"` (default),
`"drop"`, or `"keep_as_control_after_L"`.

## Advanced nuisance specifications

`covariates` supplies the same predetermined variables to both nuisance
models. For diagnostics or research designs that require distinct bases, use
`ps_covariates=` and `or_covariates=` in `fit`. Generated lagged outcome
changes can be requested with `outcome_change_lags=`, but any change that is
an algebraic component of a placebo outcome is omitted and recorded in
metadata.

## Estimators and inference

- `ra`: regression adjustment;
- `ipw`: logit inverse-probability weighting;
- `ipt`: inverse-probability tilting;
- `dr-ipw`: logit-based doubly robust estimation;
- `dr-ipt`: IPT/WLS doubly robust estimation.

`LPDID` remains the Dube et al. literature benchmark (`vw`, `rw`, and `ra`).
Its default RW weights are horizon-specific; `pre_weighting="post_target"` is
an explicit compatibility option.

The default analytic covariance uses `small_sample_correction="cr1"` and
Student `t` reference tails with `G-1` degrees of freedom, where `G` is the
realized number of clusters for the reported target. Set
`small_sample_correction="none"` for an uncorrected cluster sandwich. These
are the only two small-sample correction choices.

Inference options are:

- `cluster`: analytic cluster-robust inference (default);
- `multiplier`: pointwise inference plus simultaneous event-study bands;
- `cluster_bootstrap`: paired cluster resampling with full re-estimation.

Multiplier weights are selected with `bootstrap_weights="rademacher"`,
`"mammen"`, or `"webb"`. Stochastic procedures record their seed, requested
replications, and bootstrap diagnostics. Observation-level HC inference is
not exposed because it ignores within-unit serial dependence in panel data.
In `0.8.0rc3`, `cluster_bootstrap` is explicitly experimental; use at least
999 replications for reported work and inspect its success diagnostics.

The legacy `horizons=(-5, 10)` syntax and tuple-valued `post_window=(0, 10)`
remain available for replication code. New code should use `pre_window`,
integer `post_window`, and `aggregate_window`.

See `FORMAL_SPEC_0.8.0.md`, `MIGRATION_0.8.0.md`, and `RUN_REPLICATION.md`.

## License

MIT License.
