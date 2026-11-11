# pydrlpdid 0.8.1

`pydrlpdid` estimates local-projection Difference-in-Differences effects for
staggered absorbing adoption and sustained treatment switch-ins. The formal
API contains RA, IPW, IPT, DR-IPW, and DR-IPT estimators on one common
horizon-specific comparison stack.

## Install

Install the current stable version directly from GitHub:

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

### Literature LP-DiD

`LPDID` is the package's single public implementation of the literature
benchmark and follows the functional pattern of `diff-diff.LPDiD`:

```python
from pydrlpdid import LPDID

result = LPDID(
    pre_window=5,
    post_window=10,
    control_group="clean",
    reweight=True,
    cluster="id",
).fit(
    df,
    outcome="y",
    unit="id",
    time="year",
    treatment="treated",
    covariates=["x1", "x2"],
)

print(result.summary())
event_study = result.to_dataframe()
pooled_pre_post = result.pooled
att = result.att
```

`LPDID` uses the same defaults as `diff-diff` 3.12:
`pre_window=2`, `post_window=0`, `control_group="clean"`, and
`reweight=False`. With `reweight=False`, the target is the variance-weighted
LP-DiD coefficient. With `reweight=True` and no controls it is the reweighted
ATT; with covariates, `ylags`, `dylags`, or `absorb`, it is regression
adjustment. `LPDID` also supports `no_composition`, `pmd`, custom
`pre_pooled`/`post_pooled` windows, `non_absorbing="first_entry"`, and
`non_absorbing="effect_stabilization"`.

`rank_deficient_action` is `"warn"`, `"error"`, or `"silent"`; every removed
column remains recorded in result metadata. `survey_design` is not yet
certified and raises explicitly. Following `diff-diff`, `LPDID` retains
overlapping outcome-change lags, including a mechanically zero
placebo when the regressor is the negative of that placebo outcome. This is
reported as a design convention and must not be interpreted as an empirical
pre-trend result. The RA method inside `DRLPDID` retains the same benchmark
convention. Propensity and doubly robust methods omit and report the overlap
instead.

### Doubly robust LP-DiD

```python
from pydrlpdid import DRLPDID

result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=5,
    post_window=10,
    control_group="clean",
    cluster="id",
    inference="cluster",
    unsupported_event_policy="drop",
).fit(
    df,
    outcome="y",
    unit="id",
    time="year",
    treatment="treated",
    covariates=["x1", "x2"],
    ylags=1,
    dylags=1,
)

print(result.summary())
event_study = result.event_study
pooled = result.pooled
att = result.att
p_value = result.p_value
```

The causal score is always evaluated on reference dates containing at least
one eligible event and one clean control. With
`unsupported_event_policy="drop"` (default), events on dates without a clean
control are excluded and counted in `result.metadata["support_diagnostics"]`;
use `"error"` when changing the requested event population is unacceptable.

For `estimation_method="ra"`, the Dube et al. default is
`or_training_support="all_clean_controls"`: clean controls from dates without
an event may estimate the common covariate slopes, but never enter the ATT as
additional support. Set `or_training_support="event_dates"` for a strictly
matched RA benchmark. Propensity-based methods keep their nuisance scores on
event dates; auxiliary OR training is not yet certified for DR-IPW or DR-IPT.

`DRLPDID` uses the same public window defaults as the literature API:
`pre_window=2`, `post_window=0`, and `control_group="clean"`. Pass
`pre_window=None, post_window=None` explicitly to evaluate the full
data-supported grid. The default post-treatment scalar is the pooled ATT over
`h=0,...,post_window`; the pre row is a pooled placebo regression over the
requested leads, excluding the normalized base `h=-1`. Both `LPDID` and
`DRLPDID` support the same explicit aggregation vocabulary:

- `pooled`: one joint Dube et al. pooled-outcome regression on a complete
  common stack;
- `equal_horizon`: an equal-weighted average of separately estimated horizon
  effects;
- `event_weighted`: a horizon average weighted by eligible treated event
  rows, so every supported event-horizon observation receives equal weight;
- `all`: report all three summaries. The headline `result.att` remains the
  first requested method (`pooled` under `all` and by default).

`mean_horizons` remains accepted as a legacy spelling of `equal_horizon`. To
request all summaries on a specific post-treatment window:

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    pre_window=5,
    post_window=10,
    aggregate_window=(0, 5),
    pre_aggregation="all",
    post_aggregation="all",
    inference="multiplier",
    seed=123,
).fit(
    df, outcome="y", unit="id", time="year",
    treatment="treated", covariates=["x1", "x2"],
)

aggregates = result.aggregates
comparison_att = result.aggregate("post", "event_weighted")
```

`event_weighted` is the natural common target for comparison with a simple
group-time ATT aggregation in Callaway--Sant'Anna and with treated-observation
aggregates in Gardner or Borusyak--Jaravel--Spiess, provided all methods use
the same sample, horizon window, support rule, and event weights. This is
comparability of target weighting, not a claim of numerical identity.
For `LPDID`, this comparison also requires `reweight=True`; the default
`reweight=False` retains the variance-weighted LP-DiD estimand within each
horizon. Neither linear aggregation by itself imposes fixed cohort
composition across horizons.
`equal_horizon` is useful when every event-time horizon should receive equal
policy weight, including summaries of an interaction-weighted Sun--Abraham
path. Each linear aggregate stores its realized `horizons` and `weights`.

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

The compatibility defaults follow `diff-diff`: `history_policy="diff_diff"`
and `left_censoring="keep_as_control_after_L"`. For a design that must never
interpret an unobserved initial state as a transition, request
`history_policy="observed"` together with `left_censoring="error"` or
`"drop"`. This choice is recorded in result metadata.

## Advanced nuisance specifications

`covariates` supplies the same predetermined variables to both nuisance
models. The public fit vocabulary accepts `ylags=`, `dylags=`, and `absorb=`.
For diagnostics or research designs that require distinct bases, use
`ps_covariates=` and `or_covariates=` in `fit`. A generated outcome change or
target-period outcome level that is an algebraic component of a placebo is
omitted and recorded rather than used to manufacture a placebo fit. The older
constructor spelling `outcome_change_lags=` remains explicit only for
replication of pre-rc6 notebooks.

## Estimators and inference

- `ra`: regression adjustment;
- `ipw`: logit inverse-probability weighting;
- `ipt`: inverse-probability tilting;
- `dr-ipw`: logit-based doubly robust estimation;
- `dr-ipt`: IPT/WLS doubly robust estimation.

`LPDID` is the Dube et al. literature estimator (`vw`, `rw`, and `ra`, selected
through `reweight` and the supplied controls). `DRLPDID` contains the article's
RA, IPW, IPT, DR-IPW, and DR-IPT extensions. All five target the equally
weighted ATT on the same supported semiparametric stack; `DRLPDID` therefore
does not expose a binary `reweight` option. Only `dr-ipw` and `dr-ipt` are
doubly robust. RA, IPW, and IPT are single-model components and benchmarks.

For `DRLPDID`, `inference="cluster"` is the stacked M-estimation sandwich. The
bread is the inverse Jacobian of the complete nuisance-and-ATT moment system;
the meat is formed from cluster-summed empirical scores. The default
`small_sample_correction="cr1"` multiplies the variance by
`G/(G-1) * (n-1)/(n-K)`, using the effective parameter count `K` of the method
actually fitted. Set it to `"none"` for the uncorrected sandwich. Pointwise
tests use Student `t` reference tails with `G-1` degrees of freedom. The bread,
meat provenance, parameter count, cluster count, Jacobian rank, and condition
number are audited by tests and compact diagnostics are retained in metadata.

`LPDID(small_sample_correction="method_default")` reproduces the
method-specific `diff-diff` covariance convention; explicit `"cr1"` and
`"none"` remain available.

`cluster` may equal the panel-unit column or name a coarser grouping such as
state for a county panel. Coarser clustering is accepted only when each panel
unit is nested in exactly one cluster. For example, estimate county outcomes
with `unit="county_fips"` and request state inference with
`cluster="state_fips"`; pydrlpdid sums county influence functions within state
and uses the realized number of states for degrees of freedom. Units that move
between clusters and missing cluster labels are rejected.

Inference options are:

- `cluster`: analytic cluster-robust inference (default);
- `multiplier`: pointwise inference plus simultaneous event-study bands;
- `cluster_bootstrap`: paired cluster resampling with full re-estimation.

Multiplier weights are selected with `bootstrap_weights="rademacher"`,
`"mammen"`, or `"webb"`. Stochastic procedures record their seed, requested
replications, and bootstrap diagnostics. Observation-level HC inference is
not exposed because it ignores within-unit serial dependence in panel data.
In `0.8.1`, `cluster_bootstrap` is explicitly experimental; use at least
999 replications for reported work and inspect its success diagnostics.

The legacy `horizons=(-5, 10)` syntax and tuple-valued `post_window=(0, 10)`
remain available for replication code. New code should use `pre_window`,
integer `post_window`, and `aggregate_window`.

See `FORMAL_SPEC_0.8.1.md`, `MIGRATION_0.8.0.md`, and `RUN_REPLICATION.md`.

## License

MIT License.
