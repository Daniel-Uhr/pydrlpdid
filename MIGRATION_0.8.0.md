# Migration from 0.7.2 to 0.8.0

The 0.8 series changes defaults that affect scientific interpretation. It was
therefore initially validated through release candidates and is finalized in
`0.8.1`. The diff-diff-compatible
surface and expanded inference options enter in `0.8.0rc2`; `0.8.0rc3`
changes the default finite-sample inference; `0.8.0rc4` adds the validated
literature-standard implementation; `0.8.0rc5` makes `LPDID` its only public
name and moves the lower-level computational engine behind the public API;
`0.8.0rc6` aligns the public `DRLPDID` vocabulary and corrects its
method-specific cluster sandwich correction; `0.8.0rc7` adds common,
explicit pre/post aggregation rules to `LPDID` and `DRLPDID`.

## Required changes

- A requested nondefault `aggregate_window` requires an explicit
  `post_aggregation`. Use `pooled`, `equal_horizon`, `event_weighted`, or
  `all`; the old `mean_horizons` spelling remains an alias.
- Replace `or_time_varying_slopes=False` with
  `or_slope_structure="common"`; use `"calendar_interacted"` only when that
  richer outcome model is substantively intended.
- For switching designs, decide how to handle units already treated at the
  beginning of the panel: `left_censoring="error"`, `"drop"`, or
  `"keep_as_control_after_L"`.

## Public defaults

- `LPDID`: `pre_window=2`, `post_window=0`, `control_group="clean"`, and
  `reweight=False`, following `diff-diff`.
- `DRLPDID`: defaults to `pre_window=2`, `post_window=0`, matching the public
  literature API. Pass `pre_window=None, post_window=None` explicitly to
  evaluate the largest data-supported event-time grid.
- `inference="cluster"`: analytic cluster-robust inference is the default;
  request `"multiplier"` for simultaneous bands.
- `DRLPDID(history_policy="diff_diff")`: follows the literature-compatible
  boundary convention by default. Request `history_policy="observed"` with an
  explicit left-censoring rule when no pre-sample transition may be imputed.
- `or_slope_structure="common"`: preserves the IPT/DR-IPT nesting identity
  when the retained OR basis lies in the IPT balancing span.
- `DRLPDID(small_sample_correction="cr1")`: finite-cluster correction is the
  default. `LPDID(small_sample_correction="method_default")` reproduces the
  method-specific `diff-diff` convention; both accept explicit `"cr1"` or
  `"none"`.

## Interface compatibility

For code written against `diff-diff.LPDiD`, import `LPDID`:

```python
from pydrlpdid import LPDID

result = LPDID(
    pre_window=5,
    post_window=10,
    control_group="clean",
    reweight=True,
    cluster="id",
).fit(
    df, outcome="y", unit="id", time="year", treatment="treated",
    covariates=["x1", "x2"],
)
```

`LPDID` follows the `diff-diff` defaults and method-specific covariance
conventions. `DRLPDID` retains CR1 as its default. In rc6 its CR1 factor uses
the effective parameter count of the fitted semiparametric moment system.

Both estimators now accept, for example,

```python
LPDID(
    pre_window=5,
    post_window=10,
    pre_aggregation="all",
    post_aggregation="all",
)
```

The default remains `pooled`. `event_weighted` is the recommended reported
aggregate when the comparison estimator targets an equal-weighted average of
treated event-horizon observations. Check `result.aggregates` for the stored
support and weights rather than comparing headline coefficients with
different aggregation rules.

New code can pass the observed binary path directly:

```python
DRLPDID(pre_window=5, post_window=10).fit(
    df, outcome="y", unit="id", time="year", treatment="treated"
)
```

The legacy `horizons=(-5, 10)` argument remains supported. A tuple-valued
`post_window=(0, 10)` is interpreted as the legacy aggregate window; new code
should use `aggregate_window=(0, 10)`. Non-absorbing designs should use
`non_absorbing="first_entry"` or `"effect_stabilization"`; `design="switching"`
remains a compatibility spelling for effect stabilization.

## Advanced covariates

The simple interface remains `covariates=[...]`. Distinct nuisance bases may
now be supplied as `ps_covariates=[...]` and `or_covariates=[...]` to `fit`.
Use `fit(..., ylags=k, dylags=k, absorb=[...])`. Overlapping generated changes
and target-period outcome levels are automatically removed from placebo leads
and reported in metadata. The constructor spelling `outcome_change_lags=k`
remains only for replication of earlier notebooks.
