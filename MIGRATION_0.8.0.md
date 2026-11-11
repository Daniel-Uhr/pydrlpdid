# Migration from 0.7.2 to 0.8.0

Version 0.8.0 changes defaults that affect scientific interpretation. It is
therefore intentionally released first as `0.8.0rc1`. The diff-diff-compatible
surface and expanded inference options enter in `0.8.0rc2`; `0.8.0rc3`
changes the default finite-sample inference.

## Required changes

- A requested nondefault `aggregate_window` requires either
  `post_aggregation="mean_horizons"` or `post_aggregation="pooled"`.
- Replace `or_time_varying_slopes=False` with
  `or_slope_structure="common"`; use `"calendar_interacted"` only when that
  richer outcome model is substantively intended.
- For switching designs, decide how to handle units already treated at the
  beginning of the panel: `left_censoring="error"`, `"drop"`, or
  `"keep_as_control_after_L"`.

## New defaults

- `pre_window=None, post_window=None`: evaluate the largest data-supported
  event-time grid. An integer `post_window` is now the maximum event horizon.
- `inference="cluster"`: analytic cluster-robust inference is the default;
  request `"multiplier"` for simultaneous bands.
- `history_policy="observed"`: no unobserved pre-sample transition is imputed.
- `or_slope_structure="common"`: preserves the IPT/DR-IPT nesting identity
  when the retained OR basis lies in the IPT balancing span.
- `small_sample_correction="cr1"`: finite-cluster correction is now the
  default. Analytic p-values and intervals use Student `t` tails with `G-1`
  degrees of freedom. Use `"none"` for the uncorrected sandwich.
- `LPDID(..., pre_weighting="horizon_specific")`: RW weights follow the
  horizon-specific clean-control sample. `"post_target"` is compatibility
  behavior.

## Interface compatibility

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
Use `outcome_change_lags=k` to generate k lagged changes. Overlapping changes
are automatically removed from placebo leads and reported in metadata.
