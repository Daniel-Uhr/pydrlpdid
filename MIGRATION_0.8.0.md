# Migration from 0.7.2 to 0.8.0

Version 0.8.0 changes defaults that affect scientific interpretation. It is
therefore intentionally released first as `0.8.0rc1`.

## Required changes

- A requested `post_window` now requires either
  `post_aggregation="mean_horizons"` or `post_aggregation="pooled"`.
- Replace `or_time_varying_slopes=False` with
  `or_slope_structure="common"`; use `"calendar_interacted"` only when that
  richer outcome model is substantively intended.
- For switching designs, decide how to handle units already treated at the
  beginning of the panel: `left_censoring="error"`, `"drop"`, or
  `"keep_as_control_after_L"`.

## New defaults

- `history_policy="observed"`: no unobserved pre-sample transition is imputed.
- `or_slope_structure="common"`: preserves the IPT/DR-IPT nesting identity
  when the retained OR basis lies in the IPT balancing span.
- `small_sample_correction="none"`: matches the article's uncorrected cluster
  influence formula and is invariant to redundant moment representations.
- `LPDID(..., pre_weighting="horizon_specific")`: RW weights follow the
  horizon-specific clean-control sample. `"post_target"` is compatibility
  behavior.

## Advanced covariates

The simple interface remains `covariates=[...]`. Distinct nuisance bases may
now be supplied as `ps_covariates=[...]` and `or_covariates=[...]` to `fit`.
Use `outcome_change_lags=k` to generate k lagged changes. Overlapping changes
are automatically removed from placebo leads and reported in metadata.
