# Migration to 0.7.2

Version 0.7.2 preserves the formal DRLPDID engine and inferential corrections
of 0.7.1. It corrects the identity and routing of regression adjustment in
replication work:

- `LPDID(target_estimand="ra")` is the literature estimator of Dube et al.;
- `DRLPDID(estimation_method="ra")` is the formal common-stack estimator and
  is reported as `DRLPDID-RA`.

Applications must be regenerated because the benchmark `LPDID-RA` now uses
its native clean-control regression sample. Formal DRLPDID point estimates,
supported stacks, and covariance calculations are unchanged from 0.7.1.

The public constructor is intentionally narrower:

```python
DRLPDID(
    estimation_method="dr-ipt",
    design="absorbing",
    control_group="not_yet_treated",
    horizons=(-10, 19),
    post_window=(0, 19),
    inference="multiplier",
    n_bootstrap=999,
    seed=123,
)
```

For switching, use `design="switching"` and
`stabilization_window=L`. The treatment path must be supplied to `.fit()`.

The following pre-0.7 implementation choices are no longer public DRLPDID
options: alternative base periods, anticipation shifts, propensity clipping,
fallback estimators, fixed composition, stable-zero-only switching controls,
and alternative bootstrap weight families.

`LPDID` remains available for literature-benchmark replication. It is not a
second causal-design API for DRLPDID. Version 0.7.2 adds
`inference="multiplier"` to this benchmark class so its simultaneous bands
come from the estimator's own cluster influence functions.
