# pydrlpdid 0.8.0rc1

`pydrlpdid` estimates local-projection Difference-in-Differences effects for
staggered absorbing adoption and sustained treatment switch-ins. The formal
API contains RA, IPW, IPT, DR-IPW, and DR-IPT estimators on one common
horizon-specific comparison stack.

## Install

From the repository folder containing `pyproject.toml`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

If `python` is not Python 3.10 on Windows, use its full executable path:

```powershell
& "C:\Users\danie\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -e ".[dev]"
```

## Minimal use

```python
from pydrlpdid import DRLPDID

result = DRLPDID(
    estimation_method="dr-ipt",
    horizons=(-5, 10),
    inference="multiplier",
    seed=123,
).fit(
    df,
    outcome="y",
    unit="id",
    time="year",
    first_treat="first_treat",  # 0 for never treated
    covariates=["x1", "x2"],
)

print(result.summary())
event_study = result.event_study
```

To request a post-treatment scalar, name its estimand explicitly:

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    horizons=(-5, 10),
    post_window=(0, 10),
    post_aggregation="mean_horizons",  # or "pooled"
    inference="multiplier",
    seed=123,
).fit(
    df, outcome="y", unit="id", time="year",
    first_treat="first_treat", covariates=["x1", "x2"],
)
```

## Sustained switch-ins

```python
result = DRLPDID(
    estimation_method="dr-ipt",
    design="switching",
    stabilization_window=2,
    horizons=(-2, 5),
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

The default formal covariance has no degrees-of-freedom correction. Set
`small_sample_correction="cr1"` to scale the full influence representation,
including scalar and multiplier inference.

See `FORMAL_SPEC_0.8.0.md`, `MIGRATION_0.8.0.md`, and `RUN_REPLICATION.md`.

## License

MIT License.
