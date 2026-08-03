# pydrlpdid 0.7.2

`pydrlpdid` implements doubly robust local-projections
Difference-in-Differences estimators for staggered absorbing adoption and
sustained treatment switch-ins.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Daniel-Uhr/pydrlpdid.git
cd pydrlpdid
pip install -e .
````

### Windows: install from the repository folder

1. Choose **Code > Download ZIP** on GitHub.
2. Extract the ZIP.
3. Open PowerShell in the extracted folder containing `pyproject.toml`.
4. Run:

```powershell
py -m pip install --upgrade pip
py -m pip install .
```

Verify:

```powershell
py verify_install.py
```

The expected version is `0.7.2`.

For an editable development installation:

```powershell
py -m pip install -e ".[dev]"
py -m pytest -q
```

A wheel installation command should be used only after the wheel has actually
been downloaded from a GitHub Release. See `INSTALL_WINDOWS.md`.

## Estimators

The formal `DRLPDID` API provides:

- `ra`: regression adjustment;
- `ipw`: logit inverse-probability weighting;
- `ipt`: inverse-probability tilting;
- `dr-ipw`: logit-based doubly robust estimation;
- `dr-ipt`: IPT--WLS doubly robust estimation.

Only `dr-ipw` and `dr-ipt` carry the double-robustness property, conditional on
the identifying assumptions.

`LPDID` remains available as the literature benchmark API. In particular,
`LPDID(target_estimand="ra")` is the Dube et al. regression-adjustment
benchmark, while `DRLPDID(estimation_method="ra")` is the regression-only
member of the common supported local-ATT stack.

## Minimal import

```python
from pydrlpdid import DRLPDID, LPDID
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

## Replication

See `RUN_REPLICATION.md` for the article notebooks and certified Monte Carlo
workflow.

## License

MIT License.
