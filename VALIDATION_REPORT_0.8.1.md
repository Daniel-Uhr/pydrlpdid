# Validation report — pydrlpdid 0.8.1

Date: 2026-11-11

## Change validated

`DRLPDID(estimation_method="ra")` now follows the Dube et al. outcome-
regression training convention while keeping causal support separate:

- supported event dates define the ATT score;
- eligible clean controls on event and control-only dates may train the common-
  slope outcome regression;
- control-only dates do not expand treated support;
- treated-only dates are explicitly dropped and counted, or rejected with
  `unsupported_event_policy="error"`;
- the auxiliary outcome-regression moments enter the stacked influence
  function at the original cluster level.

Propensity and DR methods remain restricted to supported event dates. Auxiliary
OR training for DR-IPW/DR-IPT is rejected rather than exposed without a
certified influence function.

## Automated tests

Python 3.10 command:

```powershell
python -m pytest tests -q -x
```

Result: **97 passed, 25 warnings, 0 failed**.

The warnings are expected diagnostics for smoke-test bootstrap draw counts,
rank reduction, left-censored treatment, absorbed-factor non-overlap, and an
external `statsmodels` square-root warning under deliberately missing outcomes.

## External numerical oracle

The full `tests/test_diffdiff_parity.py` suite ran against `diff-diff 3.12.0`.
It covers:

- absorbing, first-entry, and effect-stabilization designs;
- variance-weighted, reweighted, and regression-adjustment LPDID;
- event-study and pooled pre/post estimates;
- missing outcomes and `pmd`;
- fixed composition;
- custom pooled windows;
- outcome-level and outcome-change lags;
- absorbed factors and rank deficiency;
- clusters coarser than the panel unit.

All declared equality gates passed. A separate nine-cell matrix comparing
estimates, standard errors, p-values, and confidence limits passed the `1e-8`
tolerance in every design-by-estimator cell.

An additional RA regression test uses both outcome-level and outcome-change
lags and verifies the Dube placebo convention. In the full applications, the
largest LPDID-versus-DRLPDID-RA difference is approximately `1.2e-13` for a
coefficient and `3.9e-9` for a CR1 standard error.

## Empirical application gate

The Dube et al. banking and democracy applications were rerun from the installed
wheel. `DRLPDID(method="ra")` and `LPDID(reweight=True)` now agree to numerical
precision for the audited horizons and CR1 standard errors.

The rerun is documented in the research project's application report,
`RERUN_REPORT_0.8.1_2026-11-11.md`.

## Artifact

Wheel: `dist/pydrlpdid-0.8.1-py3-none-any.whl`
SHA-256: `92c76c445714474cefcadd84cec28288069e683ed324902abcaaabce14b64537`
