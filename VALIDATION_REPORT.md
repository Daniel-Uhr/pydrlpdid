# Validation report — pydrlpdid 0.7.2

## Completed in the build workspace

- Parsed and byte-compiled every package, test, example, and notebook code
  cell.
- Confirmed the public `DRLPDID` constructor has only ten keyword arguments:
  estimator, design, control group/history length, horizons, post window, and
  inference settings.
- Built the wheel without downloading build dependencies.
- Inspected the wheel contents and confirmed version `0.7.2` in both package
  source and distribution metadata.
- Confirmed that all three notebooks are valid, clean nbformat 4 documents with no
  saved outputs.
- Added automated tests for common-stack equality, complete scalar windows,
  multiplier/base handling, and the IPT–DRLPDID-IPT point-estimate and
  standard-error identities under an intentionally rank-deficient nuisance
  design.
- Added automated identity tests that distinguish the Dube et al.
  `LPDID-RA` sample from formal `DRLPDID-RA`, and added native multiplier-band
  coverage for the literature benchmark.
- Confirmed statically that the Bacon notebook routes `LPDID-RA` through
  `LPDID(target_estimand="ra")`, while the formal Monte Carlo and switching
  notebooks label `DRLPDID(estimation_method="ra")` as `DRLPDID-RA`.
- Removed the representation-dependent CR1 factor so the reported sandwich
  equals the influence-function formula in the article.

## Runtime test required after installation

The build workspace did not contain `patsy` or `statsmodels` and network access
was unavailable, so the numerical unit tests and the empirical notebooks could
not be executed here. After installing the declared dependencies, run:

```powershell
py -m unittest discover -s tests -v
```

Then execute all three notebooks from top to bottom. They contain hard validation
gates for the common stack, complete requested windows, native standard errors,
and the nested-basis identity. The Monte Carlo similarly stops if any fit
fails; it never certifies a summary after silently dropping replications.

Version 0.7.2 therefore ships as a reproducible candidate release whose wheel
and source structure are validated, while the final numerical certification
must be produced in the replication environment with the declared scientific
Python dependencies.
