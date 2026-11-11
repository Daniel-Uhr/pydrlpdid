# Validation report — pydrlpdid 0.8.0rc5

Date: 2026-11-11
Certified runtime: Python 3.10 on Windows
External oracle: diff-diff 3.12.0

## Public architecture

The package has exactly two public estimator classes:

- `LPDID`: literature-standard LP-DiD benchmark;
- `DRLPDID`: article extensions RA, IPW, IPT, DR-IPW, and DR-IPT.

`LPDiD` is not exported and is not present as an attribute of the installed
package. The lower-level LP-DiD computational engine is private implementation,
not a second user API.

The public `LPDID` constructor follows the `diff-diff.LPDiD` vocabulary:
`pre_window`, `post_window`, `control_group`, `reweight`, `no_composition`,
`pmd`, `cluster`, `non_absorbing`, `stabilization_window`, and
`df_convention`. Its `fit` method accepts `treatment`, covariates, outcome
lags, absorbed factors, and custom pooled windows. Public scalar output is
limited to the pooled pre-treatment placebo and pooled post-treatment ATT.

## External numerical validation

The 32 external-oracle comparisons in `tests/test_diffdiff_parity.py` cover:

- absorbing, first-entry, and effect-stabilization designs;
- VW, RW, and RA;
- event paths and pooled pre/post estimates;
- estimates, standard errors, p-values, and confidence intervals;
- missing outcomes, PMD, fixed post composition, custom pooled windows,
  outcome-level and outcome-change lags, absorbed factors, rank deficiency,
  and nested coarser clustering.

All comparisons pass with absolute point-estimate tolerance `1e-10` and
relative standard-error tolerance `1e-8`. A separate API-contract test verifies
that `LPDID` is exported and `LPDiD` is absent.

## Software gates

- Complete source-tree suite: 86 passed.
- External-oracle comparisons: 32 passed.
- Public-name contract: passed.
- Source compilation: passed.
- Installed-wheel example: passed for `LPDID` and `DRLPDID`.
- Installed package exposed no `LPDiD` attribute.
- Wheel: `dist/pydrlpdid-0.8.0rc5-py3-none-any.whl`.
- SHA-256:
  `19514da4c661b8f8f9fd2de6a9005b6eb40345939f9a3c300dc9a36a703484a3`.

## Scientific boundaries

- `LPDID` follows the literature package's method-specific analytic covariance
  convention by default; users can explicitly request CR1 or no correction.
- Following `diff-diff`, an overlapping lagged outcome change can make a
  placebo mechanically zero. This is documented as an algebraic convention,
  not evidence supporting parallel trends. `DRLPDID` omits and reports the
  overlap.
- `DRLPDID(estimation_method="ra")` remains distinct from literature `LPDID`
  RA because it belongs to the common support and nuisance architecture used
  for the article's estimator comparisons.
- Survey-design inference is not certified. Paired cluster bootstrap remains
  experimental. The final DR switch-in Monte Carlo remains a scientific
  validation task rather than an API-equivalence task.

## Release assessment

The user-facing estimator architecture is closed for `0.8.0rc5`: one
literature benchmark (`LPDID`) and one extension class (`DRLPDID`). No public
alternative spelling or estimand-oriented LP-DiD constructor remains.
