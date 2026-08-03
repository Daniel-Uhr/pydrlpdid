# Changelog

## 0.7.2

- Separates the literature estimator `LPDID-RA` from the formal common-stack
  estimator `DRLPDID-RA` in result metadata and all replication files.
- Routes the Bacon and Dube-compatible benchmark paths through the native
  Dube et al. regression-adjustment sample instead of the formal
  propensity-supported stack.
- Adds multiplier-bootstrap simultaneous bands to `LPDID` using its own
  cluster influence functions.
- Renames the RA member of the formal Monte Carlo and switching designs to
  `DRLPDID-RA`; its formula and supported stack are unchanged.
- Does not alter the 0.7.1 DRLPDID point estimators, convergence gates,
  treatment-history rules, or covariance formula.

## 0.7.1

- Aligns finite-sample output with the article's cluster influence-function
  formula by removing the representation-dependent CR1 degrees-of-freedom
  factor.
- Makes cluster-robust standard errors invariant to algebraically redundant
  nuisance blocks.
- Requires IPT and DRLPDID-IPT to agree in both point estimates and
  influence-function standard errors when their retained bases are nested.
- Updates the three complete replication notebooks and gives Monte Carlo
  checkpoints a package-and-settings signature with atomic writes.
- Does not change any formal point estimator, target population, supported stack,
  treatment-history rule, or public API.

## 0.7.0

- Aligns the public API with the two formal designs in the article.
- Uses one common supported reference-date stack across the five local ATT
  estimators.
- Adds deterministic rank reduction for all local nuisance designs.
- Tightens IPT acceptance to the solved balancing moments.
- Certifies the IPT–DRLPDID-IPT point-estimate identity under nested retained
  bases.
- Uses the measured retained-basis imbalance and fitted outcome coefficients
  to certify numerical IPT–DRLPDID-IPT discrepancies, avoiding a scale-free
  false failure at floating-point precision.
- Keeps normalized or unsupported horizons outside simultaneous bands.
- Keeps incomplete prespecified post windows out of scalar output.
- Fixes paired cluster bootstrap handling when no scalar term is requested.
- Adds a dedicated absorbing-adoption Monte Carlo driver for `N=500`.
- Adds complete Bacon and formal switching replication notebooks.

Version 0.7.0 does not add alternative treatment-history estimands or public
nuisance-model options.
