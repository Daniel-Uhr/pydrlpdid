# Changelog

## 0.8.0rc1

- Separates observed treatment history from the explicit `diff_diff`
  compatibility convention; initial treated levels are never inferred to be
  switch-ins under the default.
- Replaces the ambiguous left-censoring `keep` option with
  `keep_as_control_after_L` and preserves the old spelling only as a deprecated
  alias in the panel utility.
- Makes the common OR slope the default and exposes calendar-interacted slopes
  by name.
- Requires an explicit `post_aggregation` whenever `post_window` is requested.
- Uses one RA design for point estimation and influence-function inference.
- Applies optional CR1 scaling to the complete cluster influence function.
- Adds multiplier inference for both pooled and mean-of-horizons scalars.
- Enforces the configured Jacobian condition-number limit.
- Reports local propensity, control-odds, effective-sample-size, rank, and
  small-sample diagnostics.
- Exposes optional separate PS and OR covariate bases and a concise
  `outcome_change_lags` option.
- Prevents placebo regressions from conditioning on generated outcome changes
  that are algebraic components of the placebo dependent variable.
- Adds deterministic four-scenario Monte Carlo validation with recorded seeds,
  failures, timing, bias, variance, RMSE, coverage, and interval length.

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
