# Changelog

## 0.8.0rc3

- Makes `small_sample_correction="cr1"` the default for DRLPDID and LPDID;
  the only alternative is the uncorrected cluster sandwich (`"none"`).
- Uses Student `t` reference tails with target-specific `G-1` degrees of
  freedom for analytic event-study and aggregate inference.
- Uses one common full-DR CR1 reference dimension across the five DRLPDID
  estimators, preserving the IPT/DR-IPT identity under a nested OR basis.
- Records cluster counts, degrees of freedom, and critical values in reported
  analytic results; the normalized base period now carries no inference.

## 0.8.0rc2

- Adds a diff-diff-compatible public surface with `pre_window`, integer
  `post_window`, `control_group="clean"`, `cluster`, and binary `treatment`.
- Infers the largest data-supported event-time grid when windows are omitted;
  preserves `horizons=` and tuple-valued `post_window` for legacy replication.
- Exposes pre/post aggregate tables and `att`, `se`, `p_value`, `conf_int`, and
  `to_dataframe()` result aliases.
- Exposes `non_absorbing="first_entry"` alongside effect stabilization.
- Exposes Rademacher, Mammen, and Webb multiplier weights and adds paired
  cluster bootstrap with explicit success diagnostics. Cluster bootstrap is
  marked experimental pending a larger outer coverage experiment.
- Requires an explicit `post_aggregation` whenever a nondefault
  `aggregate_window` is requested.

## 0.8.0rc1

- Separates observed treatment history from the explicit `diff_diff`
  compatibility convention; initial treated levels are never inferred to be
  switch-ins under the default.
- Replaces the ambiguous left-censoring `keep` option with
  `keep_as_control_after_L` and preserves the old spelling only as a deprecated
  alias in the panel utility.
- Makes the common OR slope the default and exposes calendar-interacted slopes
  by name.
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
