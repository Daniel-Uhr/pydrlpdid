# Changelog

## Unreleased

## 0.8.1

- Separates supported treated-event dates from auxiliary outcome-regression
  training dates.
- Makes `DRLPDID(estimation_method="ra")` follow the Dube et al. regression-
  adjustment convention by default: the ATT is evaluated on event dates with
  clean controls, while all eligible clean controls may train the common-slope
  counterfactual model.
- Adds `or_training_support={"method_default","event_dates",`
  `"all_clean_controls"}` and `unsupported_event_policy={"drop","error"}`.
- Keeps propensity scores and DR scores on supported event dates; auxiliary OR
  training is not yet certified for DR-IPW or DR-IPT.
- Reports requested, supported, and unsupported treated events separately from
  auxiliary OR rows and includes them in the stacked cluster influence function.
- Makes the RA path retain the Dube/diff-diff placebo-lag convention so RA is
  comparable over both pre- and post-treatment horizons; propensity and doubly
  robust methods retain the stricter overlap-omission diagnostic.

## 0.8.0rc7

- Adds one common pre/post aggregation API to `LPDID` and `DRLPDID`:
  `pooled`, `equal_horizon`, `event_weighted`, and `all`.
- Keeps `pooled` as the diff-diff-compatible default and preserves
  `mean_horizons` as a legacy alias for `equal_horizon`.
- Adds treated-event weighting for comparisons with estimators that report an
  overall ATT over supported treated event-horizon observations.
- Combines cluster influence functions using the declared horizon weights,
  retaining cross-horizon covariance; paired cluster bootstrap refits and
  recomputes the weights.
- Exposes all requested rows through `result.aggregates` and a named selector
  `result.aggregate(window, aggregation)`, while retaining `result.pooled`.
- Records the horizon set, realized weights, weight basis, and primary
  post-treatment aggregation in result metadata.

## 0.8.0rc6

- Aligns the public `DRLPDID` design vocabulary with `LPDID`: window defaults
  `pre_window=2`, `post_window=0`, clean controls, cluster degrees of freedom,
  rank-deficiency policy, and `fit(..., ylags=, dylags=, absorb=)`.
- Keeps the semiparametric distinction explicit: `DRLPDID` has no binary
  `reweight` flag; `estimation_method` selects RA, IPW, IPT, DR-IPW, or DR-IPT
  on the common eligible-entry ATT stack.
- Makes `inference="cluster"` an auditable stacked M-estimation sandwich and
  records Jacobian, score, cluster, and variance diagnostics.
- Corrects CR1 to use the effective parameter count of the method actually
  fitted rather than an artificial common full-DR reference dimension.
- Adds explicit `df_convention="cluster"`, `rank_deficient_action`,
  `only_event`, and `only_pooled` behavior.
- Uses the literature-compatible history boundary convention by default while
  retaining the stricter observed-history policy as an explicit option.
- Extends placebo-overlap protection to generated target-period outcome levels.

## 0.8.0rc5

- Makes `LPDID` the single public literature-benchmark estimator, with the
  constructor and `fit` vocabulary validated against `diff-diff` 3.12.
- Removes the separate public `LPDiD` name and the former estimand-oriented
  public constructor. The computational engine is now private implementation.
- Keeps `DRLPDID` as the only public extension surface for RA, IPW, IPT,
  DR-IPW, and DR-IPT.
- Migrates examples, installation checks, tests, and current replication
  scripts so user-facing code imports only `LPDID` and `DRLPDID`.

## 0.8.0rc4

- Supports analytic, multiplier, and paired-bootstrap inference at a coarser
  cluster level when panel units are strictly nested in clusters (for example,
  counties nested in states).
- Relabels panel units when resampling a coarse cluster more than once.
- Adds deterministic rank reduction for exactly collinear LPDID linear designs;
  the treatment coefficient must remain identified.
- Validates direct state clustering against manual county influence-function
  aggregation in the Cengiz et al. minimum-wage application.
- Adds `LPDiD`, a `diff-diff` 3.12-compatible facade, while retaining `LPDID`
  as the explicit estimand-oriented engine and `DRLPDID` as the semiparametric
  extension.
- Matches `diff-diff` event paths and pooled pre/post estimates for VW, RW,
  and RA under absorbing, first-entry, and effect-stabilization designs.
- Adds compatible `pmd`, `no_composition`, `ylags`, `dylags`, absorbed factors,
  custom pooled windows, output aliases, and rank-deficiency actions.
- Corrects first-entry membership: treated entries must remain treated through
  the target horizon and controls must never have entered treatment through
  that horizon. Pooled outcomes now require every requested target outcome.
- Keeps the `diff-diff` mechanically zero placebo convention when an included
  outcome-change lag is the negative of the placebo outcome only in `LPDiD`;
  native `LPDID` omits and records overlapping controls by default.
- Exposes `rank_deficient_action={"warn","error","silent"}` and records every
  deterministically removed design column.

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
