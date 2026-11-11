# Validation report — pydrlpdid 0.8.0rc4

Date: 2026-11-11
Certified runtime: Python 3.10 on Windows
External oracle: diff-diff 3.12.0

## Scope

This candidate adds a `diff_diff.LPDiD`-compatible facade without replacing
the explicit `LPDID` benchmark engine or the five-method `DRLPDID` API. The
validation concerns software equivalence and inferential bookkeeping; it does
not establish a new causal-identification theorem for doubly robust
non-absorbing switch-ins.

## External-oracle matrix

`tests/test_diffdiff_parity.py` contains 32 external-oracle tests covering:

- absorbing treatment, first entry, and effect stabilization;
- variance-weighted (VW), reweighted (RW), and regression-adjusted (RA)
  estimates;
- horizon estimates, analytic standard errors, p-values, and pooled pre/post
  estimates;
- missing outcomes and `no_composition=True`;
- PMD with a fixed number of periods and all available pre-periods;
- custom pooled pre/post windows;
- lagged outcome levels (`ylags`) and changes (`dylags`);
- absorbed factors, including failure-closed behavior when an absorbed level
  has no clean-control overlap;
- coarser clustering with units nested in clusters;
- rank-deficiency actions and result-name aliases.

All 32 tests pass. Equality tolerances are `1e-10` for point estimates and
`1e-8` relative tolerance for standard errors. The comparison uses the
method-specific `diff-diff` convention: CR1 for VW/RW and its uncorrected
influence-function covariance for RA. Users may override the compatibility
choice with `small_sample_correction="cr1"` or `"none"`.

## Scientific boundary for lagged changes

When a placebo outcome is exactly the negative of an included lagged outcome
change, `diff-diff` returns a mechanically zero lead. `LPDiD` reproduces this
for compatibility. Native `LPDID` and `DRLPDID` instead omit and record such
an overlapping regressor. A mechanically zero compatibility lead must not be
interpreted as empirical evidence of parallel trends.

## Treatment and sample corrections

- First-entry treated events remain treated at every date through the target
  horizon.
- First-entry controls have never entered treatment through that horizon;
  previously treated off-periods cannot enter the clean-control sample.
- Pooled estimates require every requested target outcome, rather than a
  unit-specific average over whichever horizons are observed.
- `no_composition` fixes post-treatment membership and outcome support while
  leaving placebo-lead samples horizon-specific, matching `diff-diff`.
- RA with absorbed factors drops treated rows whose absorbed level is absent
  among clean controls, avoiding unidentified counterfactual extrapolation.

## Software gates

- Source compilation: passed.
- Complete source-tree suite: 85 passed.
- External-oracle subset: 32 passed.
- Whitespace/error-marker audit: passed.
- Wheel:
  `dist/pydrlpdid-0.8.0rc4-py3-none-any.whl`.
- SHA-256:
  `2746bfae3ac99b279fa8b55c8439d99cd05b76300945836aecaa65d125d09d34`.
- The wheel was force-installed in a separate Python 3.10 environment. Import
  resolved to that environment's `site-packages`, reported version
  `0.8.0rc4`, and exposed `LPDiD`, `LPDID`, and `DRLPDID`.
- The installed-wheel example completed for the compatibility LP-DiD and
  DR-IPT paths and returned event-study plus pooled pre/post output.

## Remaining boundaries

- Survey-design inference in the `LPDiD` facade is not certified and raises an
  explicit `NotImplementedError`.
- Paired cluster bootstrap remains experimental; at least 999 successful
  draws and inspection of failure diagnostics are required for reported use.
- The software parity of doubly robust switch-in estimators does not replace
  estimator-specific identification arguments or the planned final Monte
  Carlo assessment. RA outperforming a DR method under a correctly specified
  outcome model is not a software failure and no DR method is labeled
  universally preferred.

## Release assessment

The analytic LP-DiD compatibility path is closed for the certified features in
`0.8.0rc4`. The native strict-history path and the semiparametric extensions
remain separately named so differences in samples, estimands, or assumptions
cannot be mistaken for a numerical replication failure.
