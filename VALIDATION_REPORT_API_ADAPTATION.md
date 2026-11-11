# Validation report — pydrlpdid 0.8.0rc2 API adaptation

Date: 2026-11-10

## Scope

This report covers the public API adaptation requested for `DRLPDID`. It does
not replace the estimator-specific Monte Carlo validation and does not certify
new empirical claims.

## Implemented contract

- `pre_window` and integer `post_window`, with data-supported inference when
  either is omitted;
- `control_group="clean"` and optional `cluster` (currently the panel unit or
  a one-to-one relabeling only);
- binary `treatment` input for absorbing designs, with optional
  `first_treat` retained and cross-validated when both are supplied;
- `non_absorbing="first_entry"` and
  `non_absorbing="effect_stabilization"`;
- `cluster`, `multiplier`, and paired `cluster_bootstrap` inference;
- Rademacher, Mammen, and Webb multiplier weights;
- pre/post aggregate output plus `att`, `se`, `p_value`, `conf_int`,
  `pooled`, and `to_dataframe()` result surfaces;
- bootstrap requested/successful counts, failure classes, and target-specific
  valid-draw counts;
- preservation of the legacy `horizons`, `design`, and tuple-valued
  `post_window` spellings.

`reweight` was intentionally not added to `DRLPDID`: IPW, IPT, DR-IPW, and
DR-IPT already define their own weighting equations on a common ATT target.
An observation-level `robust`/HC option was also not added because it would
ignore within-unit serial dependence; analytic cluster inference is the
public robust covariance for the panel design.

## Checks executed

- Python compilation succeeded for the modified inference, results, and
  estimator modules.
- `git diff --check` reported no whitespace errors.
- The pre-existing 35-test suite passed after the API changes (one expected
  left-censoring warning in the explicit drop-policy test).
- Five additional first-entry method checks passed for RA, IPW, IPT, DR-IPW,
  and DR-IPT.
- Five full-reestimation cluster-bootstrap smoke checks passed for RA, IPW,
  IPT, DR-IPW, and DR-IPT.
- Rademacher, Mammen, and Webb multiplier choices each passed the public API
  and metadata check.
- The final 45-test suite passed against the rebuilt wheel in the separate
  Python 3.10 environment (8 formal API, 34 contract, and 3 estimator-identity
  tests). Expected warnings identify smoke-level bootstrap draw counts,
  experimental cluster bootstrap, and the explicit left-censoring drop test.
- Five regression checks establish numerical identity, to `1e-12`, between
  the legacy and new absorbing-treatment surfaces for RA, IPW, IPT, DR-IPW,
  and DR-IPT.
- Wheel `pydrlpdid-0.8.0rc2-py3-none-any.whl` was built without downloading
  dependencies; SHA-256 is
  `0A9F1B54172D78BFE9D304905F3F88C01304414A6D7C7EE0FDAE1DF5AFC96C39`.
- A separate Python 3.10 environment installed and imported the wheel from its
  own `site-packages`; `verify_install.py` passed.
- The installed-wheel example reproduced the generated ATT truth closely
  (`ATT=2.0497` for a DGP with truth 2), returned both aggregate rows, and
  displayed the complete event-study path.

## Remaining validation before release

- Validate coarser-than-unit clustering before permitting it publicly.
- A larger outer Monte Carlo with reported percentile coverage is required
  before cluster bootstrap can lose its experimental label. This does not
  block `rc2`, whose certified release paths are analytic cluster and
  multiplier inference.

