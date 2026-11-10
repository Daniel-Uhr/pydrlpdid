# Scientific audit — pydrlpdid 0.8.0rc1

## Closed implementation issues

- RA point estimation and RA influence functions use the same retained-rank
  design, sample, coefficients, and ATT estimate.
- Cluster-inference failures are explicit; there is no nonclustered fallback.
- Optional CR1 correction scales the complete cluster influence vector, so
  event-path, scalar, and multiplier inference are coherent.
- The Jacobian condition-number threshold is enforced.
- `post_window` cannot silently choose between a pooled outcome and an average
  of horizon-specific effects.
- Pooled and mean-of-horizons scalars have analytic and multiplier inference.
- Common OR slopes are the formal default. Calendar-interacted slopes are an
  explicit model choice and need not preserve IPT/DR-IPT nesting.
- RW placebo weights are horizon-specific by default; post-target weights are
  retained only as a named compatibility option.
- Initial treated levels are not inferred to be switch-ins. Left-censored
  spells require an explicit policy.
- Placebo leads never condition on generated outcome changes that algebraically
  constitute their own dependent variable.
- Monte Carlo seeds are deterministic and independent of Python's `hash()`.
- Failures and warnings are never globally suppressed in the validation driver.

## Scientific interpretation that must remain in the article

- A lead equals zero only under the relevant no-anticipation and conditional
  path restrictions. It is not a software identity and is not forced to zero.
- IPT and DR-IPT coincide when the retained OR basis is nested in the IPT
  balancing basis. This is a structural sample identity, not evidence that two
  empirically independent estimators agree.
- Double robustness is consistency under either a correct local PS or a correct
  local OR, conditional on the causal assumptions. It is not finite-sample
  dominance and does not imply that a DR estimator must have lower RMSE than RA
  when the OR is correct.
- Stable-one switching controls require the article's conditional-mean
  stabilization restriction. The package verifies observed paths, not that
  identifying restriction.
- Poor overlap remains visible through propensity, odds, and effective-control
  diagnostics; the package does not silently clip weights.

## Manuscript items outside the software release

- Rewrite the current Assumption 4 if it is an algebraic identity rather than
  an identifying restriction.
- State the LCPT conditioning history explicitly and align it with the
  observed-history implementation.
- Replace any blanket recommendation of DR-IPT with a criterion based on the
  maintained nuisance assumptions, overlap, and the Monte Carlo evidence.
- Recompute article applications under 0.8.0rc1 before replacing any published
  0.7.2 number. Results from the two versions should not be mixed.
