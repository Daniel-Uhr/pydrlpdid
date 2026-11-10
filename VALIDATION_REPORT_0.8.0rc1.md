# Validation report — pydrlpdid 0.8.0rc1

Date: 2026-11-10  
Python: 3.10.0  
Platform: Windows 10  

## Software gates

- Test suite: 24 passed.
- Editable installation: passed; import resolves to this development tree.
- Built wheel: `pydrlpdid-0.8.0rc1-py3-none-any.whl`.
- Wheel SHA-256:
  `31CD5D81EC7911284CEE6DEF5A8BB3D41664698E52C3A65F3F7C3922E6185596`.
- Monte Carlo fitting failures: 0 of 8,000 estimator fits.
- The Python environment emitted an unrelated warning about a leftover
  `~atplotlib-3.7.1.dist-info` directory; pydrlpdid does not depend on
  matplotlib for estimation and all gates passed.

## Final Monte Carlo

The final run uses 200 replications, 1,000 units, two treatment designs
(absorbing and sustained switch-in), four nuisance scenarios, five estimators,
and a common ATT truth of 2. All raw rows, seeds, inference, timing, and
failures are stored under `results/v080_final_R200_N1000/`.

For the post-treatment scalar ATT[0,2]:

| Scenario | Absorbing evidence | Switching evidence |
|---|---|---|
| Both correct | RA, DR-IPW, IPT and DR-IPT biases from -0.02 to 0.00; RMSE about 0.08 | biases about 0.00; RMSE about 0.08 |
| OR only | RA and both DR estimators have bias about 0.00; IPW/IPT biases 0.16–0.20 | RA/DR biases -0.01; IPW/IPT biases 0.15–0.19 |
| PS only | IPT/DR-IPT bias about 0.00; DR-IPW bias -0.02 but RMSE 0.31; RA bias 0.84 | IPT/DR-IPT and DR-IPW bias about 0.00; DR-IPW RMSE 0.26; RA bias 0.85 |
| Neither correct | biases 0.17–0.31 and coverage 0.21–0.63 | biases 0.17–0.30 and coverage 0.23–0.62 |

The h=-2 placebo is near zero when the relevant OR or PS restriction is
correct for the corresponding estimator. RA is strongly biased in the PS-only
case (about -0.41 to -0.43), confirming that leads are not constrained to zero
by code. No placebo conditions on its own generated outcome change.

## Interpretation

- The experiment supports the intended double-robustness logic in both formal
  designs.
- It does not show universal finite-sample dominance. Generic logit IPW and
  DR-IPW can have much larger RMSE under a correct PS because their odds are
  more variable; IPT/DR-IPT are markedly more stable in this DGP.
- When the OR is correct, RA can match or slightly outperform DR estimators in
  RMSE. The article must not call a DR method uniformly preferred.
- When both nuisances are wrong, none of the estimators is protected.

This Monte Carlo is a controlled nuisance-model validation, not an exhaustive
study of treatment-effect heterogeneity, many staggered cohorts, or every
failure mode of the stabilization restriction. Those broader experiments
remain article-level work and should not be inferred from this release gate.
