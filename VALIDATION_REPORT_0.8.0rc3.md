# Validation report — pydrlpdid 0.8.0rc3

Date: 2026-11-10  
Certified runtime: Python 3.10.0 on Windows

## Inferential contract

- `small_sample_correction="cr1"` is the default for DRLPDID and LPDID.
- `small_sample_correction="none"` is the only alternative.
- Analytic event-study and aggregate inference uses Student `t` tails with
  target-specific `G-1` degrees of freedom.
- DRLPDID uses a common full-DR reference dimension in the CR1 factor so
  nested IPT and DR-IPT remain identical in both estimates and standard
  errors. LPDID uses its realized regression or stacked-RA dimension.
- The normalized base period is a zero normalization without an estimated
  standard error, p-value, or confidence interval.

## Software gates

- Source-tree tests under Python 3.10: 48 passed.
- Installed-wheel tests under Python 3.10: 48 passed.
- Post-installation import resolved to the clean validation environment, not
  the source tree.
- Wheel: `dist/pydrlpdid-0.8.0rc3-py3-none-any.whl`.
- SHA-256:
  `a3db539fb861364180c48b89a871005c319550fe40bb83877a556c784297e02c`.

The warnings in the test suite are prespecified safeguards: intentionally low
bootstrap draw counts in smoke tests, an explicit left-censoring case, and the
experimental cluster-bootstrap label.

## External diff-diff equivalence

The installed wheel was compared with `diff-diff 3.12.0` for:

- absorbing treatment, first entry, and effect stabilization;
- VW, RW, and RA;
- event-time effects from -2 through 2 and the pooled post ATT;
- estimates, standard errors, p-values, and confidence intervals.

The native diff-diff convention was reproduced using CR1 for VW/RW and the
uncorrected sandwich for RA. All nine design-estimand combinations passed a
`1e-8` tolerance. The largest coefficient discrepancy was `1.16e-14`; the
largest discrepancy among inferential quantities was `8.31e-11`.

Reproducible script:
`replication/compare_diff_diff_rc3.py`.

Installed-wheel output:
`results/diff_diff_equivalence_rc3_installed.csv`.

## Focused CR1 Monte Carlo

A focused validation under Python 3.10 used 100 replications, 120 units,
DR-IPT, the three treatment designs, and both correction choices. It produced
600 fitted models and 3,000 target rows with zero failure. Repeating the same
seeds under Python 3.11 produced zero numerical difference in estimates,
standard errors, intervals, and p-values.

| Inference | Mean coverage | Placebo rejection | Mean interval length |
|---|---:|---:|---:|
| Uncorrected sandwich | 0.9387 | 0.0700 | 1.0259 |
| CR1 | 0.9440 | 0.0667 | 1.0560 |

Point estimates were exactly invariant: the maximum difference between CR1
and the uncorrected fit was zero. CR1 increased mean interval length by about
2.9% and moved descriptive coverage toward 95%. This focused experiment is a
release validation, not a universal finite-sample optimality claim.

Outputs:
`results/inference_cr1_rc3_R100_N120_py310/` and
`results/inference_cr1_rc3_R100_N120/`.

## Release decision

The analytic cluster path is closed for `0.8.0rc3`. Multiplier inference
remains a supported path. Paired cluster bootstrap remains experimental and
requires at least 999 draws for reported use; its later coverage certification
does not block this release candidate.
