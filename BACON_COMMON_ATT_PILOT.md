# Bacon common-ATT pilot

Date: 2026-11-10

## Scope

This pilot uses `data/bacon_example.dta` to check the revised pydrlpdid
implementation against diff-diff LP-DiD and to construct an initial
fixed-composition comparison with Callaway--Sant'Anna. It is not yet the final
replacement for Table 7.

The source-data SHA-256 is
`5A555448E605F75603160BE20F56CA343600EC81863F3060994BAC1285C29094`.

## Environment

- Python 3.10.0;
- pydrlpdid 0.8.0rc3, installed from the validated wheel;
- diff-diff 3.12.0;
- pandas 2.3.3;
- NumPy 2.0.2.

The run used an isolated Python 3.10 environment. No pre-existing Python
package was uninstalled.

## Sample and target

The data contain 49 units. Eight units treated throughout the observed panel
are excluded because they have no observed pretreatment period. The remaining
sample contains 36 switchers and five never-treated units. For the fixed
post-treatment window `h=0,...,19`, 33 switchers in nine adoption cohorts
(1969--1977) have complete support. The late cohorts 1980, 1984, and 1985 are
not target cohorts for this scalar.

The primary pilot target is the equal-horizon, fixed-composition average over
`h=0,...,19`, with the homicide mortality rate (`asmrh`) as the common
covariate and state-level clustered inference.

## Results

| Estimator | Implementation | Estimate | SE | p-value | 95% interval |
|---|---|---:|---:|---:|---:|
| LPDID-RA | pydrlpdid | -6.7190 | 3.1623 | 0.0402 | [-13.1208, -0.3173] |
| LPDID-RA | diff-diff | -6.7190 | 3.1623 | 0.0402 | [-13.1208, -0.3173] |
| Callaway--Sant'Anna DR | diff-diff | -6.1069 | 3.6148 | 0.0989 | [-13.4127, 1.1988] |
| Callaway--Sant'Anna REG | diff-diff | -6.1311 | 3.6142 | 0.0976 | [-13.4357, 1.1736] |
| DRLPDID-IPT pooled | pydrlpdid | -7.4313 | 5.7056 | 0.2008 | [-18.9919, 4.1293] |

The pydrlpdid and diff-diff LP-DiD post-treatment results pass the numerical
parity gate. The scalar estimate differs by `1.51e-14`, its standard error by
`5.16e-11`, the largest post-horizon point-estimate difference by `6.93e-14`,
and the largest post-horizon standard-error difference by `4.51e-09`.

The pretreatment paths are not certified as equivalent under the two current
fixed-composition implementations: the largest lead point-estimate difference
is 7.382. This does not affect the post-treatment scalar parity reported above,
but it must be resolved before claiming complete event-study equivalence.

## Diagnostics and interpretation

The common Callaway--Sant'Anna DR scalar is computed from the event-study
covariance matrix after imposing `balance_e=19`; the package-native overall ATT
is not used. All 372 reported propensity-score cohort-time cells fall below
the package's EPV threshold, with minimum EPV 1 and median EPV 2. The fit also
reports nonconvergence, large coefficients, near-separation, and trimming in
some cells. The regression-only CS estimate is nevertheless close to the DR
estimate, but that agreement does not eliminate the overlap concern.

The DRLPDID-IPT row uses a pooled common post-window stack. It has the same
calendar window and fixed-window causal target, but it is not yet the same
equal-horizon computational aggregation used in the LP-DiD and CS rows.
Accordingly, it is a provisional diagnostic rather than the final Table 7
comparison.

The empirical application has no known ground truth. Differences across these
point estimates measure estimator sensitivity, not relative accuracy. A paired
state-cluster bootstrap is still required to quantify differences between
estimators.

## Required next step for Table 7

1. Add an explicit target-cohort/target-cell restriction to the DRLPDID
   benchmarking path while retaining late adopters as eligible controls before
   treatment.
2. Apply the same fixed cohort-horizon weights to DRLPDID, LP-DiD, CS, BJS,
   Sun--Abraham, and Gardner.
3. Re-estimate all methods under common state-cluster bootstrap draws.
4. Move package-native overall ATTs to a separate sensitivity table.

The reproducible pilot is `replication/compare_bacon_common_att.py`. Machine
outputs are written to the ignored directory
`results/bacon_common_att_pilot/`.
