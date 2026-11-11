# Inference pilot — pydrlpdid 0.8.0rc2

Date: 2026-11-10

## Design

The operational pilot used the correctly specified nuisance DGP from the
prespecified 0.8.0 Monte Carlo. It crossed:

- absorbing adoption, first entry, and effect stabilization;
- RA, IPW, IPT, DR-IPW, and DR-IPT;
- analytic cluster inference, CR1, Rademacher/Mammen/Webb multiplier
  inference, and paired cluster bootstrap;
- three replications with 120 panel units;
- 99 multiplier draws and five cluster-bootstrap draws.

The pilot is a software and coherence gate. Three replications cannot estimate
coverage accurately.

## Operational results

- 270 top-level fits and 1,350 target rows completed with no fit failure.
- Point estimates were exactly invariant across inference choices; the maximum
  within-design/method/replication/target spread was zero.
- All cluster-bootstrap fits met their requested success rule; mean success was
  100% in the broad pilot.
- Mean elapsed time per fit was about 0.34 seconds for analytic/CR1/multiplier
  inference and 2.59 seconds for cluster bootstrap with only five draws.
- CR1 modestly widened intervals relative to the uncorrected cluster sandwich.
- All multiplier choices returned finite pointwise and simultaneous intervals.

Across all methods, designs, horizons, and the post ATT, the descriptive
coverage rates were 0.933 for cluster, 0.956 for CR1, 0.902 for Mammen, 0.916
for Rademacher, and 0.911 for Webb. These fractions are diagnostics, not Monte
Carlo coverage estimates.

With only five draws, cluster-bootstrap percentile intervals were unstable:
mean descriptive coverage was 0.729 and mean interval length was materially
shorter than the analytic alternatives. Five draws are therefore unsuitable
for inferential reporting.

## Focused cluster-bootstrap check

A second run used DR-IPT, the three treatment designs, three replications, 120
units, 49 bootstrap draws, and a 90% minimum-success gate.

- No top-level fit failed.
- Target-specific bootstrap success ranged from 95.9% to 100% (mean 99.1%).
- Interval lengths increased materially relative to the five-draw smoke test,
  confirming that the earlier short intervals were driven by an inadequate
  number of draws.
- Mean elapsed time was 21.0 seconds for absorbing adoption, 21.8 seconds for
  first entry, and 27.2 seconds for effect stabilization per fitted result.

Even 49 draws and three replications do not certify percentile coverage.

## Release decision

The pilot closes the implementation question: all inference routes execute,
preserve point estimates, expose diagnostics, and fail explicitly. Analytic
cluster and multiplier inference can remain certified release paths, supported
also by the prespecified 200-replication estimator validation.

Paired cluster bootstrap remains **experimental** in `0.8.0rc2`. The public
API warns users, recommends at least 999 draws for reported work, and exposes
success diagnostics. Promoting it to a certified stable path requires a larger
coverage experiment with enough outer replications and bootstrap draws.

Outputs are preserved under:

- `results/inference_pilot_rc2_R3_N120/`;
- `results/inference_pilot_rc2_bootstrap_B49/`.

