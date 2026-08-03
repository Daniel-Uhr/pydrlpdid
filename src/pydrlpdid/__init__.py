"""pydrlpdid — Doubly Robust Local Projections Difference-in-Differences.

Implements the formal DR-LP-DiD estimators for absorbing staggered adoption
and sustained switch-ins, together with ``LPDID`` as a literature benchmark.

Estimators
----------
DRLPDID
    Doubly-robust LP-DiD. For each horizon-specific clean-control stack it
    estimates a local clean-history outcome regression and/or a local treatment
    propensity score. Five estimator identities are exposed:

    * ``'ra'``  — regression adjustment (outcome model only).
    * ``'ipw'`` — logit inverse-probability weighting.
    * ``'ipt'`` — inverse-probability tilting.
    * ``'dr-ipw'`` — logit-based doubly robust estimation.
    * ``'dr-ipt'`` — improved IPT--WLS doubly robust estimation (default).

    Inference is influence-function based, with a multiplier bootstrap for
    simultaneous event-study bands; clustering is at the unit level.

LPDID
    Benchmark LP-DiD estimator (variance-weighted, reweighted, and
    regression-adjustment estimands; absorbing and non-absorbing clean-control
    rules). Provided here as the base method and for benchmarking.

Utilities
---------
plot_event_study
    Quick event-study plot from any result object.

Common interface::

    from pydrlpdid import DRLPDID
    res = DRLPDID(estimation_method="dr-ipt").fit(
        data, outcome="y", unit="id", time="t", first_treat="g",
        covariates=["x1", "x2"],
    )
    res.print_summary()

References
----------
Uhr, D. de A. P., & Moura, G. V. (2026). Doubly Robust Local Projections
    Difference-in-Differences. Working paper.

Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025). A local projections
    approach to difference-in-differences. *Journal of Applied Econometrics*,
    40, 741–758. https://doi.org/10.1002/jae.70000

Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
    estimators. *Journal of Econometrics*, 219, 101–122.
"""

from .lpdid import LPDID
from .drlpdid import DRLPDID
from ._plotting import plot_event_study
from ._results import LPDIDResults, DRLPDIDResults
from ._errors import (DRLPDIDError, PanelValidationError, TreatmentValidationError,
                      SupportError, NuisanceConvergenceError, JacobianError,
                      InferenceError, ExperimentalFeatureError)

__version__ = "0.7.2"

__all__ = [
    "DRLPDID",
    "LPDID",
    "DRLPDIDResults",
    "LPDIDResults",
    "plot_event_study",
    "DRLPDIDError", "PanelValidationError", "TreatmentValidationError",
    "SupportError", "NuisanceConvergenceError", "JacobianError",
    "InferenceError", "ExperimentalFeatureError",
]
