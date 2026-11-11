"""Formal semiparametric DR-LP-DiD estimator.

Each reported horizon is estimated on the clean local event stack defined in
the paper. Five estimator identities share that stack: regression adjustment,
logit IPW, inverse-probability tilting, logit-based doubly robust estimation,
and the improved IPT--WLS doubly robust estimator.

The public API exposes only the two causal designs established in the paper:
absorbing staggered adoption and sustained switch-ins. Inference clusters at
the panel-unit level. ``inference='multiplier'`` adds simultaneous sup-t bands
without replacing the analytic cluster-robust pointwise intervals.

References
----------
Sant'Anna, P. H. C., & Zhao, J. (2020).
    Doubly robust difference-in-differences estimators.
    *Journal of Econometrics*, 219(1), 101–122.

Graham, B. S., Pinto, C. C., & Egel, D. (2012).
    Inverse probability tilting for moment condition models with missing data.
    *Review of Economic Studies*, 79(3), 1053–1079.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize
from scipy.special import expit

from ._errors import (
    NuisanceConvergenceError, JacobianError, InferenceError, SupportError,
)

from ._inference import (
    run_cluster_bootstrap,
    run_multiplier_bootstrap,
    se_from_influence,
    stacked_influence,
)
from ._panel_utils import (
    analytic_cluster_inference,
    build_local_sample,
    check_columns,
    infer_windows,
    prepare_panel,
    precompute_ccs,
)
from ._results import DRLPDIDResults
from ._aggregation import horizon_aggregate, normalize_aggregations


# ---------------------------------------------------------------------------
# Data container for fitted propensity-score models
# ---------------------------------------------------------------------------

@dataclass
class LocalPSResult:
    """Container for a locally fitted propensity-score model.

    Attributes
    ----------
    params : np.ndarray
        Estimated parameter vector (gamma_hat).
    exog : np.ndarray
        Design matrix used in estimation.
    design_info : object
        Patsy ``DesignInfo`` object for out-of-sample prediction.
    formula : str
        Patsy formula string.
    method : str
        ``'logit'`` or ``'ipt'``.
    success : bool
        Whether the optimizer converged.
    optimizer_result : object, optional
        Raw result from ``scipy.optimize.minimize``.
    balance_error : float
        Maximum absolute sample IPT balancing-moment residual.
    """

    params: np.ndarray
    exog: np.ndarray
    design_info: object
    formula: str
    method: str
    success: bool
    optimizer_result: object = None
    balance_error: float = np.nan
    column_names: Tuple[str, ...] = ()
    dropped_columns: Tuple[str, ...] = ()


@dataclass
class LocalORResult:
    """Container for a full-rank local outcome-regression fit."""

    params: np.ndarray
    exog: np.ndarray
    design_info: object
    formula: str
    column_names: Tuple[str, ...]
    dropped_columns: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _control_columns(
    covariates: Optional[List[str]],
    include_ldy: bool,
    n_change_lags: int,
    n_level_lags: int = 0,
) -> List[str]:
    """Collect the list of right-hand-side control column names."""
    cols: List[str] = []
    if n_level_lags > 0:
        cols.extend([f"ly{k}" for k in range(1, int(n_level_lags) + 1)])
    if include_ldy and n_change_lags > 0:
        cols.extend([f"ldy{k}" for k in range(1, int(n_change_lags) + 1)])
    if covariates:
        cols.extend(list(covariates))
    return cols


def _build_formula(
    lhs: str,
    time: str,
    controls: List[str],
    add_time_fe: bool = True,
    time_varying_slopes: bool = False,
) -> str:
    """Construct a patsy formula for nuisance model estimation.

    When ``time_varying_slopes=True`` and time fixed effects are included,
    user-supplied controls enter interacted with calendar time. This yields a
    saturated local propensity-score specification of the form
    ``C(time) + C(time):X`` and imposes IPT balance within calendar-time cells
    of the horizon-specific LP-DiD stack.
    """
    controls = list(controls or [])
    rhs: List[str] = []

    if add_time_fe:
        rhs.append(f"C({time})")

    if controls:
        if add_time_fe and time_varying_slopes:
            rhs.extend([f"C({time}):{c}" for c in controls])
        else:
            rhs.extend(controls)

    if not rhs:
        return f"{lhs} ~ 1"
    return lhs + " ~ " + " + ".join(rhs)


def _restrict_to_treated_time_cells(
    local_sample: pd.DataFrame,
    time: str,
) -> pd.DataFrame:
    """Keep only time cells with both treated entrants and clean controls.

    With saturated time-by-covariate propensity-score models, time cells that
    contain only controls have no treated covariate moments to balance and can
    create separation-like IPT problems. They also do not contribute to a
    treated-entry ATT contrast. This restriction is therefore applied only to
    propensity-score based estimators when ``ps_time_varying_slopes=True``.
    """
    if local_sample.empty:
        return local_sample.copy()

    cell_stats = local_sample.groupby(time)["D_local"].agg(["sum", "count"])
    valid_times = cell_stats.index[
        (cell_stats["sum"] > 0) & (cell_stats["sum"] < cell_stats["count"])
    ]

    out = local_sample.loc[local_sample[time].isin(valid_times)].copy()
    if out.empty:
        raise ValueError(
            "No calendar-time cell contains both treated entrants and clean "
            "controls for propensity-score estimation."
        )
    return out.reset_index(drop=True)


def _drop_all_treated_time_cells(
    local_sample: pd.DataFrame,
    time: str,
) -> pd.DataFrame:
    """Drop calendar-time cells that contain no clean control.

    A treated-only cell has an unidentified time fixed effect in the
    outcome regression (so ``predict`` fails) and produces perfect
    separation in any time-fixed-effect propensity score. Treated units in
    such cells cannot be imputed or reweighted and are excluded. Control-only
    cells are kept: they inform the shared covariate coefficients and
    contribute (near-)zero propensity weight. This is the unconditional
    counterpart of the RA-LP-DiD restriction in the lpdid module and is
    required because build_local_sample now keeps control-less cells to match
    reghdfe on covariate specifications.
    """
    if local_sample.empty:
        return local_sample.copy()
    n_ctrl = local_sample.loc[local_sample["D_local"] == 0].groupby(time).size()
    keep = n_ctrl.loc[n_ctrl > 0].index
    out = local_sample.loc[local_sample[time].isin(keep)].copy()
    return out.reset_index(drop=True)


def _partition_support_samples(
    local_sample: pd.DataFrame,
    time: str,
    *,
    unsupported_event_policy: str,
    or_training_support: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Separate the causal target stack from auxiliary OR training rows.

    The target stack contains only reference dates with at least one eligible
    event and one clean control.  Treated-only dates are therefore outside the
    nonparametrically supported ATT.  Under the Dube-style outcome-regression
    convention, clean controls from dates without an event may additionally
    train the common-slope outcome model, but they never enter the propensity
    score or the ATT score.
    """
    if local_sample.empty:
        return local_sample.copy(), local_sample.copy(), {
            "n_requested_event_rows": 0,
            "n_supported_event_rows": 0,
            "n_unsupported_event_rows": 0,
            "n_control_only_dates": 0,
            "n_auxiliary_or_control_rows": 0,
            "n_target_rows": 0,
            "n_or_training_rows": 0,
        }

    cell = local_sample.groupby(time)["D_local"].agg(["sum", "count"])
    n_controls = cell["count"] - cell["sum"]
    mixed_times = cell.index[(cell["sum"] > 0) & (n_controls > 0)]
    treated_only_times = cell.index[(cell["sum"] > 0) & (n_controls == 0)]
    control_only_times = cell.index[(cell["sum"] == 0) & (n_controls > 0)]

    unsupported_mask = (
        local_sample[time].isin(treated_only_times)
        & local_sample["D_local"].eq(1)
    )
    n_unsupported = int(unsupported_mask.sum())
    if n_unsupported and unsupported_event_policy == "error":
        raise SupportError(
            "The requested horizon contains "
            f"{n_unsupported} treated event(s) on "
            f"{len(treated_only_times)} reference date(s) without a clean "
            "control. Use unsupported_event_policy='drop' to estimate the "
            "supported-event ATT explicitly."
        )

    target = local_sample.loc[local_sample[time].isin(mixed_times)].copy()
    target["_target_support"] = True

    if or_training_support == "all_clean_controls":
        or_training = local_sample.loc[
            local_sample[time].isin(mixed_times)
            | (
                local_sample[time].isin(control_only_times)
                & local_sample["D_local"].eq(0)
            )
        ].copy()
        or_training["_target_support"] = or_training[time].isin(mixed_times)
    else:
        or_training = target.copy()

    diagnostics = {
        "n_requested_event_rows": int(local_sample["D_local"].eq(1).sum()),
        "n_supported_event_rows": int(target["D_local"].eq(1).sum()),
        "n_unsupported_event_rows": n_unsupported,
        "n_unsupported_event_dates": int(len(treated_only_times)),
        "n_control_only_dates": int(len(control_only_times)),
        "n_auxiliary_or_control_rows": int(
            (
                or_training["D_local"].eq(0)
                & ~or_training["_target_support"]
            ).sum()
        ),
        "n_target_rows": int(len(target)),
        "n_or_training_rows": int(len(or_training)),
    }
    return target, or_training, diagnostics



def _odds_from_prob(
    p: np.ndarray,
    clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Convert probabilities to odds ratios, with optional clipping."""
    p = np.asarray(p, dtype=float)
    if clip is not None:
        p = np.clip(p, clip[0], clip[1])
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p / (1.0 - p)


def _clip_linear_index(xg: np.ndarray, lower: float = -700.0, upper: float = 700.0) -> np.ndarray:
    """Compatibility helper for logistic evaluation only.

    v0.6.0 no longer clips IPT estimating equations at +/-50. Logistic
    probabilities may be evaluated at machine-safe bounds without changing
    the score in the empirically relevant range.
    """
    return np.clip(np.asarray(xg, dtype=float), lower, upper)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x > 700):
        raise IPTConvergenceError("IPT linear index exceeds the floating-point exponential range.")
    return np.exp(x)


def _overlap_diagnostics(
    d: np.ndarray, e_hat: np.ndarray, odds: np.ndarray
) -> dict:
    """Observable local-overlap diagnostics for propensity-based fits."""
    d = np.asarray(d, dtype=float)
    e_hat = np.asarray(e_hat, dtype=float)
    odds = np.asarray(odds, dtype=float)
    control_odds = odds[d == 0]
    weight_sum = float(np.sum(control_odds))
    weight_sq_sum = float(np.sum(control_odds ** 2))
    ess = (
        weight_sum ** 2 / weight_sq_sum
        if weight_sq_sum > 0 and np.isfinite(weight_sq_sum)
        else np.nan
    )
    return {
        "propensity_min": float(np.min(e_hat)),
        "propensity_max": float(np.max(e_hat)),
        "control_odds_max": float(np.max(control_odds)),
        "control_odds_p99": float(np.quantile(control_odds, 0.99)),
        "control_weight_ess": float(ess),
    }


def _sandwich_diagnostics(influence: dict) -> dict:
    """Return compact, serializable diagnostics for a stacked sandwich."""
    covariance = np.asarray(influence["covariance"], dtype=float)
    return {
        "sandwich_jacobian_rank": int(influence["jacobian_rank"]),
        "sandwich_jacobian_condition": float(influence["jacobian_condition"]),
        "sandwich_variance_uncorrected": float(covariance.reshape(-1)[0]),
        "sandwich_n_scores": int(influence["n_obs"]),
        "sandwich_n_clusters": int(influence["n_clusters"]),
    }


def _build_exog_from_fit(fit, data: pd.DataFrame) -> np.ndarray:
    """Re-evaluate a patsy design matrix on new data using a fitted model."""
    if isinstance(fit, LocalORResult):
        if len(fit.exog) == len(data):
            return np.asarray(fit.exog, dtype=float)
        rebuilt = patsy.build_design_matrices(
            [fit.design_info], data, return_type="dataframe"
        )[0]
        return np.asarray(rebuilt.loc[:, list(fit.column_names)], dtype=float)
    if isinstance(fit, LocalPSResult):
        if len(fit.exog) != len(data):
            raise ValueError(
                "A retained-rank nuisance matrix can only be reused on the "
                "local stack on which it was constructed."
            )
        return np.asarray(fit.exog, dtype=float)
    return np.asarray(
        patsy.build_design_matrices(
            [fit.model.data.design_info], data, return_type="dataframe"
        )[0],
        dtype=float,
    )


def _full_rank_columns(
    matrix: np.ndarray,
    column_names,
    *,
    reference_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...], np.ndarray]:
    """Retain a deterministic, original-order basis for a column space.

    The rank decision is made after column scaling and, when supplied,
    ``reference_matrix`` determines the estimable columns.  The latter is
    used for outcome regressions, whose coefficients must be identified from
    clean controls.  Returning the selected indices lets all nuisance moments
    use exactly the same retained columns.
    """
    matrix = np.asarray(matrix, dtype=float)
    ref = matrix if reference_matrix is None else np.asarray(reference_matrix, dtype=float)
    if matrix.ndim != 2 or ref.ndim != 2 or matrix.shape[1] != ref.shape[1]:
        raise ValueError("Nuisance design matrices must be two-dimensional with matching columns.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(ref)):
        raise ValueError("Nuisance design matrices contain non-finite values.")

    names = tuple(str(name) for name in column_names)
    if len(names) != matrix.shape[1]:
        raise ValueError("The number of nuisance column names does not match the design matrix.")

    scales = np.linalg.norm(ref, axis=0)
    usable = scales > np.finfo(float).eps
    scaled = np.zeros_like(ref, dtype=float)
    scaled[:, usable] = ref[:, usable] / scales[usable]
    tol = max(ref.shape) * np.finfo(float).eps * 100.0
    keep: list[int] = []
    current_rank = 0
    for j in range(ref.shape[1]):
        if not usable[j]:
            continue
        candidate = scaled[:, keep + [j]]
        new_rank = int(np.linalg.matrix_rank(candidate, tol=tol))
        if new_rank > current_rank:
            keep.append(j)
            current_rank = new_rank
    if not keep:
        raise ValueError("The local nuisance design has no estimable column.")
    dropped = tuple(names[j] for j in range(len(names)) if j not in keep)
    retained = tuple(names[j] for j in keep)
    indices = np.asarray(keep, dtype=int)
    return matrix[:, indices], retained, dropped, indices


def _outcome_design(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    *,
    weights: Optional[np.ndarray] = None,
    add_time_fe: bool = True,
    time_varying_slopes: bool = False,
) -> tuple[LocalORResult, np.ndarray]:
    """Fit a full-rank control outcome regression on one local stack.

    With ``time_varying_slopes=True`` the covariates are interacted with calendar
    time (saturated ``C(time):X``), so the outcome-trend slope may differ by period.
    """
    formula = _build_formula(
        "outcome_local", time, controls, add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )
    y_mat, X_df = patsy.dmatrices(formula, local_sample, return_type="dataframe")
    y = np.asarray(y_mat, dtype=float).reshape(-1)
    X_full = np.asarray(X_df, dtype=float)
    control = local_sample["D_local"].to_numpy(dtype=float) == 0
    if not np.any(control):
        raise ValueError("No clean controls available for outcome regression.")
    X, retained, dropped, _ = _full_rank_columns(
        X_full, X_df.columns, reference_matrix=X_full[control]
    )
    X0 = X[control]
    y0 = y[control]
    if weights is None:
        root_w = np.ones_like(y0)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y.shape:
            raise ValueError("Outcome-regression weights must match the local stack.")
        w0 = w[control]
        if np.any(~np.isfinite(w0)) or np.any(w0 <= 0):
            raise ValueError("Outcome-regression weights must be finite and positive.")
        root_w = np.sqrt(w0)
    beta, _, rank, _ = np.linalg.lstsq(X0 * root_w[:, None], y0 * root_w, rcond=None)
    if rank != X.shape[1] or not np.all(np.isfinite(beta)):
        raise NuisanceConvergenceError(
            "The retained local outcome-regression basis is not full rank."
        )
    fit = LocalORResult(
        params=np.asarray(beta, dtype=float),
        exog=X,
        design_info=X_df.design_info,
        formula=formula,
        column_names=retained,
        dropped_columns=dropped,
    )
    return fit, X @ fit.params


# ---------------------------------------------------------------------------
# Propensity-score estimation
# ---------------------------------------------------------------------------

class IPTConvergenceError(NuisanceConvergenceError):
    """Raised when IPT fails its numerical balancing conditions."""


def _fit_ps_logit(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    ps_clip: Tuple[float, float],
    add_time_fe: bool = False,
    time_varying_slopes: bool = False,
) -> tuple:
    """Fit a local propensity score via maximum-likelihood logistic regression.

    Returns
    -------
    fit : GLMResultsWrapper or None
        Fitted model object (``None`` when no covariates are present).
    e_hat : np.ndarray
        Clipped propensity-score predictions.
    """
    formula = _build_formula(
        "D_local",
        time,
        controls,
        add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )
    y_mat, X_df = patsy.dmatrices(formula, local_sample, return_type="dataframe")
    d = np.asarray(y_mat, dtype=float).reshape(-1)
    X_full = np.asarray(X_df, dtype=float)
    X, retained, dropped, _ = _full_rank_columns(X_full, X_df.columns)
    fit_glm = sm.GLM(d, X, family=sm.families.Binomial()).fit(disp=False)
    if (
        not bool(getattr(fit_glm, "converged", True))
        or not np.all(np.isfinite(np.asarray(fit_glm.params, dtype=float)))
    ):
        raise NuisanceConvergenceError(
            "The local logit propensity model did not converge."
        )
    raw_e = expit(X @ np.asarray(fit_glm.params, dtype=float))
    if not np.all(np.isfinite(raw_e)):
        raise NuisanceConvergenceError(
            "The local logit propensity model produced non-finite predictions."
        )
    e_hat = np.clip(raw_e, ps_clip[0], ps_clip[1]) if ps_clip is not None else raw_e
    fit = LocalPSResult(
        params=np.asarray(fit_glm.params, dtype=float),
        exog=X,
        design_info=None,
        formula=formula,
        method="logit",
        success=True,
        optimizer_result=fit_glm,
        column_names=retained,
        dropped_columns=dropped,
    )
    return fit, e_hat


def _fit_ps_ipt(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    ps_clip: Tuple[float, float],
    add_time_fe: bool = False,
    time_varying_slopes: bool = False,
) -> tuple:
    """Fit a local propensity score via inverse probability tilting (IPT).

    IPT minimises the empirical Kullback–Leibler divergence between the
    treated and reweighted control distributions, yielding a first-order
    efficient moment-condition estimator (Graham et al. 2012).

    Returns
    -------
    fit : LocalPSResult
        Fitted IPT model container.
    e_hat : np.ndarray
        Clipped propensity-score predictions.
    """
    formula = _build_formula(
        "D_local",
        time,
        controls,
        add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )

    if formula == "D_local ~ 1":
        d = np.asarray(local_sample["D_local"], dtype=float)
        p = float(np.mean(d))
        if ps_clip is not None:
            p = np.clip(p, ps_clip[0], ps_clip[1])
        else:
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
        fit = LocalPSResult(
            params=np.array([np.log(p / (1.0 - p))]),
            exog=np.ones((len(local_sample), 1), dtype=float),
            design_info=None,
            formula=formula,
            method="ipt",
            success=True,
            balance_error=0.0,
            column_names=("Intercept",),
            dropped_columns=(),
        )
        return fit, np.full(len(local_sample), p, dtype=float)

    y_mat, X_df = patsy.dmatrices(formula, local_sample, return_type="dataframe")
    d = np.asarray(y_mat).reshape(-1).astype(float)
    X_full = np.asarray(X_df, dtype=float)
    X, retained, dropped, _ = _full_rank_columns(X_full, X_df.columns)

    # Warm-start from logistic regression; fall back to intercept-only on failure
    pbar = np.clip(np.mean(d), ps_clip[0] if ps_clip else 1e-6, ps_clip[1] if ps_clip else 1-1e-6)
    x0 = np.zeros(X.shape[1], dtype=float)
    x0[0] = np.log(pbar / (1.0 - pbar))
    try:
        glm_fit = sm.GLM(d, X, family=sm.families.Binomial()).fit(disp=False)
        if np.all(np.isfinite(glm_fit.params)):
            x0 = np.asarray(glm_fit.params, dtype=float)
    except Exception:
        pass

    def obj(gamma: np.ndarray) -> float:
        eta = np.asarray(X @ gamma, dtype=float)
        try:
            ee = _safe_exp(eta)
        except IPTConvergenceError:
            return np.inf
        return -float(np.mean(d * eta - (1.0 - d) * ee))

    def grad(gamma: np.ndarray) -> np.ndarray:
        eta = np.asarray(X @ gamma, dtype=float)
        try:
            ee = _safe_exp(eta)
        except IPTConvergenceError:
            return np.full(X.shape[1], np.nan)
        score = X * (d - (1.0 - d) * ee)[:, None]
        return -np.mean(score, axis=0)

    opt = optimize.minimize(
        obj, x0=x0, jac=grad, method="BFGS",
        options={"gtol": 1e-10, "maxiter": 5000},
    )
    if not opt.success or not np.all(np.isfinite(opt.x)):
        opt = optimize.minimize(
            obj, x0=x0, jac=grad, method="L-BFGS-B",
            options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 5000},
        )
    if not np.all(np.isfinite(opt.x)):
        raise IPTConvergenceError(
            f"Local IPT propensity estimation failed: {opt.message}"
        )

    gamma_hat = np.asarray(opt.x, dtype=float)
    balance_error = float(np.max(np.abs(grad(gamma_hat))))
    if not np.isfinite(balance_error) or balance_error > 1e-9:
        def score(gamma: np.ndarray) -> np.ndarray:
            return -grad(gamma)

        def jac(gamma: np.ndarray) -> np.ndarray:
            ee = _safe_exp(X @ gamma)
            return -(X.T @ (((1.0 - d) * ee)[:, None] * X)) / len(d)

        root = optimize.least_squares(
            score,
            gamma_hat,
            jac=jac,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=5000,
        )
        if np.all(np.isfinite(root.x)):
            gamma_hat = np.asarray(root.x, dtype=float)
            balance_error = float(np.max(np.abs(grad(gamma_hat))))
            opt = root
    if not np.isfinite(balance_error) or balance_error > 1e-9:
        raise IPTConvergenceError(
            "Local IPT propensity estimation did not satisfy the balancing "
            f"moments (maximum absolute residual={balance_error:.3e})."
        )
    # For IPT the natural weight is exp(gamma'X); we expose expit for
    # overlap diagnostics only. When ps_clip is None (default), return
    # exp(xg) directly as the effective "propensity odds" so that the
    # point estimate and the influence function use the same quantity.
    raw_e = expit(X @ gamma_hat)
    e_hat = np.clip(raw_e, ps_clip[0], ps_clip[1]) if ps_clip is not None else raw_e
    fit = LocalPSResult(
        params=gamma_hat,
        exog=X,
        design_info=X_df.design_info,
        formula=formula,
        method="ipt",
        success=True,
        optimizer_result=opt,
        balance_error=balance_error,
        column_names=retained,
        dropped_columns=dropped,
    )
    return fit, e_hat


# ---------------------------------------------------------------------------
# Outcome regression
# ---------------------------------------------------------------------------

def _fit_or_ols_controls(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    add_time_fe: bool = True,
    time_varying_slopes: bool = False,
) -> tuple:
    """OLS outcome regression on clean-control observations.

    Returns ``(fitted_model, predicted_values_for_full_sample)``.
    """
    return _outcome_design(
        local_sample, time, controls, weights=None, add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )


def _fit_or_wls_controls(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    weights: np.ndarray,
    add_time_fe: bool = True,
    time_varying_slopes: bool = False,
) -> tuple:
    """IPT-weighted least-squares outcome regression on clean-control observations.

    Used by the ``'ipt'`` and ``'dr-ipt'`` estimators to match the weighted
    covariate distribution of the treated group (Sant'Anna--Zhao improved logic).
    """
    return _outcome_design(
        local_sample,
        time,
        controls,
        weights=np.asarray(weights, dtype=float),
        add_time_fe=add_time_fe,
        time_varying_slopes=time_varying_slopes,
    )


# ---------------------------------------------------------------------------
# Semiparametric estimators
# ---------------------------------------------------------------------------


def _compute_ra(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    controls: List[str],
    compute_influence: bool,
    or_time_varying_slopes: bool = False,
    or_training_sample: Optional[pd.DataFrame] = None,
) -> dict:
    """Regression-adjustment ATT-h estimator.

    .. math::
        \\hat{\\tau}_h^{RA} = \\frac{1}{N_1}
            \\sum_{i: D_i=1} \\bigl(Y_i - \\hat{m}_0(X_i)\\bigr)

    where :math:`\\hat{m}_0` is estimated on clean controls only.
    """
    # RA imputation requires a clean control in every time cell used: the
    # outcome regression carries a calendar-time fixed effect that is
    # unidentified for a treated-only cell, so prediction on such a cell fails.
    # Restrict to time cells with at least one clean control (treated units in
    # control-less cells cannot be imputed and are excluded from the RA
    # average). This mirrors the RA-LP-DiD restriction in the lpdid module and
    # is required because build_local_sample now keeps control-less cells.
    _hc = local_sample.loc[local_sample["D_local"] == 0].groupby(time).size()
    local_sample = local_sample.loc[
        local_sample[time].isin(_hc.loc[_hc > 0].index)
    ].copy()
    training_sample = (
        local_sample.copy()
        if or_training_sample is None else or_training_sample.copy()
    )
    if "_target_support" not in training_sample:
        training_sample["_target_support"] = True

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    fit_or, _ = _fit_or_ols_controls(
        training_sample, time, controls,
        time_varying_slopes=or_time_varying_slopes,
    )
    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    m0_hat = z_all @ beta_hat
    resid = y - m0_hat
    mu1_hat = float(np.mean(resid[d == 1]))

    y_train = np.asarray(training_sample["outcome_local"], dtype=float)
    d_train = np.asarray(training_sample["D_local"], dtype=float)
    target_train = np.asarray(
        training_sample["_target_support"], dtype=bool
    ).astype(float)
    z_train = _build_exog_from_fit(fit_or, training_sample)
    cluster_ids = training_sample[unit].to_numpy()

    n_params = int(len(beta_hat) + 1)
    diagnostics = {
        "or_retained_columns": tuple(fit_or.column_names),
        "or_dropped_columns": tuple(fit_or.dropped_columns),
        "n_target_rows": int(len(local_sample)),
        "n_or_training_rows": int(len(training_sample)),
        "n_auxiliary_or_control_rows": int(
            (
                training_sample["D_local"].eq(0)
                & ~training_sample["_target_support"].astype(bool)
            ).sum()
        ),
    }
    if not compute_influence:
        return {
            "estimate": mu1_hat, "se": np.nan,
            "n_moment_parameters": n_params,
            "n_inference_rows": int(len(training_sample)),
            "n_inference_clusters": int(training_sample[unit].nunique()),
            **diagnostics,
        }

    theta_hat = np.concatenate([beta_hat, np.array([mu1_hat])])

    def moments_obs(theta):
        pb = z_train.shape[1]
        beta, mu1 = theta[:pb], theta[pb]
        resid_theta = y_train - z_train @ beta
        m_beta = (
            ((1.0 - d_train))[:, None]
            * z_train
            * resid_theta[:, None]
        )
        m_mu1 = (
            target_train * d_train * (resid_theta - mu1)
        )[:, None]
        return np.column_stack([m_beta, m_mu1])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-1] = 1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": mu1_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "n_moment_parameters": n_params,
        "n_inference_rows": int(len(training_sample)),
        "n_inference_clusters": int(training_sample[unit].nunique()),
        **_sandwich_diagnostics(infl),
        **diagnostics,
    }


def _compute_ipw(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    controls: List[str],
    ps_method: str,
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
) -> dict:
    """IPW ATT-h estimator.

    .. math::
        \\hat{\\tau}_h^{IPW} = \\bar{Y}_1
            - \\frac{\\sum_{i:D_i=0} o_i Y_i}{\\sum_{i:D_i=0} o_i}

    where :math:`o_i = e(X_i) / (1 - e(X_i))` are propensity-score odds.
    The propensity score is estimated by logistic regression (``'generic'``)
    or IPT (``'improved'``).
    """
    if ps_with_time_fe:
        # A saturated time fixed effect is not finitely identified in a
        # calendar cell containing controls only. Such cells do not enter
        # the treated-entry ATT contrast and have population propensity zero.
        # Restrict propensity-based estimators to event-time cells containing
        # both treated entrants and clean controls.
        local_sample = _restrict_to_treated_time_cells(local_sample, time)
    else:
        local_sample = _drop_all_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()

    if ps_method == "improved":
        fit_ps, e_hat = _fit_ps_ipt(
            local_sample,
            time,
            controls,
            ps_clip,
            add_time_fe=ps_with_time_fe,
            time_varying_slopes=ps_time_varying_slopes,
        )
        X = np.asarray(fit_ps.exog, dtype=float)
        gamma_hat = np.asarray(fit_ps.params, dtype=float)
        score_moment = "ipt"
    else:
        fit_ps, e_hat = _fit_ps_logit(
            local_sample,
            time,
            controls,
            ps_clip,
            add_time_fe=ps_with_time_fe,
            time_varying_slopes=ps_time_varying_slopes,
        )
        X = np.asarray(fit_ps.exog, dtype=float)
        gamma_hat = np.asarray(fit_ps.params, dtype=float)
        score_moment = "logit"

    # With clipping disabled by the formal API, logit and IPT odds are
    # evaluated as exp(X gamma), avoiding hidden probability truncation.
    odds = _safe_exp(X @ gamma_hat)
    mu1_hat = float(np.mean(y[d == 1]))
    lam = odds[d == 0]
    lam_sum = float(lam.sum())
    if lam_sum <= 1e-12:
        raise ValueError(
            "IPW control weight sum is effectively zero; "
            "local overlap is insufficient at this horizon."
        )
    mu0_hat = float(np.dot(lam, y[d == 0]) / lam_sum)
    tau_hat = mu1_hat - mu0_hat
    diagnostics = {
        "ipt_balance_error": (
            float(fit_ps.balance_error) if ps_method == "improved" else np.nan
        ),
        "ps_retained_columns": tuple(fit_ps.column_names),
        "ps_dropped_columns": tuple(fit_ps.dropped_columns),
        **_overlap_diagnostics(d, e_hat, odds),
    }
    n_params = int(len(gamma_hat) + 2)

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "n_moment_parameters": n_params,
            "n_inference_rows": int(len(local_sample)),
            "n_inference_clusters": int(local_sample[unit].nunique()),
            **diagnostics,
        }

    theta_hat = np.concatenate([gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pg = X.shape[1]
        gamma, mu1, mu0 = theta[:pg], theta[pg], theta[pg + 1]
        xg = _clip_linear_index(X @ gamma)
        pihat = (np.clip(expit(xg), ps_clip[0], ps_clip[1])
                 if ps_clip is not None else expit(xg))
        if score_moment == "ipt":
            # IPT estimating equation uses exp(gamma'X) as the tilting odds.
            # The ATT component uses the same effective odds as the point
            # estimate; with ps_clip=None this is exactly exp(gamma'X).
            ipt_odds_score = _safe_exp(xg)
            odds_theta = ipt_odds_score
            m_gamma = X * (d - (1.0 - d) * ipt_odds_score)[:, None]
            m_mu1 = (d * (y - mu1))[:, None]
            m_mu0 = ((1.0 - d) * odds_theta * (y - mu0))[:, None]
        else:
            m_gamma = X * (d - pihat)[:, None]
            odds_theta = _safe_exp(xg)
            m_mu1 = (d * (y - mu1))[:, None]
            m_mu0 = ((1.0 - d) * odds_theta * (y - mu0))[:, None]
        return np.column_stack([m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "n_moment_parameters": n_params,
        "n_inference_rows": int(len(local_sample)),
        "n_inference_clusters": int(local_sample[unit].nunique()),
        **_sandwich_diagnostics(infl),
        **diagnostics,
    }


def _compute_dr_generic(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    ps_controls: List[str],
    or_controls: List[str],
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
    or_time_varying_slopes: bool = False,
) -> dict:
    """Doubly robust ATT-h estimator (generic logit + OLS specification).

    The DR correction applies IPW reweighting to the outcome-regression
    residuals, yielding consistency when *either* the propensity score *or*
    the outcome regression is correctly specified.

    .. math::
        \\hat{\\tau}_h^{DR} = \\frac{1}{N_1}\\sum_{i:D_i=1}(Y_i - \\hat{m}_0(X_i))
            - \\frac{\\sum_{i:D_i=0} o_i (Y_i - \\hat{m}_0(X_i))}{\\sum_{i:D_i=0} o_i}
    """
    if ps_with_time_fe:
        # A saturated time fixed effect is not finitely identified in a
        # calendar cell containing controls only. Such cells do not enter
        # the treated-entry ATT contrast and have population propensity zero.
        # Restrict propensity-based estimators to event-time cells containing
        # both treated entrants and clean controls.
        local_sample = _restrict_to_treated_time_cells(local_sample, time)
    else:
        local_sample = _drop_all_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()

    ps_controls = list(ps_controls or [])
    or_controls = list(or_controls or [])

    fit_ps, e_hat = _fit_ps_logit(
        local_sample,
        time,
        ps_controls,
        ps_clip,
        add_time_fe=ps_with_time_fe,
        time_varying_slopes=ps_time_varying_slopes,
    )
    fit_or, m0_hat = _fit_or_ols_controls(
        local_sample, time, or_controls, time_varying_slopes=or_time_varying_slopes
    )
    X = np.asarray(fit_ps.exog, dtype=float)
    gamma_hat = np.asarray(fit_ps.params, dtype=float)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    resid = y - m0_hat
    odds = _safe_exp(X @ gamma_hat)
    mu1_hat = float(np.mean(resid[d == 1]))
    lam = odds[d == 0]
    lam_sum = float(lam.sum())
    if lam_sum <= 1e-12:
        raise ValueError(
            "DR-IPW control weight sum is effectively zero; "
            "local overlap is insufficient at this horizon."
        )
    mu0_hat = float(np.dot(lam, resid[d == 0]) / lam_sum)
    tau_hat = mu1_hat - mu0_hat
    diagnostics = {
        "ps_retained_columns": tuple(fit_ps.column_names),
        "ps_dropped_columns": tuple(fit_ps.dropped_columns),
        "or_retained_columns": tuple(fit_or.column_names),
        "or_dropped_columns": tuple(fit_or.dropped_columns),
        **_overlap_diagnostics(d, e_hat, odds),
    }
    n_params = int(len(beta_hat) + len(gamma_hat) + 2)

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "n_moment_parameters": n_params,
            "n_inference_rows": int(len(local_sample)),
            "n_inference_clusters": int(local_sample[unit].nunique()),
            **diagnostics,
        }

    z_all = _build_exog_from_fit(fit_or, local_sample)
    theta_hat = np.concatenate([beta_hat, gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pb, pg = z_all.shape[1], X.shape[1]
        beta, gamma = theta[:pb], theta[pb:pb + pg]
        mu1, mu0 = theta[pb + pg], theta[pb + pg + 1]
        xg = X @ gamma
        pihat = expit(xg)
        odds_theta = _safe_exp(xg)
        resid_theta = y - z_all @ beta
        m_beta = ((1.0 - d))[:, None] * z_all * resid_theta[:, None]
        m_gamma = X * (d - pihat)[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
        m_mu0 = ((1.0 - d) * odds_theta * (resid_theta - mu0))[:, None]
        return np.column_stack([m_beta, m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "n_moment_parameters": n_params,
        "n_inference_rows": int(len(local_sample)),
        "n_inference_clusters": int(local_sample[unit].nunique()),
        **_sandwich_diagnostics(infl),
        **diagnostics,
    }


def _compute_dr_improved(
    local_sample: pd.DataFrame,
    unit: str,
    time: str,
    ps_controls: List[str],
    or_controls: List[str],
    ps_clip: Tuple[float, float],
    ps_with_time_fe: bool,
    ps_time_varying_slopes: bool,
    compute_influence: bool,
    or_time_varying_slopes: bool = False,
) -> dict:
    """Improved DR ATT-h estimator in the Sant'Anna--Zhao style.

    The improved specification uses inverse probability tilting (IPT) for the
    local propensity score and IPT-weighted least squares for the clean-control
    untreated-outcome regression. This is the only improved DR specification
    exposed by the package.
    """
    if ps_with_time_fe:
        # A saturated time fixed effect is not finitely identified in a
        # calendar cell containing controls only. Such cells do not enter
        # the treated-entry ATT contrast and have population propensity zero.
        # Restrict propensity-based estimators to event-time cells containing
        # both treated entrants and clean controls.
        local_sample = _restrict_to_treated_time_cells(local_sample, time)
    else:
        local_sample = _drop_all_treated_time_cells(local_sample, time)

    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()
    ps_controls = list(ps_controls or [])
    or_controls = list(or_controls or [])

    # First nuisance: local IPT propensity score.
    fit_ps, e_hat = _fit_ps_ipt(
        local_sample,
        time,
        ps_controls,
        ps_clip,
        add_time_fe=ps_with_time_fe,
        time_varying_slopes=ps_time_varying_slopes,
    )
    X = np.asarray(fit_ps.exog, dtype=float)
    gamma_hat = np.asarray(fit_ps.params, dtype=float)
    odds = _safe_exp(X @ gamma_hat)

    # Second nuisance: IPT-weighted outcome regression on clean controls.
    fit_or, m0_hat = _fit_or_wls_controls(
        local_sample,
        time,
        or_controls,
        weights=odds,
        add_time_fe=True,
        time_varying_slopes=or_time_varying_slopes,
    )

    resid = y - m0_hat
    mu1_hat = float(np.mean(resid[d == 1]))
    lam = odds[d == 0]
    lam_sum = float(lam.sum())
    if lam_sum <= 1e-12:
        raise ValueError(
            "Improved DR control residual weight sum is effectively zero; "
            "local overlap is insufficient at this horizon."
        )
    mu0_hat = float(np.dot(lam, resid[d == 0]) / lam_sum)
    tau_hat = mu1_hat - mu0_hat

    # The formal improved estimator satisfies IPT = DRIPT whenever the
    # outcome-regression basis is nested in the IPT basis. Verify that
    # identity instead of trusting the optimizer status alone.
    z_all = _build_exog_from_fit(fit_or, local_sample)
    rank_x = np.linalg.matrix_rank(X)
    nested_or_basis = (
        np.linalg.matrix_rank(np.column_stack([X, z_all])) == rank_x
    )
    tau_ipt = float(
        np.mean(y[d == 1])
        - np.dot(odds[d == 0], y[d == 0]) / np.sum(odds[d == 0])
    )
    beta_hat = np.asarray(fit_or.params, dtype=float)
    nested_difference = abs(tau_hat - tau_ipt)
    if nested_or_basis:
        treated_z_mean = np.mean(z_all[d == 1], axis=0)
        weighted_control_z_mean = np.average(
            z_all[d == 0], axis=0, weights=odds[d == 0]
        )
        nested_basis_imbalance = (
            treated_z_mean - weighted_control_z_mean
        )
        nested_basis_balance_error = float(
            np.max(np.abs(nested_basis_imbalance))
        )
        # Under exact IPT balance, tau_IPT == tau_DRIPT. In floating-point
        # arithmetic the observed discrepancy is bounded by the fitted OR
        # coefficients times the residual imbalance in its retained basis.
        # Certify that measured numerical bound rather than imposing a
        # scale-free absolute cutoff on the treatment-effect estimate.
        identity_roundoff = (
            100.0
            * np.finfo(float).eps
            * max(
                1.0,
                abs(tau_hat),
                abs(tau_ipt),
                float(np.max(np.abs(y))),
            )
        )
        identity_tolerance = float(
            max(
                1e-10,
                10.0
                * np.dot(
                    np.abs(beta_hat),
                    np.abs(nested_basis_imbalance),
                )
                + identity_roundoff,
            )
        )
    else:
        nested_basis_balance_error = np.nan
        identity_tolerance = np.nan
    if (
        nested_or_basis
        and (
            not np.isfinite(identity_tolerance)
            or nested_difference > identity_tolerance
        )
    ):
        raise IPTConvergenceError(
            "The IPT--DRIPT nesting identity failed despite nested nuisance "
            "bases: the discrepancy exceeds the tolerance implied by the "
            "measured residual balance "
            f"(difference={nested_difference:.3e}, "
            f"tolerance={identity_tolerance:.3e})."
        )
    diagnostics = {
        "ipt_balance_error": float(fit_ps.balance_error),
        "ipt_dript_nested_basis": bool(nested_or_basis),
        "ipt_dript_nested_difference": float(nested_difference),
        "ipt_nested_or_balance_error": float(nested_basis_balance_error),
        "ipt_dript_identity_tolerance": float(identity_tolerance),
        "ipt_retained_columns": tuple(fit_ps.column_names),
        "ipt_dropped_columns": tuple(fit_ps.dropped_columns),
        "or_retained_columns": tuple(fit_or.column_names),
        "or_dropped_columns": tuple(fit_or.dropped_columns),
        **_overlap_diagnostics(d, e_hat, odds),
    }
    n_params = int(len(beta_hat) + len(gamma_hat) + 2)

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
            "n_moment_parameters": n_params,
            "n_inference_rows": int(len(local_sample)),
            "n_inference_clusters": int(local_sample[unit].nunique()),
            **diagnostics,
        }

    theta_hat = np.concatenate([beta_hat, gamma_hat, np.array([mu1_hat, mu0_hat])])

    def moments_obs(theta):
        pb, pg = z_all.shape[1], X.shape[1]
        beta, gamma = theta[:pb], theta[pb:pb + pg]
        mu1, mu0 = theta[pb + pg], theta[pb + pg + 1]
        xg = _clip_linear_index(X @ gamma)
        resid_theta = y - z_all @ beta

        # IPT weighted least-squares score on clean controls.
        ipt_odds_score = _safe_exp(xg)
        odds_theta = ipt_odds_score
        m_beta = (((1.0 - d) * odds_theta)[:, None]) * z_all * resid_theta[:, None]

        # IPT propensity-score moment.
        m_gamma = X * (d - (1.0 - d) * ipt_odds_score)[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
        m_mu0 = ((1.0 - d) * odds_theta * (resid_theta - mu0))[:, None]
        return np.column_stack([m_beta, m_gamma, m_mu1, m_mu0])

    def target_grad(theta):
        g = np.zeros_like(theta)
        g[-2] = 1.0
        g[-1] = -1.0
        return g

    infl = stacked_influence(
        theta_hat, moments_obs, cluster_ids, target_grad=target_grad
    )
    psi = infl["psi"].reshape(-1)
    return {
        "estimate": tau_hat,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(
            psi, index=pd.Index(infl["cluster_labels"]), dtype=float
        ),
        "n_moment_parameters": n_params,
        "n_inference_rows": int(len(local_sample)),
        "n_inference_clusters": int(local_sample[unit].nunique()),
        **_sandwich_diagnostics(infl),
        **diagnostics,
    }


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class DRLPDID:
    """Estimate a formally defined DR-LP-DiD event-study path.

    ``estimation_method`` selects ``'ra'``, ``'ipw'``, ``'ipt'``,
    ``'dr-ipw'``, or ``'dr-ipt'`` on one common horizon-specific stack.
    ``pre_window`` and ``post_window`` follow the public ``diff-diff`` LP-DiD
    convention and default to 2 and 0. Passing both as ``None`` evaluates the
    largest data-supported event-time grid. ``horizons=(min_h, max_h)``
    remains as an explicit compatibility alias for older notebooks.
    For non-absorbing paths, ``non_absorbing`` is ``'first_entry'`` or
    ``'effect_stabilization'``.

    ``inference='cluster'`` reports the stacked M-estimation cluster sandwich;
    ``'multiplier'`` additionally reports simultaneous sup-t bands; and
    ``'cluster_bootstrap'`` fully re-estimates paired cluster samples.
    """

    def __init__(
        self,
        estimation_method: str = "dr-ipt",
        pre_window: Optional[int] = 2,
        post_window: Optional[Union[int, Tuple[int, int]]] = 0,
        control_group: str = "clean",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
        non_absorbing: Optional[str] = None,
        stabilization_window: Optional[int] = None,
        df_convention: str = "cluster",
        *,
        inference: str = "cluster",
        small_sample_correction: str = "cr1",
        bootstrap_weights: str = "rademacher",
        n_bootstrap: int = 999,
        min_bootstrap_success: float = 0.90,
        seed: Optional[int] = None,
        or_slope_structure: str = "common",
        history_policy: str = "diff_diff",
        left_censoring: str = "keep_as_control_after_L",
        or_training_support: str = "method_default",
        unsupported_event_policy: str = "drop",
        pre_aggregation=None,
        post_aggregation=None,
        # Explicit legacy spellings retained for reproducibility of notebooks
        # created before the literature-standard public surface.
        design: Optional[str] = None,
        horizons: Optional[Tuple[int, int]] = None,
        aggregate_window: Optional[Tuple[int, int]] = None,
        outcome_change_lags: int = 0,
    ) -> None:
        """Create a semiparametric DR-LP-DiD estimator.

        The public design vocabulary follows ``LPDID``. There is deliberately
        no ``reweight`` flag: every method targets the eligible-entry ATT on
        the supported semiparametric stack, while ``estimation_method`` selects
        the nuisance-score construction.
        """
        methods = {
            "ra": ("ra", None),
            "ipw": ("ipw", "generic"),
            "ipt": ("ipw", "improved"),
            "dr-ipw": ("dr", "generic"),
            "dr-ipt": ("dr", "improved"),
        }
        em = str(estimation_method).lower().replace("_", "-").replace(" ", "")
        if em not in methods:
            raise ValueError(
                "estimation_method must be one of "
                "{'ra','ipw','ipt','dr-ipw','dr-ipt'}."
            )
        self.estimation_method = em
        self._family, self._ps_method = methods[em]
        requested_or_support = str(or_training_support).lower()
        if requested_or_support not in {
            "method_default", "event_dates", "all_clean_controls"
        }:
            raise ValueError(
                "or_training_support must be 'method_default', "
                "'event_dates', or 'all_clean_controls'."
            )
        if requested_or_support == "method_default":
            # Dube-style RA learns common slopes from every eligible clean
            # control. Propensity-based methods remain on the formal local
            # event-date stack because the PS is undefined on control-only
            # calendar cells.
            requested_or_support = (
                "all_clean_controls" if self._family == "ra" else "event_dates"
            )
        if requested_or_support == "all_clean_controls" and self._family != "ra":
            raise NotImplementedError(
                "or_training_support='all_clean_controls' is currently "
                "certified only for estimation_method='ra'. DR-IPW and "
                "DR-IPT keep both nuisance scores on event dates until the "
                "auxiliary-OR influence function is separately certified."
            )
        self.or_training_support = requested_or_support
        self.unsupported_event_policy = str(unsupported_event_policy).lower()
        if self.unsupported_event_policy not in {"drop", "error"}:
            raise ValueError(
                "unsupported_event_policy must be 'drop' or 'error'."
            )
        self.dr_method = self._ps_method
        if rank_deficient_action not in {"warn", "error", "silent"}:
            raise ValueError(
                "rank_deficient_action must be 'warn', 'error', or 'silent'."
            )
        if df_convention != "cluster":
            raise NotImplementedError(
                "DRLPDID currently certifies df_convention='cluster' only."
            )
        self.rank_deficient_action = rank_deficient_action
        self.df_convention = df_convention

        legacy_design = None if design is None else str(design).lower()
        if legacy_design not in {None, "absorbing", "switching"}:
            raise ValueError("design must be 'absorbing' or 'switching'.")
        non_absorbing_mode = (
            None if non_absorbing is None else str(non_absorbing).lower()
        )
        if non_absorbing_mode not in {
            None, "first_entry", "effect_stabilization"
        }:
            raise ValueError(
                "non_absorbing must be None, 'first_entry', or "
                "'effect_stabilization'."
            )
        if legacy_design == "switching" and non_absorbing_mode is None:
            non_absorbing_mode = "effect_stabilization"
        if legacy_design == "absorbing" and non_absorbing_mode is not None:
            raise ValueError(
                "design='absorbing' conflicts with a non_absorbing mode."
            )
        self.non_absorbing = non_absorbing_mode
        self.design = "switching" if non_absorbing_mode is not None else "absorbing"

        requested_control = str(control_group).lower()
        if self.design == "absorbing":
            aliases = {"clean": "not_yet_treated", "not_yet_treated": "not_yet_treated",
                       "never_treated": "never_treated"}
            if requested_control not in aliases:
                raise ValueError(
                    "For absorbing treatment, control_group must be 'clean', "
                    "'not_yet_treated', or 'never_treated'."
                )
            self.clean_control = aliases[requested_control]
            if self.clean_control not in {"not_yet_treated", "never_treated"}:
                raise ValueError(
                    "For design='absorbing', control_group must be "
                    "'not_yet_treated' or 'never_treated'."
                )
            if stabilization_window is not None:
                raise ValueError(
                    "stabilization_window is only defined for design='switching'."
                )
            self.effect_stabilization = None
        else:
            if requested_control != "clean":
                raise ValueError(
                    "Non-absorbing designs use their formally defined clean "
                    "controls; set control_group='clean'."
                )
            if non_absorbing_mode == "effect_stabilization" and stabilization_window is None:
                raise ValueError(
                    "non_absorbing='effect_stabilization' requires "
                    "stabilization_window=L."
                )
            if non_absorbing_mode == "first_entry":
                if stabilization_window is not None:
                    raise ValueError(
                        "stabilization_window does not apply to "
                        "non_absorbing='first_entry'."
                    )
                self.clean_control = "first_entry"
                self.effect_stabilization = None
            else:
                if isinstance(stabilization_window, bool):
                    raise ValueError("stabilization_window must be a positive integer.")
                L = int(stabilization_window)
                if L != stabilization_window or L < 1:
                    raise ValueError("stabilization_window must be a positive integer.")
                self.clean_control = "stabilized"
                self.effect_stabilization = L
        self.control_group = requested_control
        self.cluster = cluster
        self.history_policy = str(history_policy).lower()
        if self.history_policy not in {"observed", "diff_diff"}:
            raise ValueError("history_policy must be 'observed' or 'diff_diff'.")
        self.left_censoring = str(left_censoring).lower()
        if self.left_censoring == "keep":
            self.left_censoring = "keep_as_control_after_l"
        if self.left_censoring not in {
            "error", "drop", "keep_as_control_after_l"
        }:
            raise ValueError(
                "left_censoring must be 'error', 'drop', or "
                "'keep_as_control_after_L'."
            )

        legacy_scalar_window = post_window if isinstance(post_window, tuple) else None
        if legacy_scalar_window is not None and aggregate_window is not None:
            raise ValueError(
                "Use aggregate_window or the legacy tuple-valued post_window, not both."
            )
        standard_window_defaults = pre_window == 2 and post_window == 0
        if horizons is not None and not (
            standard_window_defaults or isinstance(post_window, tuple)
        ):
            raise ValueError(
                "Use pre_window/post_window or the legacy horizons argument, not both."
            )
        if horizons is not None:
            try:
                raw_h_lo, raw_h_hi = horizons[0], horizons[1]
                h_lo, h_hi = int(raw_h_lo), int(raw_h_hi)
            except Exception as exc:
                raise ValueError("horizons must be a pair (min_h, max_h).") from exc
            if (
                isinstance(raw_h_lo, bool) or isinstance(raw_h_hi, bool)
                or h_lo != raw_h_lo or h_hi != raw_h_hi
            ):
                raise ValueError("horizons must contain two integers.")
            if h_lo > -1 or h_hi < 0 or h_hi < h_lo:
                raise ValueError(
                    "horizons must include the base h=-1 and at least h=0."
                )
            self.max_pre, self.max_post = abs(h_lo), h_hi
            self._automatic_aggregates = False
        else:
            def _window(value, name):
                if value is None:
                    return None
                if isinstance(value, bool) or int(value) != value or int(value) < 0:
                    raise ValueError(f"{name} must be a non-negative integer or None.")
                return int(value)
            self.max_pre = _window(pre_window, "pre_window")
            self.max_post = _window(
                None if legacy_scalar_window is not None else post_window,
                "post_window",
            )
            self._automatic_aggregates = True
        self.base_period = -1
        self.anticipation = 0

        requested_aggregate_window = aggregate_window or legacy_scalar_window
        if requested_aggregate_window is None:
            self.policy_window = None
        else:
            try:
                raw_lo, raw_hi = requested_aggregate_window[0], requested_aggregate_window[1]
                lo, hi = int(raw_lo), int(raw_hi)
            except Exception as exc:
                raise ValueError("aggregate_window must be a pair (lo, hi).") from exc
            if (
                isinstance(raw_lo, bool)
                or isinstance(raw_hi, bool)
                or lo != raw_lo
                or hi != raw_hi
            ):
                raise ValueError("aggregate_window must contain two integers.")
            if lo < 0 or hi < lo or (self.max_post is not None and hi > self.max_post):
                raise ValueError(
                    "aggregate_window must satisfy 0 <= lo <= hi <= post window."
                )
            self.policy_window = (lo, hi)

        self.or_slope_structure = str(or_slope_structure).lower()
        if self.or_slope_structure not in {"common", "calendar_interacted"}:
            raise ValueError(
                "or_slope_structure must be 'common' or 'calendar_interacted'."
            )
        self.or_time_varying_slopes = (
            self.or_slope_structure == "calendar_interacted"
        )
        if self.policy_window is not None and post_aggregation is None:
            raise ValueError(
                "aggregate_window changes the scalar estimand; explicitly set "
                "post_aggregation='pooled', 'equal_horizon', "
                "'event_weighted', or 'all'."
            )
        self.pre_aggregations = normalize_aggregations(
            pre_aggregation,
            name="pre_aggregation",
            default=(("pooled",) if self._automatic_aggregates else ()),
        )
        self.post_aggregations = normalize_aggregations(
            post_aggregation,
            name="post_aggregation",
            default=(("pooled",) if self._automatic_aggregates else ()),
        )
        self.pre_aggregation = (
            self.pre_aggregations[0]
            if len(self.pre_aggregations) == 1 else (
                "all" if len(self.pre_aggregations) == 3 else self.pre_aggregations
            )
        )
        self.post_aggregation = (
            self.post_aggregations[0]
            if len(self.post_aggregations) == 1 else (
                "all" if len(self.post_aggregations) == 3 else self.post_aggregations
            )
        )
        self.small_sample_correction = str(small_sample_correction).lower()
        if self.small_sample_correction not in {"none", "cr1"}:
            raise ValueError("small_sample_correction must be 'none' or 'CR1'.")

        self.inference = str(inference).lower()
        if self.inference not in {"cluster", "multiplier", "cluster_bootstrap"}:
            raise ValueError(
                "inference must be 'cluster', 'multiplier', or "
                "'cluster_bootstrap'."
            )
        if self.inference == "cluster_bootstrap":
            warnings.warn(
                "cluster_bootstrap is experimental in pydrlpdid 0.8.1. "
                "Use at least 999 replications for reported results and "
                "inspect bootstrap success diagnostics.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap:
            raise ValueError("n_bootstrap must be an integer.")
        self.n_bootstrap = int(n_bootstrap)
        if self.inference in {"multiplier", "cluster_bootstrap"} and self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1.")
        if self.inference in {"multiplier", "cluster_bootstrap"} and self.n_bootstrap < 99:
            warnings.warn(
                "Fewer than 99 bootstrap draws are suitable only for smoke "
                "testing, not reported inference.",
                UserWarning,
                stacklevel=2,
            )
        self.bootstrap_weights = str(bootstrap_weights).lower()
        if self.bootstrap_weights not in {"rademacher", "mammen", "webb"}:
            raise ValueError(
                "bootstrap_weights must be 'rademacher', 'mammen', or 'webb'."
            )
        self.min_bootstrap_success = float(min_bootstrap_success)
        if not 0.0 < self.min_bootstrap_success <= 1.0:
            raise ValueError("min_bootstrap_success must lie in (0, 1].")
        self.alpha = float(alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        self.seed = seed

        if isinstance(outcome_change_lags, bool):
            raise ValueError("outcome_change_lags must be a non-negative integer.")
        self.n_lagged_outcome_changes = int(outcome_change_lags)
        if (
            self.n_lagged_outcome_changes != outcome_change_lags
            or self.n_lagged_outcome_changes < 0
        ):
            raise ValueError("outcome_change_lags must be a non-negative integer.")
        self.include_lagged_outcome_change = self.n_lagged_outcome_changes > 0
        self.ps_with_time_fe = True
        self.ps_time_varying_slopes = False
        self.ps_clip = None
        self.lag_covariates = False
        self.control_pool = "stabilized_all"
        self.switch_in = "sustained"
        self.control_window = "horizon"
        self.time_policy = "calendar"
        self.results_ = None
        self.is_fitted_ = False

    def _resolve_pooled_horizons(self, pooled, *, kind: str):
        """Resolve diff-diff-style pooled-window requests."""
        if isinstance(pooled, bool):
            raise ValueError(
                f"{kind}_pooled must be None, an int, or a length-2 tuple."
            )
        if pooled is None:
            return None
        if isinstance(pooled, int):
            horizons = (
                list(range(-pooled, -1))
                if kind == "pre" else list(range(0, pooled + 1))
            )
        elif isinstance(pooled, tuple) and len(pooled) == 2:
            horizons = list(range(int(pooled[0]), int(pooled[1]) + 1))
        else:
            raise ValueError(
                f"{kind}_pooled must be None, an int, or a length-2 tuple."
            )
        if kind == "pre":
            invalid = [h for h in horizons if h >= -1]
            if self.max_pre is not None:
                invalid.extend(h for h in horizons if h < -self.max_pre)
        else:
            invalid = [h for h in horizons if h < 0]
            if self.max_post is not None:
                invalid.extend(h for h in horizons if h > self.max_post)
        invalid = sorted(set(invalid))
        if invalid:
            raise ValueError(
                f"Requested pooled {kind} horizons {invalid} fall outside "
                "the event-study window."
            )
        return horizons

    # ------------------------------------------------------------------
    # Dispatch to semiparametric estimator
    # ------------------------------------------------------------------

    def _fit_pooled_window(
        self, df, outcome, unit, time, horizons, kind,
        stack_controls, stack_user_covariates, ps_controls, or_controls,
        compute_influence, inference_cluster=None,
    ):
        """Estimate a pooled pre/post scalar on one complete common stack."""
        from .lpdid import _build_pooled_local

        hs = [int(h) for h in horizons]
        local = _build_pooled_local(
            df, outcome, unit, time, hs, kind, self.base_period,
            self.clean_control, self.effect_stabilization, stack_controls,
            user_covariates=stack_user_covariates,
            lag_covariates=self.lag_covariates,
            control_pool=self.control_pool,
            switch_in=self.switch_in,
            control_window=self.control_window,
        )
        if local.empty:
            return None
        local, or_training, support_diagnostics = _partition_support_samples(
            local,
            time,
            unsupported_event_policy=self.unsupported_event_policy,
            or_training_support=self.or_training_support,
        )
        if local.empty or local["D_local"].sum() == 0 or (local["D_local"] == 0).sum() == 0:
            return None
        result = self._fit_one(
            local,
            inference_cluster or unit,
            time,
            ps_controls,
            or_controls,
            compute_influence,
            or_training_sample=or_training,
        )
        result["n_event_rows"] = int(local["D_local"].eq(1).sum())
        result.update(support_diagnostics)
        return result

    def _fit_one(
        self,
        local_sample: pd.DataFrame,
        unit: str,
        time: str,
        ps_controls: List[str],
        or_controls: List[str],
        compute_influence: bool,
        or_training_sample: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Dispatch to the appropriate semiparametric estimator."""
        if self._family == "ra":
            result = _compute_ra(
                local_sample, unit, time, or_controls, compute_influence,
                or_time_varying_slopes=self.or_time_varying_slopes,
                or_training_sample=or_training_sample,
            )
        elif self._family == "ipw":
            result = _compute_ipw(
                local_sample,
                unit,
                time,
                ps_controls,
                self._ps_method,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
            )
        elif self._ps_method == "generic":
            result = _compute_dr_generic(
                local_sample,
                unit,
                time,
                ps_controls,
                or_controls,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
                or_time_varying_slopes=self.or_time_varying_slopes,
            )
        else:
            result = _compute_dr_improved(
                local_sample,
                unit,
                time,
                ps_controls,
                or_controls,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
                or_time_varying_slopes=self.or_time_varying_slopes,
            )

        factor = 1.0
        if (
            compute_influence
            and self.small_sample_correction == "cr1"
            and "psi_by_cluster" in result
        ):
            # CR1 follows the same cluster-sandwich convention used by the
            # benchmark LPDID engine: G/(G-1) times (n-1)/(n-K), with K equal
            # to the effective dimension of the moment system actually fitted.
            # The previous common full-DR reference dimension was convenient
            # for cross-method numerical identities but was not the method's
            # own finite-sample sandwich correction.
            k_reference = int(result["n_moment_parameters"])
            n = int(result["n_inference_rows"])
            g = int(result["n_inference_clusters"])
            if g <= 1 or n <= k_reference:
                raise InferenceError(
                    "CR1 requires at least two clusters and more inference "
                    "rows than reference parameters; "
                    f"got G={g}, n={n}, K_ref={k_reference}."
                )
            factor = float(np.sqrt(
                (g / (g - 1.0))
                * ((n - 1.0) / (n - k_reference))
            ))
            result["psi_by_cluster"] = factor * result["psi_by_cluster"]
            result["se"] = se_from_influence(result["psi_by_cluster"].to_numpy())
            result["cr1_parameters"] = k_reference
        result["small_sample_factor"] = factor
        result["covariance_estimator"] = "cluster_sandwich"
        return result

    # ------------------------------------------------------------------
    # Core estimation loop
    # ------------------------------------------------------------------

    def _fit_core(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: Optional[str],
        treatment: Optional[str],
        covariates: Optional[List[str]],
        compute_influence: bool = True,
        *,
        _ps_covariates: Optional[List[str]] = None,
        _or_covariates: Optional[List[str]] = None,
        _n_lagged_outcome_changes: Optional[int] = None,
        _n_lagged_outcome_levels: int = 0,
        _absorbed_columns: Optional[List[str]] = None,
        _pooled_pre_horizons: Optional[List[int]] = None,
        _pooled_post_horizons: Optional[List[int]] = None,
    ) -> dict:
        n_change_lags = (
            self.n_lagged_outcome_changes
            if _n_lagged_outcome_changes is None
            else int(_n_lagged_outcome_changes)
        )
        n_level_lags = int(_n_lagged_outcome_levels)
        include_change_lags = n_change_lags > 0
        base_covariates = list(covariates or [])
        ps_base_covariates = (
            base_covariates
            if _ps_covariates is None
            else list(_ps_covariates)
        )
        or_base_covariates = (
            base_covariates
            if _or_covariates is None
            else list(_or_covariates)
        )
        all_base_covariates = list(dict.fromkeys(
            base_covariates + ps_base_covariates + or_base_covariates
        ))
        if all_base_covariates:
            check_columns(data, all_base_covariates)
        df = prepare_panel(
            data, outcome, unit, time, first_treat, treatment,
            nonabsorbing=(self.non_absorbing is not None),
            n_lagged_outcome_changes=(
                n_change_lags
                if include_change_lags
                else 0
            ),
            n_lagged_outcome_levels=n_level_lags,
            left_censoring=self.left_censoring,
            time_policy=self.time_policy,
            history_policy=self.history_policy,
        )
        inference_cluster = self.cluster or unit
        data_max_pre, data_max_post = infer_windows(
            df, time, self.base_period, None, None
        )
        max_pre = data_max_pre if self.max_pre is None else self.max_pre
        max_post = data_max_post if self.max_post is None else self.max_post
        # The reported grid is the requested or data-inferred grid.
        # Data-unavailable horizons
        # remain explicit NaN rows rather than disappearing from the result.
        if self.effect_stabilization is not None:
            df = precompute_ccs(
                df, unit, int(self.effect_stabilization), max_pre, max_post
            )

        ps_controls = _control_columns(
            ps_base_covariates,
            include_change_lags,
            n_change_lags,
            n_level_lags,
        )
        or_controls = _control_columns(
            or_base_covariates,
            include_change_lags,
            n_change_lags,
            n_level_lags,
        )
        stack_controls = list(dict.fromkeys(ps_controls + or_controls))
        stack_user_covariates = all_base_covariates
        cluster_universe = pd.Index(df[inference_cluster].dropna().unique())
        psi_by_h: Dict[int, pd.Series] = {}
        clusters_by_h: Dict[int, pd.Index] = {}
        rows: list[dict] = []
        unsupported_horizons: dict[int, str] = {}
        nuisance_diagnostics: dict[int, dict] = {}
        support_diagnostics_by_h: dict[int, dict] = {}
        control_unit_ids: set = set()
        omitted_overlapping_outcome_controls: dict[int, tuple[str, ...]] = {}

        pre_horizons = list(range(-max_pre, -self.anticipation))
        post_horizons = list(range(0, max_post + 1))
        if (
            isinstance(self.base_period, int)
            and self.base_period == -1
            and -1 in pre_horizons
        ):
            pre_horizons.remove(-1)

        for h in pre_horizons + post_horizons:
            # For a placebo lead h<-1, outcome_local = Y_{t+h}-Y_{t-1}.
            # Generated changes ldy1,...,ldy(-h-1), and the target-period
            # level ly(-h), are algebraic components of that dependent
            # variable. Conditioning on them can make the placebo mechanically
            # rather than scientifically predictable.
            # RA is the Dube/diff-diff benchmark inside DRLPDID and therefore
            # retains the same lag basis, including a mechanically fitted
            # placebo when a lagged outcome change spans the placebo outcome.
            # Propensity and doubly robust methods keep the stricter native
            # convention: algebraic components of a placebo outcome are
            # omitted and reported rather than interpreted as pre-trend
            # evidence.
            overlapping = (
                {f"ldy{k}" for k in range(1, -int(h))}
                | {f"ly{-int(h)}"}
                if h < -1 and self._family != "ra" else set()
            )
            ps_controls_h = [c for c in ps_controls if c not in overlapping]
            or_controls_h = [c for c in or_controls if c not in overlapping]
            stack_controls_h = list(dict.fromkeys(ps_controls_h + or_controls_h))
            omitted = tuple(sorted(overlapping.intersection(stack_controls)))
            if omitted:
                omitted_overlapping_outcome_controls[int(h)] = omitted
            local = build_local_sample(
                df, outcome, unit, time, h, self.base_period,
                self.clean_control, self.effect_stabilization, stack_controls_h,
                user_covariates=stack_user_covariates,
                lag_covariates=self.lag_covariates,
                fixed_composition_H=None,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
            )
            local, or_training, support_diagnostics = _partition_support_samples(
                local,
                time,
                unsupported_event_policy=self.unsupported_event_policy,
                or_training_support=self.or_training_support,
            )
            support_diagnostics_by_h[int(h)] = support_diagnostics
            if (
                local.empty
                or local["D_local"].sum() == 0
                or (local["D_local"] == 0).sum() == 0
            ):
                unsupported_horizons[int(h)] = "no_supported_reference_date"
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cluster_universe)
                clusters_by_h[int(h)] = pd.Index([])
                continue

            event_mask = local["D_local"] == 1
            control_mask = ~event_mask
            control_unit_ids.update(local.loc[control_mask, unit].tolist())
            counts = {
                "n_event_rows": int(event_mask.sum()),
                "n_event_units": int(local.loc[event_mask, unit].nunique()),
                "n_control_rows": int(control_mask.sum()),
                "n_control_units": int(local.loc[control_mask, unit].nunique()),
                "n_reference_dates": int(local[time].nunique()),
            }
            if self.design == "switching":
                counts.update({
                    "n_stable_zero_control_rows": int(
                        (control_mask & (local["_treat"] == 0)).sum()
                    ),
                    "n_stable_one_control_rows": int(
                        (control_mask & (local["_treat"] == 1)).sum()
                    ),
                })
            counts.update(support_diagnostics)
            try:
                fit = self._fit_one(
                    local, inference_cluster, time,
                    ps_controls_h, or_controls_h,
                    compute_influence,
                    or_training_sample=or_training,
                )
            except IPTConvergenceError as exc:
                raise IPTConvergenceError(
                    f"Horizon h={int(h)}: {exc}"
                ) from exc
            except NuisanceConvergenceError as exc:
                raise NuisanceConvergenceError(
                    f"Horizon h={int(h)}: {exc}"
                ) from exc
            except JacobianError as exc:
                raise JacobianError(f"Horizon h={int(h)}: {exc}") from exc

            est, se = fit["estimate"], fit["se"]
            nuisance_diagnostics[int(h)] = {
                key: value for key, value in fit.items()
                if (
                    key.startswith("ipt_")
                    or key.startswith("ps_")
                    or key.startswith("or_")
                    or key.startswith("propensity_")
                    or key.startswith("control_odds_")
                    or key.startswith("sandwich_")
                    or key in {
                        "control_weight_ess", "small_sample_factor",
                        "cr1_parameters", "covariance_estimator",
                        "n_moment_parameters", "n_inference_rows",
                        "n_inference_clusters",
                    }
                )
            }
            local_psi = fit.get(
                "psi_by_cluster",
                pd.Series(
                    0.0, index=pd.Index(or_training[inference_cluster].unique())
                ),
            )
            clusters_by_h[int(h)] = pd.Index(
                or_training[inference_cluster].dropna().unique()
            )
            psi_by_h[int(h)] = local_psi.reindex(
                cluster_universe, fill_value=0.0
            )
            rows.append(_make_row(
                h, est, se, self.alpha,
                int(fit.get("n_inference_clusters", len(clusters_by_h[int(h)]))),
                counts=counts,
            ))

        if isinstance(self.base_period, int) and self.base_period == -1:
            rows.append({
                "horizon": -1, "estimate": 0.0, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "df": np.nan, "n_clusters": 0, "critical_value": np.nan,
                "n_event_rows": 0, "n_event_units": 0,
                "n_control_rows": 0, "n_control_units": 0,
                "n_reference_dates": 0,
            })
            if self.design == "switching":
                rows[-1].update({
                    "n_stable_zero_control_rows": 0,
                    "n_stable_one_control_rows": 0,
                })
            psi_by_h[-1] = pd.Series(0.0, index=cluster_universe)
            clusters_by_h[-1] = pd.Index([])

        event_df = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        if unsupported_horizons:
            details = ", ".join(
                f"h={h} ({reason})" for h, reason in sorted(unsupported_horizons.items())
            )
            warnings.warn(
                "Some horizons were not identified and were returned as NaN: " + details,
                stacklevel=2,
            )

        # Scalar summaries. New-style calls expose pre/post aggregates by
        # default; legacy ``horizons=`` calls retain their opt-in behavior.
        scalar_rows: list[dict] = []
        scalar_specs: dict = {}
        psi_by_scalar: Dict[str, pd.Series] = {}
        scalar_missing_horizons: list[int] = []

        effective_policy_window = self.policy_window
        if self._automatic_aggregates and effective_policy_window is None:
            effective_policy_window = (0, max_post)

        resolved_pre_aggregations = list(self.pre_aggregations)
        if _pooled_pre_horizons is not None and not resolved_pre_aggregations:
            resolved_pre_aggregations = ["pooled"]
        if resolved_pre_aggregations:
            requested_pre = (
                list(range(-max_pre, -1))
                if _pooled_pre_horizons is None
                else [int(h) for h in _pooled_pre_horizons]
            )
            by_h = event_df.set_index("horizon")
            unsupported_pre = [
                h for h in requested_pre
                if h not in by_h.index or not np.isfinite(by_h.loc[h, "estimate"])
            ]
            if requested_pre and not unsupported_pre and "pooled" in resolved_pre_aggregations:
                pooled_overlap = (
                    set().union(*(
                        (
                            {f"ldy{k}" for k in range(1, -int(h))}
                            | {f"ly{-int(h)}"}
                        )
                        for h in requested_pre if h < -1
                    ))
                    if any(h < -1 for h in requested_pre)
                    and self._family != "ra"
                    else set()
                )
                pooled_ps_controls = [
                    control for control in ps_controls
                    if control not in pooled_overlap
                ]
                pooled_or_controls = [
                    control for control in or_controls
                    if control not in pooled_overlap
                ]
                pooled_stack_controls = list(dict.fromkeys(
                    pooled_ps_controls + pooled_or_controls
                ))
                fit_pre = self._fit_pooled_window(
                    df, outcome, unit, time, requested_pre, "pre",
                    pooled_stack_controls, stack_user_covariates,
                    pooled_ps_controls, pooled_or_controls,
                    compute_influence, inference_cluster=inference_cluster,
                )
                estimate_pre = (
                    float(fit_pre["estimate"]) if fit_pre is not None else np.nan
                )
                se_pre = (
                    float(fit_pre["se"])
                    if fit_pre is not None and compute_influence else np.nan
                )
                term_pre = "pre [pooled]"
                if compute_influence and fit_pre is not None:
                    psi_by_scalar[term_pre] = fit_pre[
                        "psi_by_cluster"
                    ].reindex(cluster_universe, fill_value=0.0)
                if np.isfinite(estimate_pre):
                    inference_pre = analytic_cluster_inference(
                        estimate_pre, se_pre, self.alpha,
                        int(fit_pre.get("n_inference_clusters", 0)),
                    )
                    scalar_specs[term_pre] = (requested_pre, "pooled")
                    scalar_rows.append({
                        "term": term_pre, "window": "pre",
                        "aggregation": "pooled",
                        "estimate": estimate_pre, "se": se_pre,
                        **inference_pre,
                        "n_horizons": len(requested_pre),
                        "horizons": tuple(requested_pre),
                        "weights": None,
                        "weight_basis": "joint_pooled_outcome_regression",
                        "n_event_rows": int(fit_pre.get("n_event_rows", 0)),
                    })
            elif unsupported_pre and "pooled" in resolved_pre_aggregations:
                warnings.warn(
                    "The pooled pre-treatment placebo was not estimated "
                    "because these requested horizons are unsupported: "
                    f"{unsupported_pre}.",
                    stacklevel=2,
                )
            for method in resolved_pre_aggregations:
                if method == "pooled":
                    continue
                row, psi, missing = horizon_aggregate(
                    event_df, psi_by_h, clusters_by_h, cluster_universe,
                    requested_pre, window="pre", method=method,
                    alpha=self.alpha, compute_influence=compute_influence,
                )
                if row is None:
                    if missing:
                        warnings.warn(
                            f"The pre-treatment {method} summary was not "
                            f"estimated because horizons {missing} are unsupported.",
                            stacklevel=2,
                        )
                    continue
                scalar_rows.append(row)
                scalar_specs[str(row["term"])] = (
                    requested_pre, tuple(row["weights"])
                )
                if psi is not None:
                    psi_by_scalar[str(row["term"])] = psi

        requested_post_override = (
            None if _pooled_post_horizons is None
            else [int(h) for h in _pooled_post_horizons]
        )
        resolved_post_aggregations = list(self.post_aggregations)
        if requested_post_override is not None and not resolved_post_aggregations:
            resolved_post_aggregations = ["pooled"]
        if requested_post_override is not None:
            effective_policy_window = (
                (min(requested_post_override), max(requested_post_override))
                if requested_post_override else None
            )

        requested: list[int] = []
        if effective_policy_window is not None:
            lo, hi = effective_policy_window
            requested = (
                requested_post_override
                if requested_post_override is not None
                else list(range(lo, hi + 1))
            )
        if (
            effective_policy_window is not None
            and "pooled" in resolved_post_aggregations
        ):
            lo, hi = effective_policy_window
            term = (
                "post [pooled]" if self._automatic_aggregates
                else f"ATT pooled [{lo},{hi}]"
            )
            fit_p = self._fit_pooled_window(
                df, outcome, unit, time, requested, "post",
                stack_controls, stack_user_covariates, ps_controls, or_controls,
                compute_influence, inference_cluster=inference_cluster,
            )
            est_p = float(fit_p["estimate"]) if fit_p is not None else np.nan
            if not np.isfinite(est_p):
                scalar_missing_horizons = requested
                warnings.warn(
                    "The pooled post-window ATT was not estimated: no identified "
                    f"common sample for horizons {requested}.",
                    stacklevel=2,
                )
            else:
                se_p = float(fit_p["se"]) if compute_influence else np.nan
                scalar_specs[term] = (requested, "pooled")
                if compute_influence:
                    psi_by_scalar[term] = fit_p["psi_by_cluster"].reindex(
                        cluster_universe, fill_value=0.0
                    )
                inference_p = analytic_cluster_inference(
                    est_p, se_p, self.alpha,
                    int(fit_p.get("n_inference_clusters", 0)),
                )
                scalar_rows.append({
                    "term": term,
                    "window": "post",
                    "aggregation": "pooled",
                    "estimate": est_p,
                    "se": se_p,
                    **inference_p,
                    "n_horizons": len(requested),
                    "horizons": tuple(requested),
                    "weights": None,
                    "weight_basis": "joint_pooled_outcome_regression",
                    "n_event_rows": int(fit_p.get("n_event_rows", 0)),
                })
        if effective_policy_window is not None:
            lo, hi = effective_policy_window
            for method in resolved_post_aggregations:
                if method == "pooled":
                    continue
                row, psi, missing = horizon_aggregate(
                    event_df, psi_by_h, clusters_by_h, cluster_universe,
                    requested, window="post", method=method,
                    alpha=self.alpha, compute_influence=compute_influence,
                )
                if row is None:
                    scalar_missing_horizons.extend(missing)
                    if missing:
                        warnings.warn(
                            f"The post-treatment {method} ATT was not estimated "
                            f"because horizons {missing} are unsupported.",
                            stacklevel=2,
                        )
                    continue
                if not self._automatic_aggregates:
                    row["term"] = (
                        f"ATT policy-window [{lo},{hi}] [{method}]"
                    )
                scalar_rows.append(row)
                scalar_specs[str(row["term"])] = (
                    requested, tuple(row["weights"])
                )
                if psi is not None:
                    psi_by_scalar[str(row["term"])] = psi
        scalar_missing_horizons = sorted(set(scalar_missing_horizons))

        rank_dropped_columns: dict[int, tuple[str, ...]] = {}
        for horizon, diagnostics in nuisance_diagnostics.items():
            dropped = sorted(set(
                diagnostics.get("ps_dropped_columns", ())
                + diagnostics.get("ipt_dropped_columns", ())
                + diagnostics.get("or_dropped_columns", ())
            ))
            if dropped:
                rank_dropped_columns[int(horizon)] = tuple(dropped)
        if rank_dropped_columns:
            detail = "; ".join(
                f"h={h}: {list(columns)}"
                for h, columns in sorted(rank_dropped_columns.items())
            )
            message = (
                "Rank-deficient DRLPDID nuisance design detected. "
                "Deterministic rank reduction dropped: " + detail + "."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(message)
            if self.rank_deficient_action == "warn":
                warnings.warn(message, UserWarning, stacklevel=2)

        metadata: dict = {
            "package_version": "0.8.1",
            "estimator_identity": f"drlpdid_{self.estimation_method}",
            "estimator_label": (
                "RA"
                if self.estimation_method == "ra"
                else (
                    "IPW" if self.estimation_method == "ipw"
                    else (
                        "IPT" if self.estimation_method == "ipt"
                        else f"DRLPDID-{self.estimation_method[3:].upper()}"
                    )
                )
            ),
            "design": self.design,
            "estimation_method": self.estimation_method,
            "control_group": self.control_group,
            "clean_control_rule": self.clean_control,
            "non_absorbing": self.non_absorbing,
            "stabilization_window": self.effect_stabilization,
            "horizons": (-max_pre, max_post),
            "evaluated_horizons": (-max_pre, max_post),
            "pre_window": max_pre,
            "post_window": max_post,
            "base_period": -1,
            "aggregate_window": effective_policy_window,
            "pre_pooled_horizons": tuple(
                list(range(-max_pre, -1))
                if _pooled_pre_horizons is None else _pooled_pre_horizons
            ),
            "post_pooled_horizons": tuple(
                list(range(effective_policy_window[0], effective_policy_window[1] + 1))
                if requested_post_override is None and effective_policy_window is not None
                else (requested_post_override or [])
            ),
            "pre_aggregation": self.pre_aggregation,
            "post_aggregation": self.post_aggregation,
            "pre_aggregations": tuple(resolved_pre_aggregations),
            "post_aggregations": tuple(resolved_post_aggregations),
            "primary_post_aggregation": (
                resolved_post_aggregations[0]
                if resolved_post_aggregations else None
            ),
            "aggregation_inference": "conditional_on_realized_design",
            "post_window_missing_horizons": scalar_missing_horizons,
            "support": "reference_dates_with_events_and_clean_controls",
            "target_support": "event_dates_with_clean_controls",
            "or_training_support": self.or_training_support,
            "unsupported_event_policy": self.unsupported_event_policy,
            "support_diagnostics": support_diagnostics_by_h,
            "composition": "as_observed",
            "cluster_level": inference_cluster,
            "df_convention": self.df_convention,
            "inference": self.inference,
            "covariance_estimator": "cluster_sandwich",
            "small_sample_correction": self.small_sample_correction,
            "reference_distribution": "student_t_G_minus_1",
            "or_slope_structure": self.or_slope_structure,
            "outcome_change_lags": n_change_lags,
            "outcome_level_lags": n_level_lags,
            "absorb": tuple(_absorbed_columns or []),
            "history_policy": self.history_policy,
            "left_censoring": self.left_censoring,
            "multiplier_weights": (
                self.bootstrap_weights if self.inference == "multiplier" else None
            ),
            "event_weighting": (
                "eligible_switch_in_events"
                if self.design == "switching" else "treated_entries"
            ),
            "covariates": base_covariates,
            "ps_covariates": ps_base_covariates,
            "or_covariates": or_base_covariates,
            "unsupported_horizons": unsupported_horizons,
            "rank_deficient_action": self.rank_deficient_action,
            "rank_dropped_columns": rank_dropped_columns,
            "nuisance_diagnostics": nuisance_diagnostics,
            "omitted_overlapping_outcome_controls": (
                omitted_overlapping_outcome_controls
            ),
        }

        return {
            "panel": df,
            "event_study": event_df,
            "scalars": pd.DataFrame(scalar_rows),
            "scalar_specs": scalar_specs,
            "psi_by_scalar": psi_by_scalar,
            "psi_by_h": psi_by_h,
            "cluster_universe": cluster_universe,
            "n_obs": int(df.shape[0]),
            "n_treated_units": int(df.loc[df["_treated_ever"] == 1, unit].nunique()),
            "n_control_units": int(len(control_unit_ids)),
            "n_cohorts": int(df.loc[df["_first_treat"] > 0, "_first_treat"].nunique()),
            "n_periods": int(df[time].nunique()),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Public fit method
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        ylags: int = 0,
        dylags: int = 0,
        absorb: Optional[List[str]] = None,
        post_pooled=None,
        pre_pooled=None,
        only_event: bool = False,
        only_pooled: bool = False,
        survey_design=None,
        *,
        first_treat: Optional[str] = None,
        ps_covariates: Optional[List[str]] = None,
        or_covariates: Optional[List[str]] = None,
    ) -> DRLPDIDResults:
        """Fit the DR-LP-DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format panel dataset.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name.
        time : str
            Time identifier column name.
        treatment : str, optional
            Binary treatment indicator column.
        covariates : list of str, optional
            Predetermined covariates used in both local nuisance models.
        ylags, dylags : int, default 0
            Predetermined outcome-level and outcome-change lags included in
            both nuisance bases. Placebo-overlapping changes are omitted and
            recorded rather than mechanically fitting a lead for propensity
            and doubly robust methods. The RA benchmark retains the Dube et al.
            convention so it remains numerically comparable with ``LPDID``.
        absorb : list of str, optional
            Categorical factors expanded into a deterministic dummy basis for
            both nuisance models.
        post_pooled, pre_pooled : int or length-2 tuple, optional
            Diff-diff-style custom pooled windows. When omitted, the public
            pre/post rows use the complete requested event-study windows.
        only_event, only_pooled : bool, default False
            Restrict the public result surface while retaining the complete
            internal calculation needed for valid aggregates.
        first_treat : str, optional
            First-treatment period column (0 for never-treated). This is an
            advanced alternative to supplying the observed treatment path.
        ps_covariates, or_covariates : list of str, optional
            Advanced overrides for the propensity-score and outcome-regression
            bases. When omitted, each falls back to ``covariates``.

        Returns
        -------
        DRLPDIDResults
            Result object with ``event_study`` (including simultaneous CIs
            when ``inference='multiplier'``), ``scalars``, and
            ``print_summary()``.
        """
        if survey_design is not None:
            raise NotImplementedError(
                "survey_design is not certified for semiparametric DRLPDID."
            )
        if only_event and only_pooled:
            raise ValueError("only_event and only_pooled cannot both be True.")
        for name, value in (("ylags", ylags), ("dylags", dylags)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if dylags and self.n_lagged_outcome_changes:
            raise ValueError(
                "Use fit(..., dylags=) or the legacy constructor "
                "outcome_change_lags=, not both."
            )
        resolved_dylags = int(dylags or self.n_lagged_outcome_changes)
        pooled_pre_horizons = self._resolve_pooled_horizons(
            pre_pooled, kind="pre"
        )
        pooled_post_horizons = self._resolve_pooled_horizons(
            post_pooled, kind="post"
        )
        if pre_pooled is not None and not pooled_pre_horizons and not only_event:
            raise ValueError(
                "pooled pre window is empty; use only_event=True or request "
                "at least one lead before h=-1."
            )
        if post_pooled is not None and self.policy_window is not None:
            raise ValueError(
                "Use fit(..., post_pooled=) or the legacy constructor "
                "aggregate_window=, not both."
            )

        covariates = list(covariates or [])
        absorb = list(absorb or [])
        data_for_fit = data
        absorbed_dummy_columns: list[str] = []
        if absorb:
            missing_columns = [column for column in absorb if column not in data]
            if missing_columns:
                raise ValueError(f"Absorbed columns not found: {missing_columns}")
            data_for_fit = data.copy()
            for factor_number, column in enumerate(absorb):
                values = data_for_fit[column]
                dummies = pd.get_dummies(
                    values,
                    prefix=f"__pyd_dr_absorb_{factor_number}",
                    drop_first=True,
                    dtype=float,
                )
                if not dummies.empty:
                    dummies.loc[values.isna(), :] = float("nan")
                    data_for_fit = pd.concat([data_for_fit, dummies], axis=1)
                    absorbed_dummy_columns.extend(dummies.columns.astype(str))
            covariates.extend(absorbed_dummy_columns)

        resolved_ps_covariates = (
            None if ps_covariates is None
            else list(ps_covariates) + absorbed_dummy_columns
        )
        resolved_or_covariates = (
            None if or_covariates is None
            else list(or_covariates) + absorbed_dummy_columns
        )

        if self.design == "absorbing" and first_treat is None and treatment is None:
            raise ValueError(
                "Absorbing treatment requires either first_treat or the "
                "observed binary treatment path."
            )
        if self.design == "switching" and treatment is None:
            raise ValueError(
                "design='switching' requires the observed binary treatment path."
            )
        if self.cluster is not None:
            check_columns(data_for_fit, [self.cluster])
            if data_for_fit[self.cluster].isna().any():
                raise ValueError("The cluster column contains missing values.")
            unit_to_cluster = data_for_fit.groupby(unit, sort=False)[self.cluster].nunique(
                dropna=False
            )
            if (unit_to_cluster != 1).any():
                raise ValueError(
                    "Each panel unit must belong to exactly one inference cluster."
                )

        compute_influence = self.inference in {"cluster", "multiplier"}
        core = self._fit_core(
            data_for_fit, outcome, unit, time, first_treat, treatment, covariates,
            compute_influence=compute_influence,
            _ps_covariates=resolved_ps_covariates,
            _or_covariates=resolved_or_covariates,
            _n_lagged_outcome_changes=resolved_dylags,
            _n_lagged_outcome_levels=int(ylags),
            _absorbed_columns=absorb,
            _pooled_pre_horizons=pooled_pre_horizons,
            _pooled_post_horizons=pooled_post_horizons,
        )
        event_df = core["event_study"].copy()
        scalars = core["scalars"].copy()

        if self.inference == "multiplier":
            rng = np.random.default_rng(self.seed)
            event_df = event_df.sort_values("horizon").reset_index(drop=True)
            event_df["multiplier_ci_lower"] = np.nan
            event_df["multiplier_ci_upper"] = np.nan
            event_df["multiplier_p_value"] = np.nan
            event_df["sim_ci_lower"] = np.nan
            event_df["sim_ci_upper"] = np.nan
            band_mask = (
                np.isfinite(event_df["estimate"])
                & np.isfinite(event_df["se"])
                & (event_df["se"] > 0)
            )
            band_es = event_df.loc[band_mask].copy()
            band_horizons = [int(h) for h in band_es["horizon"].tolist()]
            if band_horizons:
                Psi = np.column_stack([
                    core["psi_by_h"][h].reindex(
                        core["cluster_universe"], fill_value=0.0
                    ).to_numpy(dtype=float)
                    for h in band_horizons
                ])
                mb = run_multiplier_bootstrap(
                    band_es["estimate"].to_numpy(dtype=float),
                    Psi, band_es["se"].to_numpy(dtype=float),
                    self.n_bootstrap, self.bootstrap_weights, self.alpha, rng,
                )
                event_df.loc[band_mask, "multiplier_ci_lower"] = mb["ci_lower"]
                event_df.loc[band_mask, "multiplier_ci_upper"] = mb["ci_upper"]
                event_df.loc[band_mask, "multiplier_p_value"] = mb["p_values"]
                event_df.loc[band_mask, "sim_ci_lower"] = mb["sim_ci_lower"]
                event_df.loc[band_mask, "sim_ci_upper"] = mb["sim_ci_upper"]
                critical_value = mb["cband_crit"]
            else:
                critical_value = np.nan
            core["metadata"].update({
                "multiplier": {
                    "critical_value": critical_value,
                    "weight_type": self.bootstrap_weights,
                    "n_bootstrap": self.n_bootstrap,
                    "seed": self.seed,
                    "simultaneous_band_horizons": tuple(band_horizons),
                }
            })
            if not scalars.empty:
                scalars["multiplier_ci_lower"] = np.nan
                scalars["multiplier_ci_upper"] = np.nan
                scalars["multiplier_p_value"] = np.nan
                scalar_bootstrapped: list[str] = []
                for idx, row in scalars.iterrows():
                    term = str(row["term"])
                    psi = core["psi_by_scalar"].get(term)
                    se = float(row["se"])
                    if psi is None or not np.isfinite(se) or se <= 0:
                        continue
                    mb_scalar = run_multiplier_bootstrap(
                        np.asarray([float(row["estimate"])]),
                        psi.reindex(
                            core["cluster_universe"], fill_value=0.0
                        ).to_numpy(dtype=float)[:, None],
                        np.asarray([se]),
                        self.n_bootstrap,
                        self.bootstrap_weights,
                        self.alpha,
                        rng,
                    )
                    scalars.loc[idx, "multiplier_ci_lower"] = mb_scalar["ci_lower"][0]
                    scalars.loc[idx, "multiplier_ci_upper"] = mb_scalar["ci_upper"][0]
                    scalars.loc[idx, "multiplier_p_value"] = mb_scalar["p_values"][0]
                    scalar_bootstrapped.append(term)
                core["metadata"]["multiplier"]["scalar_terms"] = tuple(
                    scalar_bootstrapped
                )

        if self.inference == "cluster_bootstrap":
            rng = np.random.default_rng(self.seed)
            original = {
                "event_index": event_df["horizon"].tolist(),
                "event_hat": dict(zip(event_df["horizon"], event_df["estimate"])),
                "scalar_terms": scalars["term"].tolist(),
                "scalar_hat": dict(zip(scalars["term"], scalars["estimate"])),
            }

            def _refit(boot_df):
                fitted = self._fit_core(
                    boot_df, outcome, unit, time, first_treat, treatment,
                    covariates, compute_influence=False,
                    _ps_covariates=resolved_ps_covariates,
                    _or_covariates=resolved_or_covariates,
                    _n_lagged_outcome_changes=resolved_dylags,
                    _n_lagged_outcome_levels=int(ylags),
                    _absorbed_columns=absorb,
                    _pooled_pre_horizons=pooled_pre_horizons,
                    _pooled_post_horizons=pooled_post_horizons,
                )
                return {
                    "event_study": fitted["event_study"],
                    "scalars": fitted["scalars"],
                }

            boot = run_cluster_bootstrap(
                _refit,
                data_for_fit.copy(),
                self.cluster or unit,
                self.n_bootstrap,
                self.alpha,
                rng,
                original,
                min_success_rate=self.min_bootstrap_success,
                panel_unit=unit,
            )
            for frame_name, boot_name, key in (
                ("event", "event_boot", "horizon"),
                ("scalar", "scalar_boot", "term"),
            ):
                frame = event_df if frame_name == "event" else scalars
                merged = frame.merge(
                    boot[boot_name], on=key, how="left", suffixes=("", "_bootstrap")
                )
                for column in ("se", "ci_lower", "ci_upper", "p_value", "t_stat"):
                    boot_column = f"{column}_bootstrap"
                    if boot_column in merged:
                        merged[column] = merged[boot_column]
                        merged.drop(columns=[boot_column], inplace=True)
                if frame_name == "event":
                    event_df = merged
                else:
                    scalars = merged
            core["metadata"]["cluster_bootstrap"] = {
                "n_requested": boot["n_requested"],
                "n_successful": boot["n_successful"],
                "success_rate": boot["success_rate"],
                "failures": boot["failures"],
                "interval": "percentile",
                "seed": self.seed,
            }

        for frame in (event_df, scalars):
            if not frame.empty:
                frame["coefficient"] = frame["estimate"]
                frame["conf_low"] = frame["ci_lower"]
                frame["conf_high"] = frame["ci_upper"]
        event_study_stable = pd.DataFrame()
        if only_event:
            scalars = scalars.iloc[0:0].copy()
        if only_pooled:
            event_df = event_df.iloc[0:0].copy()

        result = DRLPDIDResults(
            estimator_name=core["metadata"]["estimator_label"],
            n_obs=core["n_obs"],
            n_treated_units=core["n_treated_units"],
            n_control_units=core["n_control_units"],
            n_cohorts=core["n_cohorts"],
            n_periods=core["n_periods"],
            base_period=self.base_period,
            clean_control=self.clean_control,
            effect_stabilization=self.effect_stabilization,
            anticipation=self.anticipation,
            inference=self.inference,
            alpha=self.alpha,
            event_study=event_df,
            event_study_stable=event_study_stable,
            scalars=scalars,
            metadata=core["metadata"],
            estimation_method=self.estimation_method,
            dr_method=self.dr_method,
            covariates=core["metadata"].get("covariates", list(covariates or [])),
            target_estimand="ATT",
        )
        self.results_ = result
        self.is_fitted_ = True
        return result


# ---------------------------------------------------------------------------
# Shared row helpers (also used by lpdid.py via direct call)
# ---------------------------------------------------------------------------

def _nan_row(h: int) -> dict:
    return {
        "horizon": int(h),
        "estimate": np.nan, "se": np.nan,
        "t_stat": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_event_rows": 0, "n_event_units": 0,
        "n_control_rows": 0, "n_control_units": 0,
        "n_reference_dates": 0,
    }


def _make_row(
    h: int,
    est: float,
    se: float,
    alpha: float,
    n_clusters: int,
    counts: Optional[dict] = None,
) -> dict:
    row = {
        "horizon": int(h),
        "estimate": est,
        "se": se,
        **analytic_cluster_inference(est, se, alpha, n_clusters),
    }
    if counts:
        row.update(counts)
    return row
