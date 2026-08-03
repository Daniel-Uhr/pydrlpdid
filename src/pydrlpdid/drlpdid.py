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
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize
from scipy.special import expit

from ._errors import (
    NuisanceConvergenceError, JacobianError,
)

from ._inference import run_multiplier_bootstrap, se_from_influence, stacked_influence
from ._panel_utils import (
    build_local_sample,
    check_columns,
    p_value_two_sided,
    prepare_panel,
    precompute_ccs,
    z_crit,
)
from ._results import DRLPDIDResults


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
    formula: str
    column_names: Tuple[str, ...]
    dropped_columns: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _control_columns(
    covariates: Optional[List[str]],
    include_ldy: bool,
    n_lags: int,
) -> List[str]:
    """Collect the list of right-hand-side control column names."""
    cols: List[str] = []
    if include_ldy and n_lags > 0:
        cols.extend([f"ldy{k}" for k in range(1, int(n_lags) + 1)])
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


def _build_exog_from_fit(fit, data: pd.DataFrame) -> np.ndarray:
    """Re-evaluate a patsy design matrix on new data using a fitted model."""
    if isinstance(fit, (LocalPSResult, LocalORResult)):
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
) -> tuple[LocalORResult, np.ndarray]:
    """Fit a full-rank control outcome regression on one local stack."""
    formula = _build_formula("outcome_local", time, controls, add_time_fe=add_time_fe)
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
) -> tuple:
    """OLS outcome regression on clean-control observations.

    Returns ``(fitted_model, predicted_values_for_full_sample)``.
    """
    return _outcome_design(
        local_sample, time, controls, weights=None, add_time_fe=add_time_fe
    )


def _fit_or_wls_controls(
    local_sample: pd.DataFrame,
    time: str,
    controls: List[str],
    weights: np.ndarray,
    add_time_fe: bool = True,
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
    y = np.asarray(local_sample["outcome_local"], dtype=float)
    d = np.asarray(local_sample["D_local"], dtype=float)
    cluster_ids = local_sample[unit].to_numpy()
    fit_or, m0_hat = _fit_or_ols_controls(local_sample, time, controls)
    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
    resid = y - m0_hat
    mu1_hat = float(np.mean(resid[d == 1]))

    if not compute_influence:
        return {
            "estimate": mu1_hat, "se": np.nan,
        }

    theta_hat = np.concatenate([beta_hat, np.array([mu1_hat])])

    def moments_obs(theta):
        pb = z_all.shape[1]
        beta, mu1 = theta[:pb], theta[pb]
        resid_theta = y - z_all @ beta
        m_beta = ((1.0 - d))[:, None] * z_all * resid_theta[:, None]
        m_mu1 = (d * (resid_theta - mu1))[:, None]
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
        )
    }

    if not compute_influence:
        return {"estimate": tau_hat, "se": np.nan, **diagnostics}

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
    fit_or, m0_hat = _fit_or_ols_controls(local_sample, time, or_controls)
    X = np.asarray(fit_ps.exog, dtype=float)
    gamma_hat = np.asarray(fit_ps.params, dtype=float)
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

    if not compute_influence:
        return {
            "estimate": tau_hat, "se": np.nan,
        }

    z_all = _build_exog_from_fit(fit_or, local_sample)
    beta_hat = np.asarray(fit_or.params, dtype=float)
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
    }

    if not compute_influence:
        return {"estimate": tau_hat, "se": np.nan, **diagnostics}

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
        **diagnostics,
    }


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class DRLPDID:
    """Estimate a formally defined DR-LP-DiD event-study path.

    ``estimation_method`` selects ``'ra'``, ``'ipw'``, ``'ipt'``,
    ``'dr-ipw'``, or ``'dr-ipt'`` on one common horizon-specific stack.
    ``design`` is either absorbing staggered adoption or sustained switch-ins.
    For absorbing adoption, ``control_group`` is ``'not_yet_treated'``
    (default) or ``'never_treated'``. For switching,
    ``stabilization_window`` supplies the clean-history length L and the
    comparison rule is fixed by the formal design.

    ``horizons=(min_h, max_h)`` is inclusive and must contain h=-1 and h=0.
    ``post_window=(lo, hi)`` optionally requests one prespecified scalar
    average, reported only when all its horizons are identified.
    ``inference='cluster'`` reports analytic unit-cluster-robust inference;
    ``'multiplier'`` additionally reports simultaneous Rademacher sup-t bands.
    """

    def __init__(
        self,
        *,
        estimation_method: str = "dr-ipt",
        design: str = "absorbing",
        control_group: Optional[str] = None,
        stabilization_window: Optional[int] = None,
        horizons: Tuple[int, int] = (-5, 10),
        post_window: Optional[Tuple[int, int]] = None,
        inference: str = "multiplier",
        n_bootstrap: int = 999,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        """Create one of the two designs formally certified by the paper.

        ``design='absorbing'`` implements staggered absorbing adoption with
        not-yet-treated or never-treated controls. ``design='switching'``
        implements the sustained-switch-in target: a clean L-period history,
        treatment sustained through the horizon, and all transition-free
        local stayers as controls. The base is fixed at h=-1, composition is
        as observed, support is the set of reference dates containing both an
        eligible event and an eligible control, and clustering is by panel
        unit.
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
        self.dr_method = self._ps_method

        self.design = str(design).lower()
        if self.design not in {"absorbing", "switching"}:
            raise ValueError("design must be 'absorbing' or 'switching'.")
        if self.design == "absorbing":
            self.clean_control = (
                "not_yet_treated" if control_group is None
                else str(control_group).lower()
            )
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
            self.left_censoring = "error"
        else:
            if control_group is not None:
                raise ValueError(
                    "The switching control group is fixed by the formal design; "
                    "do not supply control_group."
                )
            if stabilization_window is None:
                raise ValueError(
                    "design='switching' requires stabilization_window=L."
                )
            if isinstance(stabilization_window, bool):
                raise ValueError("stabilization_window must be a positive integer.")
            L = int(stabilization_window)
            if L != stabilization_window or L < 1:
                raise ValueError("stabilization_window must be a positive integer.")
            self.clean_control = "stabilized"
            self.effect_stabilization = L
            # Conservative formal sample: remove units whose initial treated
            # spell is left-censored, with an explicit warning from prepare_panel.
            self.left_censoring = "drop"

        try:
            raw_h_lo, raw_h_hi = horizons[0], horizons[1]
            h_lo, h_hi = int(raw_h_lo), int(raw_h_hi)
        except Exception as exc:
            raise ValueError("horizons must be a pair (min_h, max_h).") from exc
        if (
            isinstance(raw_h_lo, bool)
            or isinstance(raw_h_hi, bool)
            or h_lo != raw_h_lo
            or h_hi != raw_h_hi
        ):
            raise ValueError("horizons must contain two integers.")
        if h_lo > -1 or h_hi < 0 or h_hi < h_lo:
            raise ValueError(
                "horizons must include the base h=-1 and at least h=0."
            )
        self.max_pre = abs(h_lo)
        self.max_post = h_hi
        self.base_period = -1
        self.anticipation = 0

        if post_window is None:
            self.policy_window = None
        else:
            try:
                raw_lo, raw_hi = post_window[0], post_window[1]
                lo, hi = int(raw_lo), int(raw_hi)
            except Exception as exc:
                raise ValueError("post_window must be a pair (lo, hi).") from exc
            if (
                isinstance(raw_lo, bool)
                or isinstance(raw_hi, bool)
                or lo != raw_lo
                or hi != raw_hi
            ):
                raise ValueError("post_window must contain two integers.")
            if lo < 0 or hi < lo or hi > self.max_post:
                raise ValueError(
                    "post_window must satisfy 0 <= lo <= hi <= max horizon."
                )
            self.policy_window = (lo, hi)

        self.inference = str(inference).lower()
        if self.inference not in {"cluster", "multiplier"}:
            raise ValueError("inference must be 'cluster' or 'multiplier'.")
        if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap:
            raise ValueError("n_bootstrap must be an integer.")
        self.n_bootstrap = int(n_bootstrap)
        if self.inference == "multiplier" and self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1.")
        self.bootstrap_weights = "rademacher"
        self.alpha = float(alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        self.seed = seed

        # Fixed features of the formal estimator; intentionally not public
        # choices in v0.7.2.
        self.include_lagged_outcome_change = False
        self.n_lagged_outcome_changes = 0
        self.ps_with_time_fe = True
        self.ps_time_varying_slopes = False
        self.ps_clip = None
        self.lag_covariates = False
        self.control_pool = "stabilized_all"
        self.switch_in = "sustained"
        self.control_window = "horizon"
        self.time_policy = "calendar"

    # ------------------------------------------------------------------
    # Dispatch to semiparametric estimator
    # ------------------------------------------------------------------

    def _fit_one(
        self,
        local_sample: pd.DataFrame,
        unit: str,
        time: str,
        ps_controls: List[str],
        or_controls: List[str],
        compute_influence: bool,
    ) -> dict:
        """Dispatch to the appropriate semiparametric estimator."""
        if self._family == "ra":
            return _compute_ra(
                local_sample, unit, time, or_controls, compute_influence
            )
        if self._family == "ipw":
            return _compute_ipw(
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
        if self._ps_method == "generic":
            return _compute_dr_generic(
                local_sample,
                unit,
                time,
                ps_controls,
                or_controls,
                self.ps_clip,
                self.ps_with_time_fe,
                self.ps_time_varying_slopes,
                compute_influence,
            )
        return _compute_dr_improved(
            local_sample,
            unit,
            time,
            ps_controls,
            or_controls,
            self.ps_clip,
            self.ps_with_time_fe,
            self.ps_time_varying_slopes,
            compute_influence,
        )

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
    ) -> dict:
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
            nonabsorbing=(self.clean_control == "stabilized"),
            n_lagged_outcome_changes=(
                self.n_lagged_outcome_changes
                if self.include_lagged_outcome_change
                else 0
            ),
            left_censoring=self.left_censoring,
            time_policy=self.time_policy,
        )
        # The reported grid is the requested grid. Data-unavailable horizons
        # remain explicit NaN rows rather than disappearing from the result.
        max_pre, max_post = self.max_pre, self.max_post
        if self.effect_stabilization is not None:
            df = precompute_ccs(
                df, unit, int(self.effect_stabilization), max_pre, max_post
            )

        ps_controls = _control_columns(
            ps_base_covariates,
            self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        )
        or_controls = _control_columns(
            or_base_covariates,
            self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        )
        stack_controls = list(dict.fromkeys(ps_controls + or_controls))
        stack_user_covariates = all_base_covariates
        z = z_crit(self.alpha)
        cluster_universe = pd.Index(df[unit].dropna().unique())
        psi_by_h: Dict[int, pd.Series] = {}
        rows: list[dict] = []
        unsupported_horizons: dict[int, str] = {}
        nuisance_diagnostics: dict[int, dict] = {}

        pre_horizons = list(range(-max_pre, -self.anticipation))
        post_horizons = list(range(0, max_post + 1))
        if (
            isinstance(self.base_period, int)
            and self.base_period == -1
            and -1 in pre_horizons
        ):
            pre_horizons.remove(-1)

        for h in pre_horizons + post_horizons:
            local = build_local_sample(
                df, outcome, unit, time, h, self.base_period,
                self.clean_control, self.effect_stabilization, stack_controls,
                user_covariates=stack_user_covariates,
                lag_covariates=self.lag_covariates,
                fixed_composition_H=None,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
            )
            if (
                local.empty
                or local["D_local"].sum() == 0
                or (local["D_local"] == 0).sum() == 0
            ):
                unsupported_horizons[int(h)] = "empty_or_one_sided_local_stack"
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cluster_universe)
                continue

            # The formal horizon target is defined on supported reference
            # dates: each retained date contains at least one eligible event
            # and one eligible clean control. This same stack is used by all
            # five estimators, including RA.
            cell = local.groupby(time)["D_local"].agg(["sum", "count"])
            supported_times = cell.index[
                (cell["sum"] > 0) & (cell["sum"] < cell["count"])
            ]
            local = local.loc[local[time].isin(supported_times)].copy()
            if (
                local.empty
                or local["D_local"].sum() == 0
                or (local["D_local"] == 0).sum() == 0
            ):
                unsupported_horizons[int(h)] = "no_supported_reference_date"
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cluster_universe)
                continue

            event_mask = local["D_local"] == 1
            control_mask = ~event_mask
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
            try:
                fit = self._fit_one(
                    local, unit, time, ps_controls, or_controls, compute_influence
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
                if key.startswith("ipt_")
            }
            psi_by_h[int(h)] = fit.get(
                "psi_by_cluster", pd.Series(0.0, index=cluster_universe)
            ).reindex(cluster_universe, fill_value=0.0)
            rows.append(_make_row(h, est, se, z, counts=counts))

        if isinstance(self.base_period, int) and self.base_period == -1:
            rows.append({
                "horizon": -1, "estimate": 0.0, "se": 0.0,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": 0.0, "ci_upper": 0.0,
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

        event_df = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        if unsupported_horizons:
            details = ", ".join(
                f"h={h} ({reason})" for h, reason in sorted(unsupported_horizons.items())
            )
            warnings.warn(
                "Some horizons were not identified and were returned as NaN: " + details,
                stacklevel=2,
            )

        # Scalar summaries
        scalar_rows: list[dict] = []
        scalar_specs: dict = {}
        scalar_missing_horizons: list[int] = []
        if self.policy_window is not None:
            lo, hi = self.policy_window
            requested = list(range(lo, hi + 1))
            by_h = event_df.set_index("horizon")
            scalar_missing_horizons = [
                h for h in requested
                if h not in by_h.index or not np.isfinite(by_h.loc[h, "estimate"])
            ]
            if scalar_missing_horizons:
                warnings.warn(
                    "The prespecified post-window scalar was not estimated "
                    "because these requested horizons are unsupported: "
                    f"{scalar_missing_horizons}.",
                    stacklevel=2,
                )
            else:
                weights = np.full(len(requested), 1.0 / len(requested))
                term = f"ATT policy-window [{lo},{hi}]"
                scalar_specs[term] = (requested, weights)
                estimates = by_h.loc[requested, "estimate"].to_numpy(dtype=float)
                estimate = float(np.dot(weights, estimates))
                se = np.nan
                if compute_influence:
                    influence = np.column_stack([
                        psi_by_h[h].reindex(
                            cluster_universe, fill_value=0.0
                        ).to_numpy(dtype=float)
                        for h in requested
                    ]) @ weights
                    se = se_from_influence(influence)
                t_stat = estimate / se if np.isfinite(se) and se > 0 else np.nan
                scalar_rows.append({
                    "term": term,
                    "estimate": estimate,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": (
                        p_value_two_sided(t_stat)
                        if np.isfinite(t_stat) else np.nan
                    ),
                    "ci_lower": estimate - z * se if np.isfinite(se) else np.nan,
                    "ci_upper": estimate + z * se if np.isfinite(se) else np.nan,
                    "n_horizons": len(requested),
                })

        metadata: dict = {
            "package_version": "0.7.2",
            "estimator_identity": f"drlpdid_{self.estimation_method}",
            "estimator_label": (
                "DRLPDID-RA"
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
            "control_group": (
                self.clean_control if self.design == "absorbing" else None
            ),
            "stabilization_window": self.effect_stabilization,
            "horizons": (-self.max_pre, self.max_post),
            "evaluated_horizons": (-max_pre, max_post),
            "base_period": -1,
            "post_window": self.policy_window,
            "post_window_missing_horizons": scalar_missing_horizons,
            "support": "reference_dates_with_events_and_clean_controls",
            "composition": "as_observed",
            "cluster_level": "panel_unit",
            "inference": self.inference,
            "multiplier_weights": (
                "rademacher" if self.inference == "multiplier" else None
            ),
            "event_weighting": (
                "eligible_switch_in_events"
                if self.design == "switching" else "treated_entries"
            ),
            "covariates": base_covariates,
            "ps_covariates": ps_base_covariates,
            "or_covariates": or_base_covariates,
            "unsupported_horizons": unsupported_horizons,
            "nuisance_diagnostics": nuisance_diagnostics,
        }

        return {
            "panel": df,
            "event_study": event_df,
            "scalars": pd.DataFrame(scalar_rows),
            "scalar_specs": scalar_specs,
            "psi_by_h": psi_by_h,
            "cluster_universe": cluster_universe,
            "n_obs": int(df.shape[0]),
            "n_treated_units": int(df.loc[df["_treated_ever"] == 1, unit].nunique()),
            "n_control_units": int(df.loc[df["_never_treated"] == 1, unit].nunique()),
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
        first_treat: Optional[str] = None,
        treatment: Optional[str] = None,
        covariates: Optional[List[str]] = None,
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
        first_treat : str, optional
            First-treatment period column (0 for never-treated).
        treatment : str, optional
            Binary treatment indicator column.
        covariates : list of str, optional
            Predetermined covariates used in the local nuisance models.

        Returns
        -------
        DRLPDIDResults
            Result object with ``event_study`` (including simultaneous CIs
            when ``inference='multiplier'``), ``scalars``, and
            ``print_summary()``.
        """
        if self.design == "absorbing" and first_treat is None:
            raise ValueError(
                "design='absorbing' requires first_treat. Use 0 for "
                "never-treated units."
            )
        if self.design == "switching" and treatment is None:
            raise ValueError(
                "design='switching' requires the observed binary treatment path."
            )
        compute_influence = self.inference in {"cluster", "multiplier"}
        core = self._fit_core(
            data, outcome, unit, time, first_treat, treatment, covariates,
            compute_influence=compute_influence,
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
                    "simultaneous_band_horizons": tuple(band_horizons),
                }
            })
        event_study_stable = pd.DataFrame()

        return DRLPDIDResults(
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
    z: float,
    counts: Optional[dict] = None,
) -> dict:
    t_stat = est / se if np.isfinite(est) and np.isfinite(se) and se > 0 else np.nan
    p_val = p_value_two_sided(t_stat) if np.isfinite(t_stat) else np.nan
    row = {
        "horizon": int(h),
        "estimate": est,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": est - z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
        "ci_upper": est + z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
    }
    if counts:
        row.update(counts)
    return row
