"""Benchmark LP-DiD estimator following Dube, Girardi, Jordà & Taylor (2025).

A single :class:`LPDID` class covers all three target estimands and both
absorbing and non-absorbing treatment settings described in the paper.

Target estimands (``target_estimand``)
---------------------------------------
``'vw'``
    Variance-weighted ATT — default OLS estimand (Section 3.1).
    Numerically equivalent to the stacked DiD of Cengiz et al. (2019).
``'rw'``
    Equally-weighted ATT via Frisch–Waugh–Lovell reweighting (Section 3.3).
    Numerically equivalent to Callaway & Sant'Anna (2020).
``'ra'``
    Equally-weighted ATT via regression-adjustment imputation (Section 3.3).
    Equivalent to Borusyak, Jaravel & Spiess (2024) with ``base_period='all_pre'``.

Clean-control conditions (``clean_control``)
---------------------------------------------
``'not_yet_treated'``  (default)
    Absorbing treatment. Controls: :math:`D_{i,t+h}=0` (eq. 8).
``'never_treated'``
    Absorbing treatment. Controls: units with :math:`p_i=\\infty` only.
``'first_entry'``
    **Non-absorbing treatment** (Section 4.2.2, eq. 12).
    Treated: first-time entrants at *t* that stay treated through :math:`t+h`.
    Controls: units untreated at :math:`t+h`.
    Requires ``nonabsorbing=True``.
``'stabilized'``
    **Non-absorbing treatment** with effect-stabilization horizon *L*
    (Section 4.2.3, eq. 13).
    Requires ``effect_stabilization=L`` and ``nonabsorbing=True``.

References
----------
Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025).
    A local projections approach to difference-in-differences.
    *Journal of Applied Econometrics*, 40, 741–758.
    https://doi.org/10.1002/jae.70000
"""

from __future__ import annotations

from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ._inference import (
    run_cluster_bootstrap,
    run_multiplier_bootstrap,
    se_from_influence,
    stacked_influence,
)
from ._panel_utils import (
    BasePeriod,
    build_local_sample,
    check_columns,
    coerce_base_period,
    compute_base_series,
    infer_windows,
    lag,
    lead,
    p_value_two_sided,
    prepare_panel,
    precompute_ccs,
    z_crit,
)
from ._results import LPDIDResults
from ._errors import ExperimentalFeatureError


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _nan_row(h: int) -> dict:
    return {
        "horizon": int(h), "estimate": np.nan, "se": np.nan,
        "t_stat": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
    }


def _make_row(h: int, est: float, se: float, z: float) -> dict:
    t = est / se if np.isfinite(est) and np.isfinite(se) and se > 0 else np.nan
    p = p_value_two_sided(t) if np.isfinite(t) else np.nan
    return {
        "horizon": int(h), "estimate": est, "se": se, "t_stat": t, "p_value": p,
        "ci_lower": est - z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
        "ci_upper": est + z * se if np.isfinite(est) and np.isfinite(se) else np.nan,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cluster_fit(model, groups: pd.Series):
    """OLS/WLS with cluster-robust SE; falls back to standard SE on failure."""
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as exc:
        warnings.warn(
            f"Cluster SE failed ({type(exc).__name__}: {exc}). "
            "Falling back to standard errors.",
            stacklevel=2,
        )
        return model.fit()


def _ctrl_cols(covariates: Optional[List[str]], ldy: bool, n: int) -> List[str]:
    cols: List[str] = []
    if ldy and n > 0:
        cols.extend([f"ldy{k}" for k in range(1, int(n) + 1)])
    if covariates:
        cols.extend(list(covariates))
    return cols


def _formula(lhs: str, treat: Optional[str], time: str, controls: List[str]) -> str:
    rhs = []
    if treat is not None:
        rhs.append(treat)
    rhs.append(f"C({time})")
    rhs.extend(controls or [])
    return lhs + " ~ " + " + ".join(rhs)


# ---------------------------------------------------------------------------
# VW / RW local estimation (OLS / WLS)
# ---------------------------------------------------------------------------

def _rw_weights(local: pd.DataFrame, time: str) -> pd.Series:
    """Frisch–Waugh–Lovell reweighting scores for the equally-weighted ATT.

    Residualizes ``D_local`` on time dummies; the residuals act as
    inverse-variance weights that deliver the equally-weighted ATT
    (Dube et al. 2025, Appendix B).
    """
    p_t = local.groupby(time)["D_local"].transform("mean")
    resid = local["D_local"].to_numpy(dtype=float) - p_t.to_numpy(dtype=float)
    num = np.where(local["D_local"].to_numpy() == 1, resid, np.nan)
    den = float(np.nansum(num))
    if den <= 1e-12 or not np.isfinite(den):
        raise ValueError(
            "Cannot compute RW weights: zero denominator. "
            "Check that treated-unit variation exists within time cells."
        )
    wt = num / den
    gw = pd.Series(wt, index=local.index).groupby(local[time]).transform("max")
    out = pd.Series(np.nan, index=local.index, dtype=float)
    valid = np.isfinite(gw) & (gw > 0)
    out.loc[valid] = 1.0 / gw.loc[valid]
    return out


def _fit_linear(
    local: pd.DataFrame, unit: str, time: str,
    controls: List[str], estimand: str,
    pre_rw: Optional[pd.Series] = None,
) -> dict:
    """OLS (VW) or WLS (RW) local projection at one horizon."""
    s = local.copy()
    f = _formula("outcome_local", "D_local", time, controls)
    if estimand == "rw":
        s["_w"] = (
            pre_rw.reindex(s.index).to_numpy(dtype=float)
            if pre_rw is not None else _rw_weights(s, time)
        )
        s = s.loc[np.isfinite(s["_w"]) & (s["_w"] > 0)].copy()
        model = smf.wls(f, data=s, weights=s["_w"])
    else:
        s["_w"] = 1.0
        model = smf.ols(f, data=s)
    fit = _cluster_fit(model, s[unit])
    est = float(fit.params["D_local"])
    se = float(fit.bse["D_local"]) if "D_local" in fit.bse.index else np.nan
    return {"estimate": est, "se": se}


def _influence_linear(
    local: pd.DataFrame, unit: str, time: str,
    controls: List[str], estimand: str,
    pre_rw: Optional[pd.Series] = None,
) -> dict:
    """Stacked GMM influence functions for VW/RW."""
    s = local.copy()
    if estimand == "rw":
        s["_w"] = (
            pre_rw.reindex(s.index).to_numpy(dtype=float)
            if pre_rw is not None else _rw_weights(s, time)
        )
        s = s.loc[np.isfinite(s["_w"]) & (s["_w"] > 0)].copy()
    else:
        s["_w"] = 1.0

    _, X_df = patsy.dmatrices(
        _formula("outcome_local", "D_local", time, controls),
        s, return_type="dataframe",
    )
    y = s["outcome_local"].to_numpy(dtype=float)
    X = np.asarray(X_df, dtype=float)
    w = s["_w"].to_numpy(dtype=float)
    theta = np.asarray(sm.WLS(y, X, weights=w).fit().params, dtype=float)
    idx = list(X_df.columns).index("D_local")

    infl = stacked_influence(
        theta,
        lambda th: w[:, None] * X * (y - X @ th)[:, None],
        s[unit].to_numpy(),
        target_grad=lambda th: np.eye(len(th))[idx],
    )
    psi = infl["psi"].reshape(-1)
    return {
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(psi, index=pd.Index(infl["cluster_labels"])),
    }


# ---------------------------------------------------------------------------
# RA local estimation (regression-adjustment imputation)
# ---------------------------------------------------------------------------

def _fit_ra(
    local: pd.DataFrame, unit: str, time: str, controls: List[str]
) -> dict:
    """RA-LP-DiD imputation at one horizon (Dube et al. 2025, eq. 11).

    Estimates :math:`\\hat{m}_0` on clean controls; the ATT is the mean
    treated residual :math:`\\bar{Y}_1 - \\overline{\\hat{m}_0}`.
    """
    # RA imputation requires a control in every time cell used: m0 carries the
    # time fixed effect, which is unidentified for a treated-only cell. Restrict
    # to time cells with at least one control (treated units in control-less
    # cells cannot be imputed and are excluded from the RA average). VW/RW keep
    # such cells to match reghdfe; RA cannot.
    _hc = local.loc[local["D_local"] == 0].groupby(time).size()
    local = local.loc[local[time].isin(_hc.loc[_hc > 0].index)].copy()
    ctrl = local.loc[local["D_local"] == 0].copy()
    if ctrl.empty:
        raise ValueError("No clean controls for RA estimation.")
    fit_or = smf.ols(_formula("outcome_local", None, time, controls), data=ctrl).fit()
    m0 = np.asarray(fit_or.predict(local), dtype=float)
    resid = local["outcome_local"].to_numpy(dtype=float) - m0
    return {"estimate": float(np.mean(resid[local["D_local"].to_numpy() == 1]))}


def _influence_ra(
    local: pd.DataFrame, unit: str, time: str, controls: List[str]
) -> dict:
    """Stacked GMM influence functions for the RA estimator.

    Moment system:

    .. math::

        m_1(\\boldsymbol{\\beta}) &= (1-D_i)(Y_i-Z_i\\boldsymbol{\\beta})Z_i \\\\
        m_2(\\mu_1,\\boldsymbol{\\beta}) &= D_i(Y_i-Z_i\\boldsymbol{\\beta}-\\mu_1)
    """
    # Same control-in-every-cell restriction as _fit_ra (RA imputation cannot
    # use treated-only time cells).
    _hc = local.loc[local["D_local"] == 0].groupby(time).size()
    local = local.loc[local[time].isin(_hc.loc[_hc > 0].index)].copy()
    y = local["outcome_local"].to_numpy(dtype=float)
    d = local["D_local"].to_numpy(dtype=float)
    fit_or = smf.ols(
        _formula("outcome_local", None, time, controls),
        data=local.loc[d == 0],
    ).fit()
    Z = np.asarray(
        patsy.build_design_matrices(
            [fit_or.model.data.design_info], local, return_type="dataframe"
        )[0],
        dtype=float,
    )
    beta = np.asarray(fit_or.params, dtype=float)
    mu1 = float(np.mean((y - Z @ beta)[d == 1]))
    theta = np.concatenate([beta, [mu1]])

    def moments(th):
        pb = Z.shape[1]
        b, m = th[:pb], th[pb]
        r = y - Z @ b
        return np.column_stack([
            ((1 - d))[:, None] * Z * r[:, None],
            (d * (r - m))[:, None],
        ])

    infl = stacked_influence(
        theta, moments, local[unit].to_numpy(),
        target_grad=lambda th: np.eye(len(th))[-1],
    )
    psi = infl["psi"].reshape(-1)
    return {
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(psi, index=pd.Index(infl["cluster_labels"])),
    }


# ---------------------------------------------------------------------------
# Unified horizon dispatch
# ---------------------------------------------------------------------------

def _fit_horizon(
    local: pd.DataFrame,
    unit: str, time: str, controls: List[str],
    estimand: str, compute_influence: bool,
    cluster_universe: pd.Index,
    pre_rw: Optional[pd.Series] = None,
) -> tuple[float, float, pd.Series]:
    """Estimate ATT-h, SE, and cluster influence for one horizon."""
    zero = pd.Series(0.0, index=cluster_universe)
    if estimand == "ra":
        est = _fit_ra(local, unit, time, controls)["estimate"]
        se = np.nan
        psi = zero
        if compute_influence:
            infl = _influence_ra(local, unit, time, controls)
            se = infl["se"]
            psi = infl["psi_by_cluster"].reindex(cluster_universe, fill_value=0.0)
    else:
        r = _fit_linear(local, unit, time, controls, estimand, pre_rw)
        est, se = r["estimate"], r["se"]
        psi = zero
        if compute_influence:
            infl = _influence_linear(local, unit, time, controls, estimand, pre_rw)
            se = infl["se"]
            psi = infl["psi_by_cluster"].reindex(cluster_universe, fill_value=0.0)
    return est, se, psi


# ---------------------------------------------------------------------------
# Scalar summaries
# ---------------------------------------------------------------------------

def _avg_scalar(
    event_df: pd.DataFrame, psi_by_h: Dict[int, pd.Series],
    cluster_universe: pd.Index, z: float, compute_influence: bool,
) -> list[dict]:
    post = event_df.loc[
        (event_df["horizon"] >= 0) & np.isfinite(event_df["estimate"])
    ]
    if post.empty:
        return []
    post_h = post["horizon"].astype(int).tolist()
    avg = float(np.mean(post["estimate"].to_numpy(dtype=float)))
    se = np.nan
    if compute_influence:
        Psi = np.column_stack([
            psi_by_h[h].reindex(cluster_universe, fill_value=0.0).to_numpy(dtype=float)
            for h in post_h
        ])
        se = se_from_influence(np.mean(Psi, axis=1))
    t = avg / se if np.isfinite(se) and se > 0 else np.nan
    return [{
        "term": "ATT avg", "estimate": avg, "se": se, "t_stat": t,
        "p_value": p_value_two_sided(t) if np.isfinite(t) else np.nan,
        "ci_lower": avg - z * se if np.isfinite(se) else np.nan,
        "ci_upper": avg + z * se if np.isfinite(se) else np.nan,
    }]


def _build_pooled_local(
    df: pd.DataFrame, outcome: str, unit: str, time: str,
    post_window: int, base_period: BasePeriod,
    clean_control: str, effect_stabilization: Optional[int],
    extra_cols: List[str],
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
    control_pool: str = "stabilized_all",
    switch_in: str = "sustained",
    control_window: str = "horizon",
) -> pd.DataFrame:
    """Pooled (averaged-outcome) local sample for Section 3.5."""
    s = df.copy()
    # Lag user covariates to t-1 before any computation (same logic as
    # build_local_sample). ldy* controls are already lags — do not re-lag.
    if lag_covariates and user_covariates:
        for c in user_covariates:
            if c in s.columns:
                s[c] = lag(s, unit, c, 1)
    post_list = [lead(s, unit, outcome, h) for h in range(0, post_window + 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s["_pooled_post_mean"] = np.nanmean(np.column_stack(post_list), axis=1)
    s["outcome_local"] = s["_pooled_post_mean"] - compute_base_series(
        s, outcome, unit, base_period
    )

    if clean_control == "never_treated":
        treat_mask = s["D_treat"] == 1
        ctrl_mask = s["_never_treated"] == 1
    elif clean_control == "not_yet_treated":
        treat_mask = s["D_treat"] == 1
        ctrl_mask = lead(s, unit, "_treat", post_window).fillna(1.0) == 0
    elif clean_control == "first_entry":
        newly = s["D_treat"] == 1
        first_t = s["_first_treat"] == s[time]
        stays = lead(s, unit, "_treat", post_window).fillna(0) == 1
        treat_mask = newly & first_t & stays
        ctrl_mask = lead(s, unit, "_treat", post_window).fillna(1.0) == 0
    elif clean_control == "stabilized":
        _ccs_col = "CCS_0" if control_window == "entry" else f"CCS_{post_window}"
        stable = s[_ccs_col] == 1
        # Treated requires a clean pre-window (CCS_0) per eq. (13), matching
        # build_local_sample and the Stata `tdemoc==1 & CCS_0==1` condition.
        treat_mask = (s["D_treat"] == 1) & (s["CCS_0"] == 1)
        ctrl_mask = (s["D_treat"] == 0) & stable
        if "_treat_obs" in s.columns:
            lag_treat_t = lag(s, unit, "_treat", 1)
            lag_obs_t = lag(s, unit, "_treat_obs", 1)
            treat_mask = (
                treat_mask & (s["_treat_obs"] == 1)
                & (lag_treat_t == 0) & (lag_obs_t == 1)
            )
            ctrl_mask = ctrl_mask & (s["_treat_obs"] == 1)
        if control_pool == "untreated_only":
            lag_treat = lag(s, unit, "_treat", 1)
            ctrl_mask = ctrl_mask & (s["_treat"] == 0) & (lag_treat == 0)
            if "_treat_obs" in s.columns:
                ctrl_mask = ctrl_mask & (lag(s, unit, "_treat_obs", 1) == 1)
        # Sustained switch-in (eq. 13): treated stays treated through the whole
        # pooled post window [0, post_window].
        if switch_in == "sustained":
            stay_on = pd.Series(True, index=s.index)
            for j in range(0, post_window + 1):
                stay_on = stay_on & (lead(s, unit, "_treat", j).fillna(0) == 1)
                if "_treat_obs" in s.columns:
                    stay_on = stay_on & (lead(s, unit, "_treat_obs", j).fillna(0) == 1)
            treat_mask = treat_mask & stay_on
    else:
        raise ValueError(f"Invalid clean_control: '{clean_control}'.")

    s = s.loc[treat_mask | ctrl_mask].copy()
    s["D_local"] = treat_mask.loc[s.index].astype(int)
    needed = [unit, time, "D_local", "outcome_local"] + list(extra_cols)
    s = s.dropna(subset=[c for c in needed if c in s.columns]).copy()
    s = s.loc[np.isfinite(s["outcome_local"])].copy()
    if s.empty:
        return s.reset_index(drop=True)
    # Exact-replication convention: keep treated-only time cells (see
    # build_local_sample); require only that some control exists overall.
    if (s["D_local"] == 0).sum() == 0:
        return s.iloc[0:0].reset_index(drop=True)
    return s.reset_index(drop=True)


def _pooled_scalar(
    df: pd.DataFrame, outcome: str, unit: str, time: str,
    event_df: pd.DataFrame, base_period: BasePeriod,
    clean_control: str, effect_stabilization: Optional[int],
    controls: List[str], estimand: str, z: float, compute_influence: bool,
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
    control_pool: str = "stabilized_all",
    switch_in: str = "sustained",
    control_window: str = "horizon",
) -> list[dict]:
    finite_post = event_df.loc[
        (event_df["horizon"] >= 0) & np.isfinite(event_df["estimate"]), "horizon"
    ].tolist()
    if not finite_post:
        return []
    pw = int(max(finite_post))
    pl = _build_pooled_local(
        df, outcome, unit, time, pw, base_period,
        clean_control, effect_stabilization, controls,
        user_covariates=user_covariates,
        lag_covariates=lag_covariates,
        control_pool=control_pool,
        switch_in=switch_in,
        control_window=control_window,
    )
    if pl.empty or pl["D_local"].sum() == 0:
        return []

    if estimand == "ra":
        est = _fit_ra(pl, unit, time, controls)["estimate"]
        se = np.nan
        if compute_influence:
            se = _influence_ra(pl, unit, time, controls)["se"]
    else:
        r = _fit_linear(pl, unit, time, controls, estimand)
        est, se = r["estimate"], r["se"]
        if compute_influence:
            se = _influence_linear(pl, unit, time, controls, estimand)["se"]

    t = est / se if np.isfinite(se) and se > 0 else np.nan
    return [{
        "term": "ATT pooled", "estimate": est, "se": se, "t_stat": t,
        "p_value": p_value_two_sided(t) if np.isfinite(t) else np.nan,
        "ci_lower": est - z * se if np.isfinite(se) else np.nan,
        "ci_upper": est + z * se if np.isfinite(se) else np.nan,
    }]


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def _cluster_bootstrap(
    estimator: "LPDID",
    data: pd.DataFrame, outcome: str, unit: str, time: str,
    first_treat: Optional[str], treatment: Optional[str],
    covariates: Optional[List[str]],
    event_df: pd.DataFrame, scalars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(estimator.seed)

    orig = {
        "event_index": event_df["horizon"].tolist(),
        "event_hat": dict(zip(event_df["horizon"].tolist(), event_df["estimate"].tolist())),
        "scalar_terms": scalars["term"].tolist(),
        "scalar_hat": dict(zip(scalars["term"].tolist(), scalars["estimate"].tolist())),
    }

    def _refit(boot_df):
        c = estimator._fit_core(
            boot_df, outcome, unit, time, first_treat, treatment,
            covariates, compute_influence=False,
        )
        return {"event_study": c["event_study"], "scalars": c["scalars"]}

    boot = run_cluster_bootstrap(
        _refit, data.copy(), unit, estimator.n_bootstrap, estimator.alpha, rng, orig
    )

    for df_ref, key, on in [
        (event_df, "event_boot", "horizon"),
        (scalars, "scalar_boot", "term"),
    ]:
        merged = df_ref.merge(boot[key], on=on, how="left", suffixes=("", "_b"))
        for c in ["se", "ci_lower", "ci_upper", "p_value", "t_stat"]:
            if f"{c}_b" in merged.columns:
                merged[c] = merged[f"{c}_b"]
                merged.drop(columns=[f"{c}_b"], inplace=True)
        if on == "horizon":
            event_df = merged
        else:
            scalars = merged
    return event_df, scalars


# ---------------------------------------------------------------------------
# LPDID — the single public estimator
# ---------------------------------------------------------------------------

class LPDID:
    """LP-DiD estimator: variance-weighted, reweighted, and regression-adjusted.

    Implements all three target estimands from Dube et al. (2025, Sections 3.1
    and 3.3), for both absorbing and non-absorbing treatment settings
    (Sections 4.2.2 and 4.2.3).

    Parameters
    ----------
    target_estimand : {'vw', 'rw', 'ra'}
        * ``'vw'`` — variance-weighted ATT (default OLS estimand, Section 3.1).
        * ``'rw'`` — equally-weighted ATT via FWL reweighting (Section 3.3);
          equivalent to Callaway & Sant'Anna (2020).
        * ``'ra'`` — equally-weighted ATT via regression-adjustment imputation
          (Section 3.3); equivalent to Borusyak, Jaravel & Spiess (2024)
          with ``base_period='all_pre'``.
    base_period : int, list of int, or 'all_pre'
        Baseline for :math:`Y_{i,t+h} - Y_{i,s}`:

        * negative integer (e.g. ``-1``) — single pre-treatment lag.
        * list of negative integers — average of those lags (PMD).
        * ``'all_pre'`` — average of all pre-treatment outcomes.
    clean_control : {'not_yet_treated', 'never_treated', 'first_entry', \
'stabilized'}
        Control-group restriction:

        * ``'not_yet_treated'`` (default) — absorbing; :math:`D_{i,t+h}=0`
          (eq. 8).
        * ``'never_treated'`` — absorbing; :math:`p_i=\\infty` only.
        * ``'first_entry'`` — **non-absorbing**; first-time entrants that
          stay treated through :math:`t+h` (Section 4.2.2, eq. 12).
          Requires ``nonabsorbing=True``.
        * ``'stabilized'`` — **non-absorbing** with effect-stabilization *L*
          (Section 4.2.3, eq. 13).
          Requires ``effect_stabilization=L`` and ``nonabsorbing=True``.
    nonabsorbing : bool
        Set to ``True`` to allow treatment to switch off (non-absorbing).
        Required when ``clean_control`` is ``'first_entry'`` or
        ``'stabilized'``.
    effect_stabilization : int or None
        Horizon *L* after which dynamic effects are assumed constant.
        Required when ``clean_control='stabilized'``.
    anticipation : int
        Periods before treatment excluded from pre-trend estimation (default 0).
    include_lagged_outcome_change : bool
        Include lagged first-differences of the outcome as controls.
    n_lagged_outcome_changes : int
        Number of lagged outcome changes (ignored unless
        ``include_lagged_outcome_change=True``).
    inference : {'cluster', 'multiplier', 'cluster_bootstrap'}
        ``'cluster'``: analytic cluster-robust SE via stacked GMM influence
        functions. ``'multiplier'``: the same pointwise cluster-robust
        inference plus simultaneous multiplier-bootstrap bands.
        ``'cluster_bootstrap'``: paired cluster bootstrap.
    n_bootstrap : int
        Bootstrap replications (default 499).
    bootstrap_weights : str
        Weight distribution for the bootstrap: ``'mammen'`` (default),
        ``'rademacher'``, or ``'webb'``.
    alpha : float
        Significance level (default 0.05 → 95 % CIs).
    seed : int or None
        Random seed for reproducibility.
    max_pre, max_post : int or None
        Maximum pre/post horizons; inferred from data when ``None``.
    lag_covariates : bool
        If ``True`` (recommended when covariates are time-varying), all
        user-supplied covariates are automatically lagged by one period
        (:math:`X_{i,t-1}`) inside each local comparison stack before
        entering the outcome regression or propensity-score model.
        This ensures covariates are predetermined (pre-treatment for treated
        units), as required by the conditional parallel trends assumption
        (Dube et al. 2025, Section 4.1, Assumptions 3–5).
        Lagged-outcome-change controls (``ldy*``) are **not** re-lagged.
        Default ``False`` to preserve backward compatibility.

    Examples
    --------
    Absorbing treatment — all three estimands:

    >>> from lpdid import LPDID
    >>> res_vw = LPDID(target_estimand='vw').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')
    >>> res_rw = LPDID(target_estimand='rw').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')
    >>> res_ra = LPDID(target_estimand='ra', base_period='all_pre').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')

    Non-absorbing treatment — first-entry condition (Section 4.2.2):

    >>> res_fe = LPDID(
    ...     target_estimand='vw',
    ...     clean_control='first_entry',
    ...     nonabsorbing=True,
    ... ).fit(df, outcome='y', unit='id', time='t', treatment='D')

    Non-absorbing with effect stabilization (Section 4.2.3):

    >>> res_st = LPDID(
    ...     target_estimand='rw',
    ...     clean_control='stabilized',
    ...     effect_stabilization=4,
    ...     nonabsorbing=True,
    ... ).fit(df, outcome='y', unit='id', time='t', treatment='D')
    """

    _VALID_ESTIMANDS = {"vw", "rw", "ra"}
    _VALID_CONTROLS = {
        "not_yet_treated", "never_treated", "first_entry", "stabilized"
    }
    _NONABSORBING_CONTROLS = {"first_entry", "stabilized"}

    def __init__(
        self,
        *,
        target_estimand: str = "vw",
        base_period: BasePeriod = -1,
        clean_control: str = "not_yet_treated",
        nonabsorbing: bool = False,
        effect_stabilization: Optional[int] = None,
        anticipation: int = 0,
        include_lagged_outcome_change: bool = False,
        n_lagged_outcome_changes: int = 0,
        inference: str = "cluster",
        n_bootstrap: int = 499,
        bootstrap_weights: str = "mammen",
        alpha: float = 0.05,
        seed: Optional[int] = None,
        max_pre: Optional[int] = None,
        max_post: Optional[int] = None,
        lag_covariates: bool = False,
        fixed_composition: bool = False,
        support_policy: str = "supported_subset",
        left_censoring: str = "error",
        time_policy: str = "calendar",
    ) -> None:
        self.target_estimand = str(target_estimand).lower()
        if self.target_estimand not in self._VALID_ESTIMANDS:
            raise ValueError(
                f"target_estimand must be one of {self._VALID_ESTIMANDS}."
            )
        self.base_period = coerce_base_period(base_period)
        self.clean_control = str(clean_control).lower()
        if self.clean_control not in self._VALID_CONTROLS:
            raise ValueError(
                f"clean_control must be one of {self._VALID_CONTROLS}."
            )
        self.nonabsorbing = bool(nonabsorbing)
        # Clean-control designs determine whether treatment is non-absorbing.
        if self.clean_control in self._NONABSORBING_CONTROLS:
            self.nonabsorbing = True
        self.effect_stabilization = effect_stabilization
        if self.clean_control == "stabilized" and self.effect_stabilization is None:
            raise ValueError(
                "clean_control='stabilized' requires effect_stabilization."
            )
        self.anticipation = int(max(anticipation, 0))
        self.include_lagged_outcome_change = bool(include_lagged_outcome_change)
        self.n_lagged_outcome_changes = int(max(n_lagged_outcome_changes, 0))
        self.inference = str(inference).lower()
        if self.inference not in {"cluster", "multiplier", "cluster_bootstrap"}:
            raise ValueError(
                "inference must be 'cluster', 'multiplier', or "
                "'cluster_bootstrap'."
            )
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_weights = str(bootstrap_weights).lower()
        self.alpha = float(alpha)
        self.seed = seed
        self.max_pre = None if max_pre is None else int(max_pre)
        self.max_post = None if max_post is None else int(max_post)
        self.lag_covariates = bool(lag_covariates)
        self.fixed_composition = bool(fixed_composition)
        # Certified equation (13) design for stabilized non-absorbing treatment.
        self.control_pool = "stabilized_all"
        self.switch_in = "sustained"
        self.control_window = "horizon"
        self.support_policy = str(support_policy).lower()
        if self.support_policy not in {"strict", "supported_subset"}:
            raise ValueError("support_policy must be 'strict' or 'supported_subset'.")
        self.left_censoring = str(left_censoring).lower()
        self.time_policy = str(time_policy).lower()

    # ------------------------------------------------------------------
    def _prepare(self, data, outcome, unit, time, first_treat, treatment, covariates):
        covariates = list(covariates or [])
        if covariates:
            check_columns(data, covariates)
        df = prepare_panel(
            data, outcome, unit, time, first_treat, treatment,
            nonabsorbing=self.nonabsorbing,
            n_lagged_outcome_changes=(
                self.n_lagged_outcome_changes
                if self.include_lagged_outcome_change else 0
            ),
            left_censoring=self.left_censoring,
            time_policy=self.time_policy,
        )
        return df, covariates

    # ------------------------------------------------------------------
    def _fit_core(
        self, data, outcome, unit, time,
        first_treat, treatment, covariates,
        compute_influence: bool = True,
    ) -> dict:
        df, covariates = self._prepare(
            data, outcome, unit, time, first_treat, treatment, covariates
        )
        max_pre, max_post = infer_windows(
            df, time, self.base_period, self.max_pre, self.max_post
        )
        if self.effect_stabilization is not None:
            df = precompute_ccs(
                df, unit, int(self.effect_stabilization), max_pre, max_post
            )

        controls = _ctrl_cols(
            covariates, self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        )
        z = z_crit(self.alpha)
        cu = pd.Index(df[unit].dropna().unique())  # cluster universe
        psi_by_h: Dict[int, pd.Series] = {}
        rows: list[dict] = []

        # Fixed composition (Section 3.6): hold the unit set across horizons
        # by fixing control membership at t+H and dropping late treated events.
        fc_H: Optional[int] = int(max_post) if self.fixed_composition else None

        # Empty-support diagnostic: collect post-treatment horizons whose
        # clean-control stack is exhausted (no clean controls available).
        empty_support_horizons: list[int] = []

        # Pre-compute h=0 RW weights for pre-period horizons (RW only)
        rw0_map: Optional[pd.Series] = None
        if self.target_estimand == "rw":
            local0 = build_local_sample(
                df, outcome, unit, time, 0, self.base_period,
                self.clean_control, self.effect_stabilization, controls,
                user_covariates=covariates,
                lag_covariates=self.lag_covariates,
                fixed_composition_H=fc_H,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
            )
            if (
                not local0.empty
                and local0["D_local"].sum() > 0
                and (local0["D_local"] == 0).sum() > 0
            ):
                _w = _rw_weights(local0, time)
                rw0_map = pd.Series(_w.values, index=local0.index)
                rw0_map.index = pd.MultiIndex.from_frame(local0[[unit, time]])

        pre_horizons = list(range(-max_pre, -self.anticipation))
        post_horizons = list(range(0, max_post + 1))
        if isinstance(self.base_period, int) and self.base_period in pre_horizons:
            pre_horizons.remove(self.base_period)

        for h in pre_horizons + post_horizons:
            local = build_local_sample(
                df, outcome, unit, time, h, self.base_period,
                self.clean_control, self.effect_stabilization, controls,
                user_covariates=covariates,
                lag_covariates=self.lag_covariates,
                fixed_composition_H=fc_H,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
            )
            if (
                local.empty
                or local["D_local"].sum() == 0
                or (local["D_local"] == 0).sum() == 0
            ):
                # Record post-horizon control-support exhaustion for the
                # consolidated diagnostic (Section 3.6 / referee point on
                # empty clean-control groups at long horizons).
                if h >= 0 and (local.empty or (local["D_local"] == 0).sum() == 0):
                    empty_support_horizons.append(int(h))
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cu)
                continue

            cell = local.groupby(time)["D_local"].agg(["sum", "count"])
            unsupported_times = cell.index[(cell["sum"] > 0) & (cell["sum"] == cell["count"])].tolist()
            if unsupported_times and self.support_policy == "strict":
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cu)
                continue
            if unsupported_times:
                local = local.loc[~local[time].isin(unsupported_times)].copy()

            # Pre-period RW: map h=0 time-cell weights to this stack
            pre_rw: Optional[pd.Series] = None
            if self.target_estimand == "rw" and h < 0 and rw0_map is not None:
                tmp0 = build_local_sample(
                    df, outcome, unit, time, 0, self.base_period,
                    self.clean_control, self.effect_stabilization, controls,
                    user_covariates=covariates,
                    lag_covariates=self.lag_covariates,
                    fixed_composition_H=fc_H,
                    control_pool=self.control_pool,
                    switch_in=self.switch_in,
                    control_window=self.control_window,
                )
                if not tmp0.empty:
                    tmap = (
                        pd.Series(_rw_weights(tmp0, time).values, index=tmp0[time].values)
                        .groupby(level=0).first()
                    )
                    pre_rw = local[time].map(tmap)

            est, se, psi = _fit_horizon(
                local, unit, time, controls, self.target_estimand,
                compute_influence, cu, pre_rw=pre_rw,
            )
            psi_by_h[int(h)] = psi
            rows.append(_make_row(h, est, se, z))

        # Normalized base-period row (estimate = 0 by construction)
        if isinstance(self.base_period, int):
            rows.append({
                "horizon": int(self.base_period), "estimate": 0.0, "se": 0.0,
                "t_stat": np.nan, "p_value": np.nan, "ci_lower": 0.0, "ci_upper": 0.0,
            })
            psi_by_h[int(self.base_period)] = pd.Series(0.0, index=cu)

        event_df = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

        # Consolidated empty-support diagnostic (one warning per fit).
        if empty_support_horizons:
            hs = sorted(set(empty_support_horizons))
            warnings.warn(
                "Clean-control support is exhausted at post-treatment "
                f"horizon(s) {hs}: no clean controls remain, so the ATT is "
                "not identified there (returned as NaN). This typically "
                "occurs at long horizons under absorbing treatment when "
                "every not-yet-treated unit has already been treated; "
                "identification at these horizons generally requires a "
                "never-treated group (clean_control='never_treated').",
                stacklevel=2,
            )

        scalar_rows = _avg_scalar(event_df, psi_by_h, cu, z, compute_influence)
        scalar_rows += _pooled_scalar(
            df, outcome, unit, time, event_df, self.base_period,
            self.clean_control, self.effect_stabilization, controls,
            self.target_estimand, z, compute_influence,
            user_covariates=covariates,
            lag_covariates=self.lag_covariates,
            control_pool=self.control_pool,
            switch_in=self.switch_in,
            control_window=self.control_window,
        )

        return {
            "panel": df, "event_study": event_df,
            "scalars": pd.DataFrame(scalar_rows),
            "psi_by_h": psi_by_h,
            "cluster_universe": cu,
            "n_obs": int(df.shape[0]),
            "n_treated_units": int(df.loc[df["_treated_ever"] == 1, unit].nunique()),
            "n_control_units": int(df.loc[df["_never_treated"] == 1, unit].nunique()),
            "n_cohorts": int(df.loc[df["_first_treat"] > 0, "_first_treat"].nunique()),
            "n_periods": int(df[time].nunique()),
            "fixed_composition": bool(self.fixed_composition),
            "support_policy": self.support_policy,
            "left_censoring": self.left_censoring,
            "time_policy": self.time_policy,
            "nonabsorbing_certified": bool(self.clean_control == "stabilized"),
            "treatment_design": (
                "recurrent_sustained_switch_in"
                if self.clean_control == "stabilized"
                else ("first_entry_switch_in" if self.clean_control == "first_entry" else "absorbing_staggered")
            ),
        }

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
    ) -> LPDIDResults:
        """Fit the LP-DiD estimator.

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
            First treatment period column (0 = never treated).
        treatment : str, optional
            Binary (0/1) treatment indicator; first period with value 1 is
            used as ``first_treat``. For non-absorbing treatment, supply
            ``treatment`` rather than ``first_treat``.
        covariates : list of str, optional
            Pre-treatment covariate columns.

        Returns
        -------
        LPDIDResults
        """
        core = self._fit_core(
            data, outcome, unit, time, first_treat, treatment, covariates,
            compute_influence=(self.inference in {"cluster", "multiplier"}),
        )
        event_df = core["event_study"].copy()
        scalars = core["scalars"].copy()

        multiplier_metadata = None
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
            band_horizons = [
                int(h) for h in event_df.loc[band_mask, "horizon"].tolist()
            ]
            if band_horizons:
                psi = np.column_stack([
                    core["psi_by_h"][h].reindex(
                        core["cluster_universe"], fill_value=0.0
                    ).to_numpy(dtype=float)
                    for h in band_horizons
                ])
                selected = event_df.loc[band_mask]
                mb = run_multiplier_bootstrap(
                    selected["estimate"].to_numpy(dtype=float),
                    psi,
                    selected["se"].to_numpy(dtype=float),
                    self.n_bootstrap,
                    self.bootstrap_weights,
                    self.alpha,
                    rng,
                )
                event_df.loc[band_mask, "multiplier_ci_lower"] = mb["ci_lower"]
                event_df.loc[band_mask, "multiplier_ci_upper"] = mb["ci_upper"]
                event_df.loc[band_mask, "multiplier_p_value"] = mb["p_values"]
                event_df.loc[band_mask, "sim_ci_lower"] = mb["sim_ci_lower"]
                event_df.loc[band_mask, "sim_ci_upper"] = mb["sim_ci_upper"]
                critical_value = mb["cband_crit"]
            else:
                critical_value = np.nan
            multiplier_metadata = {
                "critical_value": critical_value,
                "weight_type": self.bootstrap_weights,
                "n_bootstrap": self.n_bootstrap,
                "simultaneous_band_horizons": tuple(band_horizons),
            }

        if self.inference == "cluster_bootstrap":
            event_df, scalars = _cluster_bootstrap(
                self, data, outcome, unit, time,
                first_treat, treatment, covariates,
                event_df, scalars,
            )

        label = (
            "LPDID-RA"
            if self.target_estimand == "ra"
            else f"LP-DiD ({self.target_estimand.upper()})"
        )
        if self.nonabsorbing:
            label += " [non-absorbing]"

        metadata = {
            "package_version": "0.7.2",
            "estimator_identity": (
                "dube_2025_lpdid_ra"
                if self.target_estimand == "ra"
                else f"dube_2025_lpdid_{self.target_estimand}"
            ),
            "estimation_sample": "dube_clean_control_sample",
            "fixed_composition": core.get("fixed_composition", False),
            "nonabsorbing_certified": bool(self.clean_control == "stabilized"),
            "treatment_design": core.get("treatment_design"),
            "nonabsorbing_switch_in": "sustained" if self.clean_control == "stabilized" else None,
            "nonabsorbing_control_window": "horizon" if self.clean_control == "stabilized" else None,
            "nonabsorbing_control_pool": "stabilized_all" if self.clean_control == "stabilized" else None,
            "estimand_weighting": "eligible_switch_in_events" if self.clean_control == "stabilized" else "treated_entries",
        }
        if multiplier_metadata is not None:
            metadata["multiplier"] = multiplier_metadata

        return LPDIDResults(
            estimator_name=label,
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
            scalars=scalars,
            metadata=metadata,
            target_estimand=self.target_estimand.upper(),
            nonabsorbing=self.nonabsorbing,
            covariates=list(covariates or []),
        )
