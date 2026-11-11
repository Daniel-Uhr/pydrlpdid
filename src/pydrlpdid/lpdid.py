"""Benchmark LP-DiD estimator following Dube, Girardi, Jordà & Taylor (2025).

This private computational engine covers all three target estimands and both
absorbing and non-absorbing treatment settings described in the paper. The
public estimator is :class:`pydrlpdid.LPDID`.

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
    analytic_cluster_inference,
    BasePeriod,
    build_local_sample,
    check_columns,
    coerce_base_period,
    compute_base_series,
    infer_windows,
    lag,
    lead,
    prepare_panel,
    precompute_ccs,
)
from ._results import LPDIDResults
from ._errors import ExperimentalFeatureError, InferenceError
from ._aggregation import horizon_aggregate


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _nan_row(h: int) -> dict:
    return {
        "horizon": int(h), "estimate": np.nan, "se": np.nan,
        "t_stat": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": 0, "n_event_rows": 0, "n_clusters": 0,
    }


def _make_row(
    h: int, est: float, se: float, alpha: float, n_clusters: int,
    n_obs: int, n_event_rows: int,
) -> dict:
    return {
        "horizon": int(h), "estimate": est, "se": se, "n_obs": int(n_obs),
        "n_event_rows": int(n_event_rows),
        **analytic_cluster_inference(est, se, alpha, n_clusters),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cluster_fit(model, groups: pd.Series):
    """OLS/WLS with cluster-robust SE; inference failures are explicit."""
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as exc:
        raise InferenceError(
            f"Cluster-robust inference failed: {type(exc).__name__}: {exc}"
        ) from exc


def _ctrl_cols(
    covariates: Optional[List[str]],
    ldy: bool,
    n: int,
    n_levels: int = 0,
) -> List[str]:
    cols: List[str] = []
    if n_levels > 0:
        cols.extend([f"ly{k}" for k in range(1, int(n_levels) + 1)])
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


def _enforce_absorbed_level_overlap(
    local: pd.DataFrame, absorb_columns: Optional[List[str]]
) -> tuple[pd.DataFrame, int]:
    """Drop RA treated rows whose absorbed level has no clean control."""
    if not absorb_columns or local.empty:
        return local, 0
    out = local
    n_before = int((out["D_local"] == 1).sum())
    for column in absorb_columns:
        control_levels = set(
            out.loc[out["D_local"].eq(0), column].dropna().tolist()
        )
        keep = out["D_local"].eq(0) | (
            out["D_local"].eq(1) & out[column].isin(control_levels)
        )
        out = out.loc[keep].copy()
    return out, n_before - int((out["D_local"] == 1).sum())


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
    else:
        s["_w"] = 1.0

    y_df, x_df = patsy.dmatrices(f, s, return_type="dataframe")
    y = np.asarray(y_df, dtype=float).reshape(-1)
    x_full = np.asarray(x_df, dtype=float)
    keep = _independent_columns(x_full)
    retained = [str(x_df.columns[j]) for j in keep]
    if "D_local" not in retained:
        raise ValueError(
            "The treatment column is not identified after deterministic "
            "rank reduction of the LP-DiD design."
        )
    x = x_full[:, keep]
    model = sm.WLS(y, x, weights=s["_w"].to_numpy(dtype=float))
    fit = _cluster_fit(model, s[unit])
    idx = retained.index("D_local")
    est = float(fit.params[idx])
    se = float(fit.bse[idx])
    return {
        "estimate": est,
        "se": se,
        "retained_columns": retained,
        "dropped_columns": [
            str(x_df.columns[j]) for j in range(x_full.shape[1]) if j not in set(keep)
        ],
    }


def _influence_linear(
    local: pd.DataFrame, unit: str, time: str,
    controls: List[str], estimand: str,
    pre_rw: Optional[pd.Series] = None,
    small_sample_correction: str = "cr1",
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
    X_full = np.asarray(X_df, dtype=float)
    keep = _independent_columns(X_full)
    retained = [str(X_df.columns[j]) for j in keep]
    if "D_local" not in retained:
        raise ValueError(
            "The treatment column is not identified after deterministic "
            "rank reduction of the LP-DiD design."
        )
    X = X_full[:, keep]
    w = s["_w"].to_numpy(dtype=float)
    theta = np.asarray(sm.WLS(y, X, weights=w).fit().params, dtype=float)
    idx = retained.index("D_local")

    infl = stacked_influence(
        theta,
        lambda th: w[:, None] * X * (y - X @ th)[:, None],
        s[unit].to_numpy(),
        target_grad=lambda th: np.eye(len(th))[idx],
    )
    psi = infl["psi"].reshape(-1)
    _n = int(X.shape[0]); _k = int(X.shape[1]); _G = int(pd.unique(s[unit]).size)
    _cr1 = 1.0
    if small_sample_correction == "cr1" and _G > 1 and _n > _k:
        _cr1 = np.sqrt(
            (_G / (_G - 1.0)) * ((_n - 1.0) / (_n - _k))
        )
    psi = _cr1 * psi
    return {
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(psi, index=pd.Index(infl["cluster_labels"])),
        "small_sample_factor": float(_cr1),
        "retained_columns": retained,
        "dropped_columns": [
            str(X_df.columns[j]) for j in range(X_full.shape[1]) if j not in set(keep)
        ],
    }


# ---------------------------------------------------------------------------
# RA local estimation (regression-adjustment imputation)
# ---------------------------------------------------------------------------

def _independent_columns(x: np.ndarray) -> np.ndarray:
    """Return a deterministic, original-order full-rank column subset.

    Modified Gram--Schmidt with one re-orthogonalization pass avoids the
    repeated large SVDs of the earlier implementation.  This matters for
    county panels with tens of thousands of rows and many calendar effects.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.asarray([], dtype=int)
    n, p = x.shape
    column_scale = np.linalg.norm(x, axis=0)
    reference = float(np.max(column_scale)) if column_scale.size else 0.0
    tolerance = (
        np.finfo(float).eps * max(n, p) * max(reference, 1.0) * np.sqrt(p)
    )
    q = np.empty((n, min(n, p)), dtype=float)
    rank = 0
    keep: List[int] = []
    for j in range(p):
        residual = x[:, j].copy()
        if rank:
            basis = q[:, :rank]
            residual -= basis @ (basis.T @ residual)
            residual -= basis @ (basis.T @ residual)
        norm = float(np.linalg.norm(residual))
        if np.isfinite(norm) and norm > tolerance:
            q[:, rank] = residual / norm
            rank += 1
            keep.append(j)
    return np.asarray(keep, dtype=int)


def _ra_components(
    local: pd.DataFrame, unit: str, time: str, controls: List[str]
) -> dict:
    """Build the single certified sample/design used by RA point and inference."""
    # RA imputation requires a control in every time cell used: m0 carries the
    # time fixed effect, which is unidentified for a treated-only cell. Restrict
    # to time cells with at least one control (treated units in control-less
    # cells cannot be imputed and are excluded from the RA average). VW/RW keep
    # such cells to match reghdfe; RA cannot.
    _hc = local.loc[local["D_local"] == 0].groupby(time).size()
    local = local.loc[local[time].isin(_hc.loc[_hc > 0].index)].copy()
    if local.empty:
        raise ValueError("No usable time cell for RA estimation.")
    y_df, z_df = patsy.dmatrices(
        _formula("outcome_local", None, time, controls),
        local,
        return_type="dataframe",
        NA_action="raise",
    )
    y = np.asarray(y_df, dtype=float).reshape(-1)
    d = local["D_local"].to_numpy(dtype=float)
    control = d == 0
    treated = d == 1
    if not np.any(control):
        raise ValueError("No clean controls for RA estimation.")
    if not np.any(treated):
        raise ValueError("No treated observations for RA estimation.")
    z_full = np.asarray(z_df, dtype=float)
    keep = _independent_columns(z_full[control])
    if keep.size == 0:
        raise ValueError("The RA control design has zero rank.")
    z = z_full[:, keep]
    beta = np.linalg.lstsq(z[control], y[control], rcond=None)[0]
    residual = y - z @ beta
    estimate = float(np.mean(residual[treated]))
    return {
        "local": local,
        "y": y,
        "d": d,
        "z": z,
        "beta": beta,
        "estimate": estimate,
        "retained_columns": [str(z_df.columns[j]) for j in keep],
        "dropped_columns": [
            str(z_df.columns[j]) for j in range(z_full.shape[1]) if j not in set(keep)
        ],
    }


def _fit_ra(
    local: pd.DataFrame, unit: str, time: str, controls: List[str]
) -> dict:
    """RA-LP-DiD imputation at one horizon (Dube et al. 2025, eq. 11)."""
    comp = _ra_components(local, unit, time, controls)
    return {
        "estimate": comp["estimate"],
        "retained_columns": comp["retained_columns"],
        "dropped_columns": comp["dropped_columns"],
    }


def _influence_ra(
    local: pd.DataFrame, unit: str, time: str, controls: List[str],
    small_sample_correction: str = "cr1",
) -> dict:
    """Stacked GMM influence functions for the RA estimator.

    Moment system:

    .. math::

        m_1(\\boldsymbol{\\beta}) &= (1-D_i)(Y_i-Z_i\\boldsymbol{\\beta})Z_i \\\\
        m_2(\\mu_1,\\boldsymbol{\\beta}) &= D_i(Y_i-Z_i\\boldsymbol{\\beta}-\\mu_1)
    """
    comp = _ra_components(local, unit, time, controls)
    local = comp["local"]
    y, d, Z, beta = comp["y"], comp["d"], comp["z"], comp["beta"]
    mu1 = comp["estimate"]
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
    n = int(len(local)); k = int(len(theta)); g = int(local[unit].nunique())
    cr1 = 1.0
    if small_sample_correction == "cr1" and g > 1 and n > k:
        cr1 = np.sqrt((g / (g - 1.0)) * ((n - 1.0) / (n - k)))
    psi = cr1 * psi
    return {
        "estimate": mu1,
        "se": se_from_influence(psi),
        "psi_by_cluster": pd.Series(psi, index=pd.Index(infl["cluster_labels"])),
        "small_sample_factor": float(cr1),
        "retained_columns": comp["retained_columns"],
        "dropped_columns": comp["dropped_columns"],
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
    small_sample_correction: str = "cr1",
) -> tuple[float, float, pd.Series, tuple[str, ...]]:
    """Estimate ATT-h, SE, cluster influence, and rank diagnostics."""
    zero = pd.Series(0.0, index=cluster_universe)
    if estimand == "ra":
        if compute_influence:
            infl = _influence_ra(
                local, unit, time, controls, small_sample_correction
            )
            est = infl["estimate"]
            se = infl["se"]
            psi = infl["psi_by_cluster"].reindex(cluster_universe, fill_value=0.0)
            dropped = tuple(infl["dropped_columns"])
        else:
            fit = _fit_ra(local, unit, time, controls)
            est = fit["estimate"]
            se = np.nan
            psi = zero
            dropped = tuple(fit["dropped_columns"])
    else:
        r = _fit_linear(local, unit, time, controls, estimand, pre_rw)
        est, se = r["estimate"], r["se"]
        dropped = tuple(r["dropped_columns"])
        psi = zero
        if compute_influence:
            infl = _influence_linear(
                local, unit, time, controls, estimand, pre_rw,
                small_sample_correction,
            )
            se = infl["se"]
            psi = infl["psi_by_cluster"].reindex(cluster_universe, fill_value=0.0)
    return est, se, psi, dropped


# ---------------------------------------------------------------------------
# Scalar summaries
# ---------------------------------------------------------------------------

def _avg_scalar(
    event_df: pd.DataFrame, psi_by_h: Dict[int, pd.Series],
    clusters_by_h: Dict[int, pd.Index], cluster_universe: pd.Index,
    alpha: float, compute_influence: bool,
) -> tuple[list[dict], Dict[str, pd.Series]]:
    post = event_df.loc[
        (event_df["horizon"] >= 0) & np.isfinite(event_df["estimate"])
    ]
    if post.empty:
        return [], {}
    post_h = post["horizon"].astype(int).tolist()
    avg = float(np.mean(post["estimate"].to_numpy(dtype=float)))
    se = np.nan
    scalar_psi: Dict[str, pd.Series] = {}
    if compute_influence:
        Psi = np.column_stack([
            psi_by_h[h].reindex(cluster_universe, fill_value=0.0).to_numpy(dtype=float)
            for h in post_h
        ])
        avg_psi = np.mean(Psi, axis=1)
        se = se_from_influence(avg_psi)
        scalar_psi["ATT avg"] = pd.Series(
            avg_psi, index=cluster_universe, dtype=float
        )
    scalar_clusters = set()
    for h in post_h:
        scalar_clusters.update(clusters_by_h.get(h, pd.Index([])).tolist())
    return [{
        "term": "ATT avg", "estimate": avg, "se": se,
        **analytic_cluster_inference(avg, se, alpha, len(scalar_clusters)),
    }], scalar_psi


def _build_pooled_local(
    df: pd.DataFrame, outcome: str, unit: str, time: str,
    horizons: List[int], kind: str, base_period: BasePeriod,
    clean_control: str, effect_stabilization: Optional[int],
    extra_cols: List[str],
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
    control_pool: str = "stabilized_all",
    switch_in: str = "sustained",
    control_window: str = "horizon",
) -> pd.DataFrame:
    """Build a pooled pre- or post-window stack on one common sample."""
    horizons = [int(h) for h in horizons]
    if not horizons or kind not in {"pre", "post"}:
        return df.iloc[0:0].copy()

    # Reuse the certified horizon-specific membership rules.  Absorbing
    # placebo pooling uses entry-date membership (h=0); non-absorbing pooling
    # must instead use the most negative placebo so its clean-history window
    # covers the entire reach-back interval.
    membership_h = (
        max(horizons)
        if kind == "post"
        else (min(horizons) if clean_control in {"first_entry", "stabilized"} else 0)
    )
    # Membership must not require the outcome at ``membership_h`` unless that
    # horizon is itself in the requested pooled window.  In particular, a
    # pooled placebo such as h=-2 uses Y_{t-2} and Y_{t-1}, not Y_t.  A dummy
    # complete outcome lets the certified membership routine apply only the
    # treatment/history rules; the actual pooled outcome support is imposed
    # immediately below.
    membership_df = df.copy()
    membership_outcome = "_pooled_membership_outcome"
    membership_df[membership_outcome] = 0.0
    local = build_local_sample(
        membership_df, membership_outcome, unit, time, membership_h, base_period,
        clean_control, effect_stabilization, extra_cols,
        user_covariates=user_covariates,
        lag_covariates=lag_covariates,
        control_pool=control_pool,
        switch_in=switch_in,
        control_window=control_window,
    )
    if local.empty:
        return local

    targets = np.column_stack([
        lead(df, unit, outcome, h).to_numpy(dtype=float) for h in horizons
    ])
    complete = np.all(np.isfinite(targets), axis=1)
    pooled_diff = np.full(len(df), np.nan, dtype=float)
    pooled_diff[complete] = (
        targets[complete].mean(axis=1)
        - compute_base_series(df, outcome, unit, base_period)
          .to_numpy(dtype=float)[complete]
    )
    pooled_by_key = pd.Series(
        pooled_diff,
        index=pd.MultiIndex.from_frame(df[[unit, time]]),
    )
    local_keys = pd.MultiIndex.from_frame(local[[unit, time]])
    local["outcome_local"] = pooled_by_key.reindex(local_keys).to_numpy(dtype=float)
    local = local.loc[np.isfinite(local["outcome_local"])].copy()
    return local.reset_index(drop=True)


def _pooled_scalar(
    df: pd.DataFrame, outcome: str, unit: str, time: str,
    event_df: pd.DataFrame, horizons: List[int], kind: str,
    base_period: BasePeriod,
    clean_control: str, effect_stabilization: Optional[int],
    controls: List[str], estimand: str, alpha: float, compute_influence: bool,
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
    control_pool: str = "stabilized_all",
    switch_in: str = "sustained",
    control_window: str = "horizon",
    cluster_universe: Optional[pd.Index] = None,
    inference_cluster: Optional[str] = None,
    small_sample_correction: str = "cr1",
    absorb_columns: Optional[List[str]] = None,
) -> tuple[list[dict], Dict[str, pd.Series], tuple[str, ...]]:
    horizons = [int(h) for h in horizons]
    supported = set(
        event_df.loc[np.isfinite(event_df["estimate"]), "horizon"].astype(int)
    )
    if not horizons or not set(horizons).issubset(supported):
        return [], {}, ()
    pl = _build_pooled_local(
        df, outcome, unit, time, horizons, kind, base_period,
        clean_control, effect_stabilization, controls,
        user_covariates=user_covariates,
        lag_covariates=lag_covariates,
        control_pool=control_pool,
        switch_in=switch_in,
        control_window=control_window,
    )
    if pl.empty or pl["D_local"].sum() == 0:
        return [], {}, ()

    if estimand == "ra":
        pl, _ = _enforce_absorbed_level_overlap(pl, absorb_columns)
        if pl.empty or pl["D_local"].sum() == 0:
            return [], {}, ()

    cluster_col = inference_cluster or unit
    term = "ATT pooled" if kind == "post" else "Placebo pooled"

    scalar_psi: Dict[str, pd.Series] = {}
    if estimand == "ra":
        fit = _fit_ra(pl, cluster_col, time, controls)
        est = fit["estimate"]
        dropped = tuple(fit["dropped_columns"])
        se = np.nan
        if compute_influence:
            infl = _influence_ra(
                pl, cluster_col, time, controls, small_sample_correction
            )
            est, se = infl["estimate"], infl["se"]
            scalar_psi[term] = infl["psi_by_cluster"]
    else:
        r = _fit_linear(pl, cluster_col, time, controls, estimand)
        est, se = r["estimate"], r["se"]
        dropped = tuple(r["dropped_columns"])
        if compute_influence:
            infl = _influence_linear(
                pl, cluster_col, time, controls, estimand,
                small_sample_correction=small_sample_correction,
            )
            se = infl["se"]
            scalar_psi[term] = infl["psi_by_cluster"]

    if compute_influence and cluster_universe is not None:
        scalar_psi[term] = scalar_psi[term].reindex(
            cluster_universe, fill_value=0.0
        )

    return [{
        "term": term, "window": kind, "aggregation": "pooled",
        "estimate": est, "se": se,
        "n_obs": int(len(pl)),
        "n_event_rows": int(pl["D_local"].eq(1).sum()),
        "n_horizons": int(len(horizons)),
        "horizons": tuple(horizons),
        "weights": None,
        "weight_basis": "joint_pooled_outcome_regression",
        **analytic_cluster_inference(
            est, se, alpha, int(pl[cluster_col].dropna().nunique())
        ),
    }], scalar_psi, dropped


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def _cluster_bootstrap(
    estimator: "_LPDIDEngine",
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
        _refit,
        data.copy(),
        estimator.cluster or unit,
        estimator.n_bootstrap,
        estimator.alpha,
        rng,
        orig,
        panel_unit=unit,
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
# Internal LP-DiD computational engine
# ---------------------------------------------------------------------------

class _LPDIDEngine:
    """Internal LP-DiD computational engine.

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

    >>> from pydrlpdid.lpdid import _LPDIDEngine
    >>> res_vw = _LPDIDEngine(target_estimand='vw').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')
    >>> res_rw = _LPDIDEngine(target_estimand='rw').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')
    >>> res_ra = _LPDIDEngine(target_estimand='ra', base_period='all_pre').fit(
    ...     df, outcome='y', unit='id', time='t', first_treat='g')

    Non-absorbing treatment — first-entry condition (Section 4.2.2):

    >>> res_fe = _LPDIDEngine(
    ...     target_estimand='vw',
    ...     clean_control='first_entry',
    ...     nonabsorbing=True,
    ... ).fit(df, outcome='y', unit='id', time='t', treatment='D')

    Non-absorbing with effect stabilization (Section 4.2.3):

    >>> res_st = _LPDIDEngine(
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
        n_lagged_outcome_levels: int = 0,
        inference: str = "cluster",
        n_bootstrap: int = 499,
        bootstrap_weights: str = "mammen",
        alpha: float = 0.05,
        seed: Optional[int] = None,
        max_pre: Optional[int] = None,
        max_post: Optional[int] = None,
        lag_covariates: bool = False,
        fixed_composition: bool = False,
        no_composition: bool = False,
        support_policy: str = "supported_subset",
        left_censoring: str = "error",
        time_policy: str = "calendar",
        history_policy: str = "observed",
        pre_weighting: str = "horizon_specific",
        cluster: Optional[str] = None,
        small_sample_correction: str = "cr1",
        rank_deficient_action: str = "warn",
        pooled_pre_horizons: Optional[List[int]] = None,
        pooled_post_horizons: Optional[List[int]] = None,
        pre_aggregations: Optional[List[str]] = None,
        post_aggregations: Optional[List[str]] = None,
        omit_overlapping_outcome_controls: bool = True,
        absorb_columns: Optional[List[str]] = None,
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
        self.n_lagged_outcome_levels = int(max(n_lagged_outcome_levels, 0))
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
        self.no_composition = bool(no_composition)
        if self.fixed_composition and self.no_composition:
            raise ValueError(
                "Use fixed_composition or no_composition, not both."
            )
        # Certified equation (13) design for stabilized non-absorbing treatment.
        self.control_pool = "stabilized_all"
        self.switch_in = "sustained"
        self.control_window = "horizon"
        self.support_policy = str(support_policy).lower()
        if self.support_policy not in {"strict", "supported_subset"}:
            raise ValueError("support_policy must be 'strict' or 'supported_subset'.")
        self.left_censoring = str(left_censoring).lower()
        self.time_policy = str(time_policy).lower()
        self.history_policy = str(history_policy).lower()
        if self.history_policy not in {"observed", "diff_diff"}:
            raise ValueError("history_policy must be 'observed' or 'diff_diff'.")
        self.pre_weighting = str(pre_weighting).lower()
        if self.pre_weighting not in {"post_target", "horizon_specific"}:
            raise ValueError(
                "pre_weighting must be 'post_target' or 'horizon_specific'."
            )
        self.cluster = cluster
        self.small_sample_correction = str(small_sample_correction).lower()
        if self.small_sample_correction not in {"none", "cr1"}:
            raise ValueError(
                "small_sample_correction must be 'cr1' or 'none'."
            )
        self.rank_deficient_action = str(rank_deficient_action).lower()
        if self.rank_deficient_action not in {"warn", "error", "silent"}:
            raise ValueError(
                "rank_deficient_action must be 'warn', 'error', or 'silent'."
            )
        self.pooled_pre_horizons = (
            None if pooled_pre_horizons is None
            else [int(h) for h in pooled_pre_horizons]
        )
        self.pooled_post_horizons = (
            None if pooled_post_horizons is None
            else [int(h) for h in pooled_post_horizons]
        )
        self.pre_aggregations = (
            None if pre_aggregations is None else list(pre_aggregations)
        )
        self.post_aggregations = (
            None if post_aggregations is None else list(post_aggregations)
        )
        self.omit_overlapping_outcome_controls = bool(
            omit_overlapping_outcome_controls
        )
        self.absorb_columns = list(absorb_columns or [])

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
            n_lagged_outcome_levels=self.n_lagged_outcome_levels,
            left_censoring=self.left_censoring,
            time_policy=self.time_policy,
            history_policy=self.history_policy,
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
        inference_cluster = self.cluster or unit
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
            self.n_lagged_outcome_levels,
        )
        cu = pd.Index(df[inference_cluster].dropna().unique())
        psi_by_h: Dict[int, pd.Series] = {}
        clusters_by_h: Dict[int, pd.Index] = {}
        rows: list[dict] = []
        omitted_overlapping_outcome_controls: dict[int, tuple[str, ...]] = {}
        control_unit_ids: set = set()
        rank_dropped_columns: dict[str, tuple[str, ...]] = {}

        # `fixed_composition` is the legacy all-horizon option.
        # `no_composition` reproduces diff-diff: hold the sample fixed only
        # across post-treatment horizons and leave placebo leads unrestricted.
        legacy_fc_H: Optional[int] = (
            int(max_post) if self.fixed_composition else None
        )

        complete_post_keys = None
        if self.no_composition:
            base_ok = np.isfinite(
                compute_base_series(df, outcome, unit, self.base_period)
            )
            post_ok = pd.Series(True, index=df.index)
            for j in range(0, int(max_post) + 1):
                post_ok &= np.isfinite(lead(df, unit, outcome, j))
            complete = df.loc[base_ok & post_ok, [unit, time]]
            complete_post_keys = pd.MultiIndex.from_frame(complete)

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
                fixed_composition_H=(
                    int(max_post) if self.no_composition else legacy_fc_H
                ),
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
            fc_H = (
                int(max_post)
                if self.no_composition and h >= 0
                else legacy_fc_H
            )
            overlapping = (
                {f"ldy{k}" for k in range(1, -int(h))}
                if self.omit_overlapping_outcome_controls and h < -1
                else set()
            )
            controls_h = [c for c in controls if c not in overlapping]
            omitted = tuple(sorted(overlapping.intersection(controls)))
            if omitted:
                omitted_overlapping_outcome_controls[int(h)] = omitted
            local = build_local_sample(
                df, outcome, unit, time, h, self.base_period,
                self.clean_control, self.effect_stabilization, controls_h,
                user_covariates=covariates,
                lag_covariates=self.lag_covariates,
                fixed_composition_H=fc_H,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
            )
            if complete_post_keys is not None and h >= 0 and not local.empty:
                local_keys = pd.MultiIndex.from_frame(local[[unit, time]])
                local = local.loc[local_keys.isin(complete_post_keys)].copy()
            if self.target_estimand == "ra":
                local, _ = _enforce_absorbed_level_overlap(
                    local, self.absorb_columns
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
                clusters_by_h[int(h)] = pd.Index([])
                continue

            cell = local.groupby(time)["D_local"].agg(["sum", "count"])
            unsupported_times = cell.index[(cell["sum"] > 0) & (cell["sum"] == cell["count"])].tolist()
            if unsupported_times and self.support_policy == "strict":
                rows.append(_nan_row(h))
                psi_by_h[int(h)] = pd.Series(0.0, index=cu)
                clusters_by_h[int(h)] = pd.Index([])
                continue
            if unsupported_times:
                local = local.loc[~local[time].isin(unsupported_times)].copy()
            control_unit_ids.update(
                local.loc[local["D_local"].eq(0), unit].tolist()
            )

            # Pre-period RW: map h=0 time-cell weights to this stack
            pre_rw: Optional[pd.Series] = None
            if (
                self.pre_weighting == "post_target"
                and self.target_estimand == "rw"
                and h < 0
                and rw0_map is not None
            ):
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

            est, se, psi, dropped = _fit_horizon(
                local, inference_cluster, time, controls_h, self.target_estimand,
                compute_influence, cu, pre_rw=pre_rw,
                small_sample_correction=self.small_sample_correction,
            )
            if dropped:
                rank_dropped_columns[f"h={int(h)}"] = dropped
            psi_by_h[int(h)] = psi
            clusters_by_h[int(h)] = pd.Index(
                local[inference_cluster].dropna().unique()
            )
            rows.append(_make_row(
                h, est, se, self.alpha, len(clusters_by_h[int(h)]), len(local),
                int(local["D_local"].eq(1).sum()),
            ))

        # Normalized base-period row (estimate = 0 by construction)
        if isinstance(self.base_period, int):
            rows.append({
                "horizon": int(self.base_period), "estimate": 0.0, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "df": np.nan, "n_obs": np.nan,
                "n_event_rows": 0,
                "n_clusters": 0, "critical_value": np.nan,
            })
            psi_by_h[int(self.base_period)] = pd.Series(0.0, index=cu)
            clusters_by_h[int(self.base_period)] = pd.Index([])

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

        public_aggregations = (
            self.pre_aggregations is not None or self.post_aggregations is not None
        )
        if public_aggregations:
            scalar_rows: list[dict] = []
            psi_by_scalar: Dict[str, pd.Series] = {}
        else:
            # Private-engine backward compatibility: retain the historical
            # horizon mean in addition to the pooled post regression.
            scalar_rows, psi_by_scalar = _avg_scalar(
                event_df, psi_by_h, clusters_by_h, cu, self.alpha,
                compute_influence,
            )
        post_pool_h = (
            post_horizons
            if self.pooled_post_horizons is None
            else self.pooled_post_horizons
        )
        post_methods = (
            list(self.post_aggregations)
            if self.post_aggregations is not None else ["pooled"]
        )
        if "pooled" in post_methods:
            pooled_rows, pooled_psi, pooled_dropped = _pooled_scalar(
                df, outcome, unit, time, event_df, post_pool_h, "post",
                self.base_period,
                self.clean_control, self.effect_stabilization, controls,
                self.target_estimand, self.alpha, compute_influence,
                user_covariates=covariates,
                lag_covariates=self.lag_covariates,
                control_pool=self.control_pool,
                switch_in=self.switch_in,
                control_window=self.control_window,
                cluster_universe=cu,
                inference_cluster=inference_cluster,
                small_sample_correction=self.small_sample_correction,
                absorb_columns=self.absorb_columns,
            )
            if pooled_dropped:
                rank_dropped_columns["ATT pooled"] = pooled_dropped
            scalar_rows += pooled_rows
            psi_by_scalar.update(pooled_psi)
        for method in post_methods:
            if method == "pooled":
                continue
            row, psi, _ = horizon_aggregate(
                event_df, psi_by_h, clusters_by_h, cu, post_pool_h,
                window="post", method=method, alpha=self.alpha,
                compute_influence=compute_influence,
            )
            if row is not None:
                scalar_rows.append(row)
                if psi is not None:
                    psi_by_scalar[str(row["term"])] = psi

        if self.pooled_pre_horizons is not None:
            pre_methods = (
                list(self.pre_aggregations)
                if self.pre_aggregations is not None else ["pooled"]
            )
            if "pooled" in pre_methods:
                pre_rows, pre_psi, pre_dropped = _pooled_scalar(
                    df, outcome, unit, time, event_df,
                    self.pooled_pre_horizons, "pre", self.base_period,
                    self.clean_control, self.effect_stabilization, controls,
                    self.target_estimand, self.alpha, compute_influence,
                    user_covariates=covariates,
                    lag_covariates=self.lag_covariates,
                    control_pool=self.control_pool,
                    switch_in=self.switch_in,
                    control_window=self.control_window,
                    cluster_universe=cu,
                    inference_cluster=inference_cluster,
                    small_sample_correction=self.small_sample_correction,
                    absorb_columns=self.absorb_columns,
                )
                if pre_dropped:
                    rank_dropped_columns["Placebo pooled"] = pre_dropped
                scalar_rows += pre_rows
                psi_by_scalar.update(pre_psi)
            for method in pre_methods:
                if method == "pooled":
                    continue
                row, psi, _ = horizon_aggregate(
                    event_df, psi_by_h, clusters_by_h, cu,
                    self.pooled_pre_horizons,
                    window="pre", method=method, alpha=self.alpha,
                    compute_influence=compute_influence,
                )
                if row is not None:
                    scalar_rows.append(row)
                    if psi is not None:
                        psi_by_scalar[str(row["term"])] = psi

        return {
            "panel": df, "event_study": event_df,
            "scalars": pd.DataFrame(scalar_rows),
            "psi_by_h": psi_by_h,
            "psi_by_scalar": psi_by_scalar,
            "cluster_universe": cu,
            "n_obs": int(df.shape[0]),
            "n_treated_units": int(df.loc[df["_treated_ever"] == 1, unit].nunique()),
            "n_control_units": int(len(control_unit_ids)),
            "n_cohorts": int(df.loc[df["_first_treat"] > 0, "_first_treat"].nunique()),
            "n_periods": int(df[time].nunique()),
            "fixed_composition": bool(self.fixed_composition),
            "no_composition": bool(self.no_composition),
            "support_policy": self.support_policy,
            "left_censoring": self.left_censoring,
            "time_policy": self.time_policy,
            "history_policy": self.history_policy,
            "pre_weighting": self.pre_weighting,
            "small_sample_correction": self.small_sample_correction,
            "rank_dropped_columns": rank_dropped_columns,
            "reference_distribution": "student_t_G_minus_1",
            "cluster_level": inference_cluster,
            "omitted_overlapping_outcome_controls": (
                omitted_overlapping_outcome_controls
            ),
            "omit_overlapping_outcome_controls": bool(
                self.omit_overlapping_outcome_controls
            ),
            "absorb_columns": tuple(self.absorb_columns),
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
        if self.cluster is not None:
            check_columns(data, [self.cluster])
            if data[self.cluster].isna().any():
                raise ValueError("The cluster column contains missing values.")
            unit_to_cluster = data.groupby(unit, sort=False)[self.cluster].nunique(
                dropna=False
            )
            if (unit_to_cluster != 1).any():
                raise ValueError(
                    "Each panel unit must belong to exactly one inference cluster."
                )
        core = self._fit_core(
            data, outcome, unit, time, first_treat, treatment, covariates,
            compute_influence=(self.inference in {"cluster", "multiplier"}),
        )
        rank_dropped = core.get("rank_dropped_columns", {})
        if rank_dropped:
            detail = "; ".join(
                f"{term}: {list(columns)}"
                for term, columns in rank_dropped.items()
            )
            message = (
                "Rank-deficient LP-DiD design detected. Deterministic rank "
                f"reduction dropped the following columns: {detail}."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(message)
            if self.rank_deficient_action == "warn":
                warnings.warn(message, stacklevel=2)
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
            if not scalars.empty:
                scalars["multiplier_ci_lower"] = np.nan
                scalars["multiplier_ci_upper"] = np.nan
                scalars["multiplier_p_value"] = np.nan
                scalar_terms: list[str] = []
                for idx, row in scalars.iterrows():
                    term = str(row["term"])
                    psi_scalar = core["psi_by_scalar"].get(term)
                    se_scalar = float(row["se"])
                    if (
                        psi_scalar is None
                        or not np.isfinite(se_scalar)
                        or se_scalar <= 0
                    ):
                        continue
                    mb_scalar = run_multiplier_bootstrap(
                        np.asarray([float(row["estimate"])]),
                        psi_scalar.reindex(
                            core["cluster_universe"], fill_value=0.0
                        ).to_numpy(dtype=float)[:, None],
                        np.asarray([se_scalar]),
                        self.n_bootstrap,
                        self.bootstrap_weights,
                        self.alpha,
                        rng,
                    )
                    scalars.loc[idx, "multiplier_ci_lower"] = mb_scalar["ci_lower"][0]
                    scalars.loc[idx, "multiplier_ci_upper"] = mb_scalar["ci_upper"][0]
                    scalars.loc[idx, "multiplier_p_value"] = mb_scalar["p_values"][0]
                    scalar_terms.append(term)
                multiplier_metadata["scalar_terms"] = tuple(scalar_terms)

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
            "package_version": "0.8.1",
            "estimator_identity": (
                "dube_2025_lpdid_ra"
                if self.target_estimand == "ra"
                else f"dube_2025_lpdid_{self.target_estimand}"
            ),
            "estimation_sample": "dube_clean_control_sample",
            "fixed_composition": core.get("fixed_composition", False),
            "no_composition": core.get("no_composition", False),
            "nonabsorbing_certified": bool(self.clean_control == "stabilized"),
            "treatment_design": core.get("treatment_design"),
            "nonabsorbing_switch_in": "sustained" if self.clean_control == "stabilized" else None,
            "nonabsorbing_control_window": "horizon" if self.clean_control == "stabilized" else None,
            "nonabsorbing_control_pool": "stabilized_all" if self.clean_control == "stabilized" else None,
            "estimand_weighting": "eligible_switch_in_events" if self.clean_control == "stabilized" else "treated_entries",
            "history_policy": core.get("history_policy"),
            "left_censoring": core.get("left_censoring"),
            "pre_weighting": core.get("pre_weighting"),
            "small_sample_correction": core.get("small_sample_correction"),
            "rank_deficient_action": self.rank_deficient_action,
            "rank_dropped_columns": rank_dropped,
            "reference_distribution": core.get("reference_distribution"),
            "omitted_overlapping_outcome_controls": core.get(
                "omitted_overlapping_outcome_controls", {}
            ),
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
