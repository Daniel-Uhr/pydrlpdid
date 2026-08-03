from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ._panel_utils import p_value_two_sided
from ._errors import JacobianError


def multiplier_weights(
    n_bootstrap: int,
    n_clusters: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a (B,G) matrix of multiplier weights."""
    weight_type = str(weight_type).lower()
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n_bootstrap, n_clusters))
    if weight_type == "mammen":
        sqrt5 = np.sqrt(5.0)
        val1 = -(sqrt5 - 1.0) / 2.0
        val2 = (sqrt5 + 1.0) / 2.0
        p1 = (sqrt5 + 1.0) / (2.0 * sqrt5)
        return rng.choice([val1, val2], size=(n_bootstrap, n_clusters), p=[p1, 1.0 - p1])
    if weight_type == "webb":
        values = np.array(
            [-np.sqrt(3.0 / 2.0), -1.0, -np.sqrt(1.0 / 2.0), np.sqrt(1.0 / 2.0), 1.0, np.sqrt(3.0 / 2.0)]
        )
        return rng.choice(values, size=(n_bootstrap, n_clusters))
    raise ValueError("weight_type must be one of {'rademacher','mammen','webb'}.")


def resample_clusters(df: pd.DataFrame, cluster_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Paired cluster resampling with replacement and unique bootstrap IDs."""
    clusters = pd.Index(df[cluster_col].drop_duplicates())
    if len(clusters) == 0:
        return df.copy()
    sampled = rng.choice(clusters.to_numpy(), size=len(clusters), replace=True)
    pieces = []
    for b, cid in enumerate(sampled):
        part = df.loc[df[cluster_col] == cid].copy()
        part[cluster_col] = f"boot_{b}_{cid}"
        pieces.append(part)
    return pd.concat(pieces, axis=0, ignore_index=True)


def percentile_ci(draws: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Percentile confidence interval from bootstrap draws."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (np.nan, np.nan)
    return (
        float(np.percentile(draws, 100.0 * alpha / 2.0)),
        float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
    )


def bootstrap_p_value(original_effect: float, centered_draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value using centered draws."""
    centered_draws = np.asarray(centered_draws, dtype=float)
    centered_draws = centered_draws[np.isfinite(centered_draws)]
    if centered_draws.size == 0 or not np.isfinite(original_effect):
        return np.nan
    return float(np.mean(np.abs(centered_draws) >= abs(original_effect)))


def numeric_jacobian(moment_func, theta_hat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of mean moments."""
    theta_hat = np.asarray(theta_hat, dtype=float)
    g0 = np.asarray(moment_func(theta_hat), dtype=float)
    q = g0.shape[0]
    p = theta_hat.shape[0]
    A = np.zeros((q, p), dtype=float)
    for j in range(p):
        step = eps * max(1.0, abs(theta_hat[j]))
        tp = theta_hat.copy()
        tp[j] += step
        tm = theta_hat.copy()
        tm[j] -= step
        gp = np.asarray(moment_func(tp), dtype=float)
        gm = np.asarray(moment_func(tm), dtype=float)
        A[:, j] = (gp - gm) / (2.0 * step)
    return A


def stacked_influence(
    theta_hat: np.ndarray,
    moments_obs_func,
    cluster_ids,
    target_grad=None,
    eps: float = 1e-6,
    *,
    condition_limit: float = 1e12,
) -> dict:
    """Compute cluster-level influence functions for a just-identified system.

    The influence sign follows ``-A^{-1} score``. Singular or severely
    ill-conditioned Jacobians are errors rather than silently pseudo-inverted.
    The returned contributions implement the article's uncorrected
    cluster-sandwich formula.  In particular, no degrees-of-freedom factor
    depending on the dimension of the chosen moment representation is applied.
    This makes inference invariant to algebraically redundant nuisance blocks.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    m_i = np.asarray(moments_obs_func(theta_hat), dtype=float)
    if m_i.ndim == 1:
        m_i = m_i[:, None]
    n, q = m_i.shape
    if n == 0:
        raise ValueError("The sample is empty for cluster-influence computation.")

    cluster_ids = pd.Series(cluster_ids).reset_index(drop=True)
    scores = pd.DataFrame(m_i)
    scores["_cluster"] = cluster_ids.to_numpy()
    grouped = scores.groupby("_cluster", sort=False).sum()
    cluster_labels = grouped.index.to_list()
    S = grouped.to_numpy(dtype=float)

    mean_moment = lambda th: np.asarray(moments_obs_func(th), dtype=float).mean(axis=0)
    A = numeric_jacobian(mean_moment, theta_hat, eps=eps)
    if A.shape[0] != A.shape[1]:
        raise JacobianError(
            f"Stacked system must be square for certified inference; got Jacobian {A.shape}."
        )
    singular_values = np.linalg.svd(A, compute_uv=False)
    tol = np.finfo(float).eps * max(A.shape) * (singular_values[0] if singular_values.size else 0.0)
    rank = int(np.sum(singular_values > tol))
    min_sv = float(singular_values[-1]) if singular_values.size else 0.0
    cond = float(np.inf if min_sv == 0 else singular_values[0] / min_sv)
    if rank < A.shape[0] or not np.isfinite(cond):
        raise JacobianError(
            f"Stacked Jacobian is singular: rank={rank}/{A.shape[0]}, "
            f"condition={cond:.3e}, min_singular_value={min_sv:.3e}."
        )
    Ainv = np.linalg.inv(A)
    if target_grad is None:
        psi = -(S @ Ainv.T) / n
        return {"cluster_labels": cluster_labels, "psi": psi}

    g = np.asarray(target_grad(theta_hat), dtype=float).reshape(-1, 1)
    v = (Ainv.T @ g).reshape(-1)
    psi_tau = -(S @ v) / n
    return {"cluster_labels": cluster_labels, "psi": psi_tau.reshape(-1, 1)}


def se_from_influence(psi: np.ndarray) -> float:
    """Cluster-robust standard error from cluster-level influence values."""
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if not np.all(np.isfinite(psi)):
        return np.nan
    return float(np.sqrt(np.sum(psi ** 2)))


def run_multiplier_bootstrap(
    tau_hat: np.ndarray,
    Psi: np.ndarray,
    se_h: np.ndarray,
    n_bootstrap: int,
    weight_type: str,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """Multiplier bootstrap for the full event-study path with simultaneous bands."""
    tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
    Psi = np.asarray(Psi, dtype=float)
    se_h = np.asarray(se_h, dtype=float).reshape(-1)
    G, H = Psi.shape
    if H != tau_hat.size:
        raise ValueError("Psi columns must match tau_hat length.")

    W = multiplier_weights(n_bootstrap, G, weight_type, rng)
    centered = W @ Psi
    draws = centered + tau_hat.reshape(1, -1)

    ci_lower = np.full(H, np.nan)
    ci_upper = np.full(H, np.nan)
    p_values = np.full(H, np.nan)
    for h in range(H):
        ci_lower[h], ci_upper[h] = percentile_ci(draws[:, h], alpha)
        p_values[h] = bootstrap_p_value(tau_hat[h], centered[:, h])

    studentized = centered / np.where(se_h > 0, se_h, np.nan)[None, :]
    with np.errstate(invalid="ignore"):
        sup_t = np.nanmax(np.abs(studentized), axis=1)
    cband_crit = float(np.nanquantile(sup_t, 1.0 - alpha)) if np.isfinite(sup_t).any() else np.nan
    sim_ci_lower = tau_hat - cband_crit * se_h
    sim_ci_upper = tau_hat + cband_crit * se_h

    return {
        "draws": draws,
        "se": se_h,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_values": p_values,
        "cband_crit": cband_crit,
        "sim_ci_lower": sim_ci_lower,
        "sim_ci_upper": sim_ci_upper,
        "weight_type": weight_type,
        "n_bootstrap": n_bootstrap,
    }


def run_cluster_bootstrap(
    fit_func,
    data: pd.DataFrame,
    unit: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
    original_estimates: dict,
) -> dict:
    """Paired cluster bootstrap for event-study paths and scalar summaries."""
    event_index = list(original_estimates["event_index"])
    scalar_terms = list(original_estimates["scalar_terms"])
    B_event = {k: [] for k in event_index}
    B_scalar = {k: [] for k in scalar_terms}

    for _ in range(n_bootstrap):
        boot_df = resample_clusters(data, unit, rng)
        boot = fit_func(boot_df)
        es = boot["event_study"].set_index("horizon")
        scalar_df = boot.get("scalars", pd.DataFrame())
        sc = (
            scalar_df.set_index("term")
            if "term" in scalar_df.columns
            else pd.DataFrame(index=pd.Index([], name="term"))
        )
        for h in event_index:
            B_event[h].append(float(es.loc[h, "estimate"]) if h in es.index else np.nan)
        for term in scalar_terms:
            B_scalar[term].append(float(sc.loc[term, "estimate"]) if term in sc.index else np.nan)

    event_rows = []
    for h in event_index:
        draws = np.asarray(B_event[h], dtype=float)
        se = float(np.nanstd(draws, ddof=1)) if np.isfinite(draws).sum() > 1 else np.nan
        lo, hi = percentile_ci(draws, alpha)
        p = bootstrap_p_value(original_estimates["event_hat"][h], draws - np.nanmean(draws))
        t = original_estimates["event_hat"][h] / se if np.isfinite(se) and se > 0 else np.nan
        event_rows.append({"horizon": h, "se": se, "ci_lower": lo, "ci_upper": hi, "p_value": p, "t_stat": t})

    scalar_rows = []
    for term in scalar_terms:
        draws = np.asarray(B_scalar[term], dtype=float)
        se = float(np.nanstd(draws, ddof=1)) if np.isfinite(draws).sum() > 1 else np.nan
        lo, hi = percentile_ci(draws, alpha)
        p = bootstrap_p_value(original_estimates["scalar_hat"][term], draws - np.nanmean(draws))
        t = original_estimates["scalar_hat"][term] / se if np.isfinite(se) and se > 0 else np.nan
        scalar_rows.append({"term": term, "se": se, "ci_lower": lo, "ci_upper": hi, "p_value": p, "t_stat": t})

    return {"event_boot": pd.DataFrame(event_rows, columns=["horizon", "se", "ci_lower", "ci_upper", "p_value", "t_stat"]), "scalar_boot": pd.DataFrame(scalar_rows, columns=["term", "se", "ci_lower", "ci_upper", "p_value", "t_stat"])}

