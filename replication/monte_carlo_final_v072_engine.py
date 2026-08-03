"""Article Monte Carlo engine for pydrlpdid 0.7.2.

This module is embedded in the accompanying notebook. It reproduces only the
Monte Carlo exercises retained in the article:

1. absorbing adoption, N=500, four nuisance specifications;
2. absorbing-adoption conditional-parallel-trends sensitivity;
3. sustained switch-ins, N in {184, 368}, four nuisance specifications;
4. the switching dynamic path under correct nuisances; and
5. switching conditional-parallel-trends sensitivity.

The separate nuisance bases are a replication-only hook. They are not added
to the public estimator API. No failed fit is silently discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit

from pydrlpdid import DRLPDID, LPDID, __version__

PACKAGE_SOURCE_FILE = Path(
    sys.modules[DRLPDID.__module__].__file__
).resolve()

Z95 = 1.959963984540054
METHOD_LABELS = {
    "ra": "DRLPDID-RA",
    "ipw": "IPW",
    "ipt": "IPT",
    "dr-ipw": "DRLPDID-IPW",
    "dr-ipt": "DRLPDID-IPT",
}
LOCAL_METHODS = tuple(METHOD_LABELS)
ABSORBING_METHODS = (
    "LPDID-RW",
    "LPDID-RW+X",
    *tuple(METHOD_LABELS.values()),
)

ABS_FULL = ("x1", "x2", "x3", "x4")
ABS_PARTIAL = ("x1", "x2")
ABS_SCENARIOS = {
    "A_both_correct": {"ps": ABS_FULL, "or": ABS_FULL},
    "B_or_misspecified": {"ps": ABS_FULL, "or": ABS_PARTIAL},
    "C_ps_misspecified": {"ps": ABS_PARTIAL, "or": ABS_FULL},
    "D_both_misspecified": {"ps": ABS_PARTIAL, "or": ABS_PARTIAL},
}

SW_FULL = ("x1", "x2_sq", "lag_dy_std")
SW_PARTIAL = ("x1", "lag_dy_std")
SW_SCENARIOS = {
    "A_both_correct": {"ps": SW_FULL, "or": SW_FULL},
    "B_or_misspecified": {"ps": SW_FULL, "or": SW_PARTIAL},
    "C_ps_misspecified": {"ps": SW_PARTIAL, "or": SW_FULL},
    "D_both_misspecified": {"ps": SW_PARTIAL, "or": SW_PARTIAL},
}


@dataclass(frozen=True)
class MCSettings:
    mode: str
    scope: str
    output_dir: str
    absorbing_replications: int
    switching_replications: int
    absorbing_seed_base: int = 20250301
    switching_seed_base: int = 20260722
    checkpoint_every: int = 10
    absorbing_N: int = 500
    absorbing_T: int = 17
    absorbing_post: int = 6
    switching_N: tuple[int, ...] = (184, 368)
    switching_T: int = 51
    switching_L: int = 9
    switching_pre: int = 10
    switching_post: int = 10
    scalar_window: tuple[int, int] = (0, 6)
    pt_deltas: tuple[float, ...] = (0.0, 0.5, 1.0)
    switching_pt_modes: tuple[str, ...] = ("pretrend", "post_only")


def make_settings(
    mode: str,
    output_dir: str | Path,
    *,
    scope: str = "all",
    absorbing_seed_base: int = 20250301,
    switching_seed_base: int = 20260722,
) -> MCSettings:
    mode = str(mode).strip().lower()
    if mode not in {"smoke", "paper"}:
        raise ValueError("mode must be 'smoke' or 'paper'.")
    scope = str(scope).strip().lower()
    if scope not in {"all", "absorbing", "switching"}:
        raise ValueError("scope must be 'all', 'absorbing', or 'switching'.")
    if mode == "paper":
        absorbing_replications = 500
        switching_replications = 1000
    else:
        absorbing_replications = 2
        switching_replications = 2
    return MCSettings(
        mode=mode,
        scope=scope,
        output_dir=str(Path(output_dir).resolve()),
        absorbing_replications=absorbing_replications,
        switching_replications=switching_replications,
        absorbing_seed_base=int(absorbing_seed_base),
        switching_seed_base=int(switching_seed_base),
    )


def _safe_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(map(str, value))
    if isinstance(value, list):
        return tuple(map(str, value))
    return (str(value),)


def _json_tuple(value: Any) -> str:
    return json.dumps(list(_safe_tuple(value)), ensure_ascii=False)


def _scalar_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise RuntimeError("The prespecified policy-window scalar is absent.")
    if "term" in frame:
        selected = frame.loc[
            frame["term"].astype(str).str.contains(
                "policy-window|ATT avg", case=False, regex=True
            )
        ]
        if not selected.empty:
            return selected.iloc[0]
    return frame.iloc[0]


def _long_frame(
    y: np.ndarray,
    treatment: np.ndarray,
    covariates: dict[str, np.ndarray],
) -> pd.DataFrame:
    n, periods = y.shape
    ids = np.repeat(np.arange(n), periods)
    times = np.tile(np.arange(1, periods + 1), n)
    out = pd.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y.reshape(-1),
            "D": treatment.reshape(-1).astype(int),
        }
    )
    for name, values in covariates.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            out[name] = np.repeat(arr, periods)
        elif arr.shape == (n, periods):
            out[name] = arr.reshape(-1)
        else:
            raise ValueError(f"Unexpected shape for {name}: {arr.shape}.")
    return out


def _fit_lpdid(
    panel: pd.DataFrame,
    covariates: Iterable[str] | None,
    *,
    max_post: int,
) -> dict[str, Any]:
    result = LPDID(
        target_estimand="rw",
        base_period=-1,
        clean_control="not_yet_treated",
        inference="cluster",
        max_pre=1,
        max_post=max_post,
        support_policy="supported_subset",
    ).fit(
        panel,
        outcome="y",
        unit="id",
        time="time",
        first_treat="g",
        covariates=list(covariates or []),
    )
    scalar = _scalar_row(result.scalars)
    return {
        "estimate": float(scalar["estimate"]),
        "se": float(scalar["se"]),
        "event": result.event_study.copy(),
        "diagnostics": {},
    }


def _fit_dr_core(
    panel: pd.DataFrame,
    *,
    method: str,
    design: str,
    ps_basis: Iterable[str],
    or_basis: Iterable[str],
    horizons: tuple[int, int],
    scalar_window: tuple[int, int],
    stabilization_window: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "estimation_method": method,
        "design": design,
        "horizons": horizons,
        "post_window": scalar_window,
        "inference": "cluster",
    }
    if design == "absorbing":
        kwargs["control_group"] = "not_yet_treated"
    else:
        kwargs["stabilization_window"] = int(stabilization_window)
    estimator = DRLPDID(**kwargs)
    core = estimator._fit_core(
        panel,
        outcome="y",
        unit="id",
        time="time",
        first_treat="g" if design == "absorbing" else None,
        treatment="D" if design == "switching" else None,
        covariates=[],
        compute_influence=True,
        _ps_covariates=list(ps_basis),
        _or_covariates=list(or_basis),
    )
    scalar = _scalar_row(core["scalars"])
    return {
        "estimate": float(scalar["estimate"]),
        "se": float(scalar["se"]),
        "event": core["event_study"].copy(),
        "diagnostics": core["metadata"].get("nuisance_diagnostics", {}),
    }


def _fit_record(
    *,
    experiment: str,
    design: str,
    N: int,
    replication: int,
    seed: int,
    scenario: str,
    delta: float,
    pt_mode: str,
    estimator: str,
    truth: float,
    fit_callable,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None]:
    scalar = {
        "experiment": experiment,
        "design": design,
        "N": int(N),
        "replication": int(replication),
        "seed": int(seed),
        "scenario": scenario,
        "delta": float(delta),
        "pt_mode": pt_mode,
        "estimator": estimator,
        "truth": float(truth),
        "estimate": np.nan,
        "se": np.nan,
        "status": "error",
        "error": "",
    }
    try:
        fit = fit_callable()
        estimate = float(fit["estimate"])
        se = float(fit["se"])
        if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
            raise RuntimeError("Non-finite estimate or non-positive standard error.")
        scalar.update(estimate=estimate, se=se, status="ok")
        event = fit["event"].copy()
        event["experiment"] = experiment
        event["design"] = design
        event["N"] = int(N)
        event["replication"] = int(replication)
        event["seed"] = int(seed)
        event["scenario"] = scenario
        event["delta"] = float(delta)
        event["pt_mode"] = pt_mode
        event["estimator"] = estimator
        diagnostic_rows: list[dict[str, Any]] = []
        for horizon, values in fit.get("diagnostics", {}).items():
            h = int(horizon)
            event_h = event.loc[event["horizon"].eq(h)]
            n_event = (
                int(event_h.iloc[0].get("n_event_rows", 0))
                if not event_h.empty else 0
            )
            n_control = (
                int(event_h.iloc[0].get("n_control_rows", 0))
                if not event_h.empty else 0
            )
            raw_balance = float(values.get("ipt_balance_error", np.nan))
            n_rows = n_event + n_control
            score_per_treated = (
                raw_balance * n_rows / n_event
                if n_event > 0 and np.isfinite(raw_balance) else np.nan
            )
            diagnostic_rows.append(
                {
                    "experiment": experiment,
                    "design": design,
                    "N": int(N),
                    "replication": int(replication),
                    "seed": int(seed),
                    "scenario": scenario,
                    "delta": float(delta),
                    "pt_mode": pt_mode,
                    "estimator": estimator,
                    "horizon": h,
                    "n_rows": n_rows,
                    "n_event_rows": n_event,
                    "ipt_balance_error_mean_score": raw_balance,
                    "ipt_balance_error_per_treated": score_per_treated,
                    "nested_basis": bool(
                        values.get("ipt_dript_nested_basis", False)
                    ),
                    "nested_difference": float(
                        values.get("ipt_dript_nested_difference", np.nan)
                    ),
                    "identity_tolerance": float(
                        values.get("ipt_dript_identity_tolerance", np.nan)
                    ),
                    "nested_or_balance_error": float(
                        values.get("ipt_nested_or_balance_error", np.nan)
                    ),
                    "ipt_retained_columns": _json_tuple(
                        values.get("ipt_retained_columns")
                    ),
                    "ipt_dropped_columns": _json_tuple(
                        values.get("ipt_dropped_columns")
                    ),
                    "or_retained_columns": _json_tuple(
                        values.get("or_retained_columns")
                    ),
                    "or_dropped_columns": _json_tuple(
                        values.get("or_dropped_columns")
                    ),
                }
            )
        return scalar, event, diagnostic_rows, None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        scalar["error"] = message
        failure = {
            key: scalar[key]
            for key in (
                "experiment", "design", "N", "replication", "seed",
                "scenario", "delta", "pt_mode", "estimator", "error",
            )
        }
        return scalar, pd.DataFrame(), [], failure


# ---------------------------------------------------------------------------
# Absorbing-adoption DGP
# ---------------------------------------------------------------------------

def simulate_absorbing(
    seed: int,
    *,
    N: int = 500,
    T: int = 17,
    max_post: int = 6,
    drift: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(N, 4))
    score = -x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2] - 0.2 * x[:, 3]
    cohorts = np.arange(9, 15)
    utilities = np.column_stack(
        [0.9 * (1.0 - j / 6.0) * score for j in range(1, 7)]
        + [np.zeros(N)]
    )
    probabilities = np.exp(utilities - utilities.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    choices = np.r_[cohorts, 0]
    g = np.array([rng.choice(choices, p=probabilities[i]) for i in range(N)])
    alpha_mean = np.where(g > 0, g, T + 1)
    alpha = rng.normal(alpha_mean, 1.0)

    rows: list[tuple] = []
    effects = np.zeros((N, T), dtype=float)
    for i in range(N):
        for t in range(1, T + 1):
            y0 = (
                210.0
                + (t / T)
                * (
                    27.4 * x[i, 0]
                    + 13.7 * x[i, 1]
                    + 13.7 * x[i, 2]
                    + 13.7 * x[i, 3]
                )
                + t
                + alpha[i]
                + rng.normal()
            )
            if g[i] > 0:
                y0 += drift * (t / T) * (1.0 + 0.2 * x[i, 0])
            tau = (
                max(t - g[i] + 1, 0) * (1.0 + 0.1 * x[i, 0])
                if g[i] > 0 else 0.0
            )
            effects[i, t - 1] = tau
            rows.append((i, t, int(g[i]), y0 + tau, *x[i]))
    panel = pd.DataFrame(
        rows,
        columns=["id", "time", "g", "y", "x1", "x2", "x3", "x4"],
    )

    truth_rows = []
    for h in range(-1, max_post + 1):
        if h < 0:
            truth_rows.append({"horizon": h, "truth": 0.0})
            continue
        valid = (g > 0) & (g + h <= T)
        value = (h + 1) * (1.0 + 0.1 * x[valid, 0])
        truth_rows.append({"horizon": h, "truth": float(np.mean(value))})
    truth = pd.DataFrame(truth_rows)
    scalar_truth = float(
        truth.loc[truth["horizon"].between(0, max_post), "truth"].mean()
    )
    return panel, truth, scalar_truth


def _run_absorbing_replication(
    replication: int,
    settings: MCSettings,
) -> dict[str, pd.DataFrame]:
    seed = settings.absorbing_seed_base + int(replication)
    scalar_rows: list[dict[str, Any]] = []
    event_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    panel, truth_path, truth = simulate_absorbing(
        seed,
        N=settings.absorbing_N,
        T=settings.absorbing_T,
        max_post=settings.absorbing_post,
        drift=0.0,
    )
    truth_map = truth_path.set_index("horizon")["truth"]
    cache: dict[tuple, dict[str, Any]] = {}

    for scenario, bases in ABS_SCENARIOS.items():
        specifications: list[tuple[str, tuple, Any]] = [
            (
                "LPDID-RW",
                ("lpdid-rw",),
                lambda: _fit_lpdid(
                    panel, None, max_post=settings.absorbing_post
                ),
            ),
            (
                "LPDID-RW+X",
                ("lpdid-rwx", tuple(bases["or"])),
                lambda bases=bases: _fit_lpdid(
                    panel, bases["or"], max_post=settings.absorbing_post
                ),
            ),
        ]
        for method, label in METHOD_LABELS.items():
            cache_key = (
                method,
                tuple(bases["ps"]) if method != "ra" else (),
                tuple(bases["or"]) if method in {"ra", "dr-ipw", "dr-ipt"} else (),
            )
            specifications.append(
                (
                    label,
                    cache_key,
                    lambda method=method, bases=bases: _fit_dr_core(
                        panel,
                        method=method,
                        design="absorbing",
                        ps_basis=bases["ps"],
                        or_basis=bases["or"],
                        horizons=(-1, settings.absorbing_post),
                        scalar_window=settings.scalar_window,
                    ),
                )
            )

        for estimator, key, fit_callable in specifications:
            def cached_fit(key=key, fit_callable=fit_callable):
                if key not in cache:
                    cache[key] = fit_callable()
                return cache[key]

            scalar, event, diag, failure = _fit_record(
                experiment="absorbing_main",
                design="absorbing",
                N=settings.absorbing_N,
                replication=replication,
                seed=seed,
                scenario=scenario,
                delta=0.0,
                pt_mode="none",
                estimator=estimator,
                truth=truth,
                fit_callable=cached_fit,
            )
            scalar_rows.append(scalar)
            if not event.empty:
                event["truth"] = event["horizon"].map(truth_map)
                event_parts.append(event)
            diagnostics.extend(diag)
            if failure:
                failures.append(failure)

    # Parallel-trends sensitivity. Delta=0 reuses the main full-basis fits.
    main_scalar = pd.DataFrame(scalar_rows)
    main_events = (
        pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    )
    for delta in settings.pt_deltas:
        if delta == 0.0:
            s0 = main_scalar.loc[
                main_scalar["scenario"].eq("A_both_correct")
            ].copy()
            s0["experiment"] = "absorbing_pt"
            scalar_rows.extend(s0.to_dict("records"))
            e0 = main_events.loc[
                main_events["scenario"].eq("A_both_correct")
            ].copy()
            e0["experiment"] = "absorbing_pt"
            if not e0.empty:
                event_parts.append(e0)
            d0 = pd.DataFrame(diagnostics)
            if not d0.empty:
                d0 = d0.loc[d0["scenario"].eq("A_both_correct")].copy()
                d0["experiment"] = "absorbing_pt"
                diagnostics.extend(d0.to_dict("records"))
            continue

        panel_d, truth_path_d, truth_d = simulate_absorbing(
            seed,
            N=settings.absorbing_N,
            T=settings.absorbing_T,
            max_post=settings.absorbing_post,
            drift=delta,
        )
        truth_map_d = truth_path_d.set_index("horizon")["truth"]
        full = ABS_SCENARIOS["A_both_correct"]
        specifications = [
            (
                "LPDID-RW",
                lambda: _fit_lpdid(
                    panel_d, None, max_post=settings.absorbing_post
                ),
            ),
            (
                "LPDID-RW+X",
                lambda: _fit_lpdid(
                    panel_d, full["or"], max_post=settings.absorbing_post
                ),
            ),
        ] + [
            (
                label,
                lambda method=method: _fit_dr_core(
                    panel_d,
                    method=method,
                    design="absorbing",
                    ps_basis=full["ps"],
                    or_basis=full["or"],
                    horizons=(-1, settings.absorbing_post),
                    scalar_window=settings.scalar_window,
                ),
            )
            for method, label in METHOD_LABELS.items()
        ]
        for estimator, fit_callable in specifications:
            scalar, event, diag, failure = _fit_record(
                experiment="absorbing_pt",
                design="absorbing",
                N=settings.absorbing_N,
                replication=replication,
                seed=seed,
                scenario="A_both_correct",
                delta=delta,
                pt_mode="treated_drift",
                estimator=estimator,
                truth=truth_d,
                fit_callable=fit_callable,
            )
            scalar_rows.append(scalar)
            if not event.empty:
                event["truth"] = event["horizon"].map(truth_map_d)
                event_parts.append(event)
            diagnostics.extend(diag)
            if failure:
                failures.append(failure)

    return {
        "scalar": pd.DataFrame(scalar_rows),
        "event": (
            pd.concat(event_parts, ignore_index=True)
            if event_parts else pd.DataFrame()
        ),
        "diagnostic": pd.DataFrame(diagnostics),
        "failure": pd.DataFrame(failures),
    }


# ---------------------------------------------------------------------------
# Sustained-switch-in DGP
# ---------------------------------------------------------------------------

def _switch_effect(h: int, x1: float, L: int) -> float:
    capped = min(int(h), int(L) - 1)
    return 0.25 * (capped + 1.0) ** 2 * math.exp(0.10 * float(x1))


def simulate_switching(
    seed: int,
    *,
    N: int,
    T: int = 51,
    L: int = 9,
    post: int = 10,
    pt_delta: float = 0.0,
    pt_mode: str = "none",
) -> dict[str, Any]:
    if pt_mode not in {"none", "pretrend", "post_only"}:
        raise ValueError("pt_mode must be 'none', 'pretrend', or 'post_only'.")
    rng = np.random.default_rng(int(seed))
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    x2_sq = x2**2 - 1.0
    alpha = rng.normal(scale=0.7, size=N)
    gamma = rng.normal(scale=0.35, size=T)
    innovation = rng.normal(scale=1.0, size=(N, T))

    y0_base = np.empty((N, T), dtype=float)
    for j in range(T):
        trend_x = (j + 1) / T * (0.80 * x1 + 2.50 * x2_sq)
        moving_average = innovation[:, j]
        if j > 0:
            moving_average = moving_average + 0.98 * innovation[:, j - 1]
        y0_base[:, j] = 14.75 + alpha + gamma[j] + trend_x + moving_average

    dy0 = np.full_like(y0_base, np.nan)
    dy0[:, 1:] = np.diff(y0_base, axis=1)
    lag_scale = float(np.nanstd(dy0, ddof=1))
    lag_dy_std = np.zeros_like(y0_base)
    lag_dy_std[:, 2:] = dy0[:, 1:-1] / lag_scale

    treatment = np.zeros((N, T), dtype=int)
    effect = np.zeros((N, T), dtype=float)
    events: list[tuple[int, int]] = []
    ever_entered = np.zeros(N, dtype=bool)
    duration = post + 1

    for opportunity_index, t_one in enumerate((20, 35)):
        j = t_one - 1
        changes = np.zeros_like(treatment)
        changes[:, 1:] = treatment[:, 1:] - treatment[:, :-1]
        clean = np.all(changes[:, j - L:j] == 0, axis=1)
        off = treatment[:, j - 1] == 0
        eligible = clean & off & ~ever_entered
        index = (
            -0.25
            + 0.50 * x1
            - 0.85 * x2_sq
            + 0.35 * lag_dy_std[:, j]
            + 0.15 * opportunity_index
        )
        enter = eligible & rng.binomial(1, expit(index), size=N).astype(bool)
        for i in np.flatnonzero(enter):
            end = min(j + duration, T)
            treatment[i, j:end] = 1
            for s in range(j, end):
                effect[i, s] += _switch_effect(s - j, x1[i], L)
            last = effect[i, end - 1]
            for q in range(L):
                s = end + q
                if s >= T:
                    break
                effect[i, s] += last * max(1.0 - (q + 1) / L, 0.0)
            events.append((int(i), int(j)))
        ever_entered |= enter

    first_event = np.full(N, -1, dtype=int)
    for i, j in events:
        if first_event[i] < 0:
            first_event[i] = j
    switcher = first_event >= 0
    drift = np.zeros((N, T), dtype=float)
    if pt_delta:
        for i in np.flatnonzero(switcher):
            if pt_mode == "pretrend":
                drift[i] = (
                    pt_delta
                    * (np.arange(1, T + 1) / T)
                    * (1.0 + 0.2 * x1[i])
                )
            elif pt_mode == "post_only":
                relative = np.maximum(np.arange(T) - first_event[i] + 1, 0)
                drift[i] = (
                    pt_delta * (relative / T) * (1.0 + 0.2 * x1[i])
                )

    y0 = y0_base + drift
    y = y0 + effect
    panel = _long_frame(
        y,
        treatment,
        {
            "x1": x1,
            "x2_sq": x2_sq,
            "lag_dy_std": lag_dy_std,
        },
    )
    return {
        "data": panel,
        "D": treatment,
        "effect": effect,
        "events": events,
    }


def _eligible_switch_events(
    treatment: np.ndarray,
    *,
    L: int,
    max_post: int,
) -> list[tuple[int, int, int]]:
    n, periods = treatment.shape
    ledger: list[tuple[int, int, int]] = []
    for i in range(n):
        changes = np.zeros(periods, dtype=int)
        changes[1:] = treatment[i, 1:] - treatment[i, :-1]
        for t in range(1, periods):
            if changes[t] != 1 or t - L < 0:
                continue
            if np.any(changes[t - L:t] != 0):
                continue
            max_h = -1
            for h in range(max_post + 1):
                if (
                    t + h >= periods
                    or np.any(treatment[i, t:t + h + 1] != 1)
                ):
                    break
                max_h = h
            if max_h >= 0:
                ledger.append((i, t, max_h))
    return ledger


def switching_truth(
    treatment: np.ndarray,
    effect: np.ndarray,
    *,
    L: int,
    pre: int,
    post: int,
    scalar_window: tuple[int, int],
) -> tuple[pd.DataFrame, float]:
    ledger = _eligible_switch_events(treatment, L=L, max_post=post)
    rows: list[dict[str, Any]] = []
    for h in range(-pre, post + 1):
        if h < 0:
            rows.append(
                {"horizon": h, "truth": 0.0, "n_true_events": len(ledger)}
            )
            continue
        values = [
            effect[i, t + h] for i, t, max_h in ledger if max_h >= h
        ]
        rows.append(
            {
                "horizon": h,
                "truth": float(np.mean(values)) if values else np.nan,
                "n_true_events": len(values),
            }
        )
    truth = pd.DataFrame(rows)
    lo, hi = scalar_window
    scalar = float(
        truth.loc[truth["horizon"].between(lo, hi), "truth"].mean()
    )
    return truth, scalar


def _run_switching_replication(
    replication: int,
    N: int,
    settings: MCSettings,
) -> dict[str, pd.DataFrame]:
    seed = (
        settings.switching_seed_base
        + N * 7919
        + int(replication)
    )
    scalar_rows: list[dict[str, Any]] = []
    event_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    simulation = simulate_switching(
        seed,
        N=N,
        T=settings.switching_T,
        L=settings.switching_L,
        post=settings.switching_post,
    )
    truth_path, truth = switching_truth(
        simulation["D"],
        simulation["effect"],
        L=settings.switching_L,
        pre=settings.switching_pre,
        post=settings.switching_post,
        scalar_window=settings.scalar_window,
    )
    truth_map = truth_path.set_index("horizon")["truth"]
    cache: dict[tuple, dict[str, Any]] = {}
    for scenario, bases in SW_SCENARIOS.items():
        for method, label in METHOD_LABELS.items():
            key = (
                method,
                tuple(bases["ps"]) if method != "ra" else (),
                tuple(bases["or"]) if method in {"ra", "dr-ipw", "dr-ipt"} else (),
            )

            def cached_fit(key=key, method=method, bases=bases):
                if key not in cache:
                    cache[key] = _fit_dr_core(
                        simulation["data"],
                        method=method,
                        design="switching",
                        ps_basis=bases["ps"],
                        or_basis=bases["or"],
                        horizons=(
                            -settings.switching_pre,
                            settings.switching_post,
                        ),
                        scalar_window=settings.scalar_window,
                        stabilization_window=settings.switching_L,
                    )
                return cache[key]

            scalar, event, diag, failure = _fit_record(
                experiment="switching_main",
                design="switching",
                N=N,
                replication=replication,
                seed=seed,
                scenario=scenario,
                delta=0.0,
                pt_mode="none",
                estimator=label,
                truth=truth,
                fit_callable=cached_fit,
            )
            scalar_rows.append(scalar)
            if not event.empty:
                event["truth"] = event["horizon"].map(truth_map)
                event["n_true_events"] = event["horizon"].map(
                    truth_path.set_index("horizon")["n_true_events"]
                )
                event_parts.append(event)
            diagnostics.extend(diag)
            if failure:
                failures.append(failure)

    main_scalar = pd.DataFrame(scalar_rows)
    main_events = (
        pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    )
    main_diag = pd.DataFrame(diagnostics)
    for mode in settings.switching_pt_modes:
        for delta in settings.pt_deltas:
            if delta == 0.0:
                s0 = main_scalar.loc[
                    main_scalar["scenario"].eq("A_both_correct")
                    & main_scalar["estimator"].eq("DRLPDID-IPT")
                ].copy()
                s0["experiment"] = "switching_pt"
                s0["pt_mode"] = mode
                scalar_rows.extend(s0.to_dict("records"))
                e0 = main_events.loc[
                    main_events["scenario"].eq("A_both_correct")
                    & main_events["estimator"].eq("DRLPDID-IPT")
                ].copy()
                e0["experiment"] = "switching_pt"
                e0["pt_mode"] = mode
                if not e0.empty:
                    event_parts.append(e0)
                if not main_diag.empty:
                    d0 = main_diag.loc[
                        main_diag["scenario"].eq("A_both_correct")
                        & main_diag["estimator"].eq("DRLPDID-IPT")
                    ].copy()
                    d0["experiment"] = "switching_pt"
                    d0["pt_mode"] = mode
                    diagnostics.extend(d0.to_dict("records"))
                continue

            sensitivity_seed = seed + int(1000 * delta)
            sim_d = simulate_switching(
                sensitivity_seed,
                N=N,
                T=settings.switching_T,
                L=settings.switching_L,
                post=settings.switching_post,
                pt_delta=delta,
                pt_mode=mode,
            )
            truth_path_d, truth_d = switching_truth(
                sim_d["D"],
                sim_d["effect"],
                L=settings.switching_L,
                pre=settings.switching_pre,
                post=settings.switching_post,
                scalar_window=settings.scalar_window,
            )
            truth_map_d = truth_path_d.set_index("horizon")["truth"]
            full = SW_SCENARIOS["A_both_correct"]
            scalar, event, diag, failure = _fit_record(
                experiment="switching_pt",
                design="switching",
                N=N,
                replication=replication,
                seed=sensitivity_seed,
                scenario="A_both_correct",
                delta=delta,
                pt_mode=mode,
                estimator="DRLPDID-IPT",
                truth=truth_d,
                fit_callable=lambda: _fit_dr_core(
                    sim_d["data"],
                    method="dr-ipt",
                    design="switching",
                    ps_basis=full["ps"],
                    or_basis=full["or"],
                    horizons=(
                        -settings.switching_pre,
                        settings.switching_post,
                    ),
                    scalar_window=settings.scalar_window,
                    stabilization_window=settings.switching_L,
                ),
            )
            scalar_rows.append(scalar)
            if not event.empty:
                event["truth"] = event["horizon"].map(truth_map_d)
                event_parts.append(event)
            diagnostics.extend(diag)
            if failure:
                failures.append(failure)

    return {
        "scalar": pd.DataFrame(scalar_rows),
        "event": (
            pd.concat(event_parts, ignore_index=True)
            if event_parts else pd.DataFrame()
        ),
        "diagnostic": pd.DataFrame(diagnostics),
        "failure": pd.DataFrame(failures),
    }


# ---------------------------------------------------------------------------
# Checkpointing, summaries, diagnostics, and article files
# ---------------------------------------------------------------------------

def _checkpoint(path: Path, builder):
    if path.exists():
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (EOFError, pickle.UnpicklingError) as exc:
            print(
                f"Rebuilding incomplete checkpoint {path.name}: "
                f"{type(exc).__name__}"
            )
    value = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return value


def _combine(results: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    keys = ("scalar", "event", "diagnostic", "failure")
    combined: dict[str, pd.DataFrame] = {}
    for key in keys:
        frames = [item[key] for item in results if not item[key].empty]
        combined[key] = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
    return combined


def _summarize_scalar(
    raw: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, dropna=False, sort=False):
        valid = group["status"].eq("ok")
        g = group.loc[valid]
        error = g["estimate"].to_numpy(float) - g["truth"].to_numpy(float)
        covered = np.abs(error) <= Z95 * g["se"].to_numpy(float)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            dict(zip(group_columns, key_tuple))
            | {
                "replications_requested": int(len(group)),
                "replications_valid": int(len(g)),
                "valid_rate": float(valid.mean()),
                "bias": float(np.mean(error)) if len(error) else np.nan,
                "rmse": (
                    float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
                ),
                "coverage": (
                    float(np.mean(covered)) if len(covered) else np.nan
                ),
                "mean_se": float(g["se"].mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize_event(
    raw: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, dropna=False, sort=False):
        valid = (
            group["estimate"].notna()
            & group["se"].notna()
            & group["truth"].notna()
        )
        g = group.loc[valid]
        error = g["estimate"].to_numpy(float) - g["truth"].to_numpy(float)
        covered = np.abs(error) <= Z95 * g["se"].to_numpy(float)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            dict(zip(group_columns, key_tuple))
            | {
                "replications_valid": int(len(g)),
                "mean_truth": float(g["truth"].mean()) if len(g) else np.nan,
                "mean_estimate": (
                    float(g["estimate"].mean()) if len(g) else np.nan
                ),
                "bias": float(np.mean(error)) if len(error) else np.nan,
                "rmse": (
                    float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
                ),
                "coverage": (
                    float(np.mean(covered)) if len(covered) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _pair_identity(event: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "experiment", "design", "N", "replication", "seed", "scenario",
        "delta", "pt_mode", "horizon",
    ]
    if event.empty or not set(keys + ["estimator", "estimate"]).issubset(
        event.columns
    ):
        return pd.DataFrame(
            columns=keys + [
                "IPT", "DRLPDID-IPT", "absolute_difference",
                "expected_nested",
            ]
        )
    subset = event.loc[
        event["estimator"].isin(["IPT", "DRLPDID-IPT"]),
        keys + ["estimator", "estimate"],
    ]
    wide = subset.pivot_table(
        index=keys, columns="estimator", values="estimate", aggfunc="first"
    ).reset_index()
    if {"IPT", "DRLPDID-IPT"}.issubset(wide.columns):
        wide["absolute_difference"] = (
            wide["IPT"] - wide["DRLPDID-IPT"]
        ).abs()
    else:
        wide["absolute_difference"] = np.nan
    wide["expected_nested"] = wide["scenario"].isin(
        ["A_both_correct", "B_or_misspecified", "D_both_misspecified"]
    )
    return wide


def _diagnostic_summary(diagnostic: pd.DataFrame) -> pd.DataFrame:
    if diagnostic.empty:
        return pd.DataFrame()
    d = diagnostic.copy()
    d["identity_pass"] = (
        ~d["nested_basis"]
        | (
            d["nested_difference"].notna()
            & d["identity_tolerance"].notna()
            & (d["nested_difference"] <= d["identity_tolerance"])
        )
    )
    d["balance_pass"] = (
        d["ipt_balance_error_mean_score"].notna()
        & (d["ipt_balance_error_mean_score"] <= 1e-9)
    )
    group = [
        "experiment", "design", "N", "scenario", "delta", "pt_mode",
    ]
    return (
        d.groupby(group, dropna=False)
        .agg(
            diagnostic_rows=("horizon", "size"),
            max_mean_score_balance_error=(
                "ipt_balance_error_mean_score", "max"
            ),
            max_score_error_per_treated=(
                "ipt_balance_error_per_treated", "max"
            ),
            max_nested_difference=("nested_difference", "max"),
            max_identity_tolerance=("identity_tolerance", "max"),
            all_balance_checks_pass=("balance_pass", "all"),
            all_identity_checks_pass=("identity_pass", "all"),
            distinct_ipt_retained_bases=("ipt_retained_columns", "nunique"),
            distinct_or_retained_bases=("or_retained_columns", "nunique"),
        )
        .reset_index()
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    def escape(value: Any) -> str:
        text = str(value)
        for old, new in (
            ("\\", r"\textbackslash{}"),
            ("_", r"\_"),
            ("%", r"\%"),
            ("&", r"\&"),
            ("#", r"\#"),
        ):
            text = text.replace(old, new)
        return text

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        return escape(value)

    alignment = "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[column]) else "l"
        for column in frame.columns
    )
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(escape(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(format_value(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_article_monte_carlo(settings: MCSettings) -> dict[str, Any]:
    if __version__ != "0.7.2":
        raise RuntimeError(
            f"This notebook requires pydrlpdid 0.7.2; imported {__version__}."
        )
    started = time.time()
    root = Path(settings.output_dir)
    signature_payload = json.dumps(
        {
            "settings": asdict(settings),
            "package_version": __version__,
            "package_core_sha256": _hash_file(PACKAGE_SOURCE_FILE),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    run_signature = hashlib.sha256(signature_payload).hexdigest()
    checkpoint_root = root / "checkpoints" / run_signature[:16]
    table_root = root / "tables"
    audit_root = root / "audit"
    root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    run_absorbing = settings.scope in {"all", "absorbing"}
    run_switching = settings.scope in {"all", "switching"}

    absorbing_parts = []
    if run_absorbing:
        for replication in range(settings.absorbing_replications):
            path = checkpoint_root / "absorbing" / f"rep_{replication:04d}.pkl"
            absorbing_parts.append(
                _checkpoint(
                    path,
                    lambda replication=replication: _run_absorbing_replication(
                        replication, settings
                    ),
                )
            )
            if (
                (replication + 1) % settings.checkpoint_every == 0
                or replication + 1 == settings.absorbing_replications
            ):
                print(
                    f"Absorbing: {replication + 1}/"
                    f"{settings.absorbing_replications}"
                )

    switching_parts = []
    if run_switching:
        for N in settings.switching_N:
            for replication in range(settings.switching_replications):
                path = (
                    checkpoint_root
                    / f"switching_N{N}"
                    / f"rep_{replication:04d}.pkl"
                )
                switching_parts.append(
                    _checkpoint(
                        path,
                        lambda replication=replication, N=N:
                        _run_switching_replication(replication, N, settings),
                    )
                )
                if (
                    (replication + 1) % settings.checkpoint_every == 0
                    or replication + 1 == settings.switching_replications
                ):
                    print(
                        f"Switching N={N}: {replication + 1}/"
                        f"{settings.switching_replications}"
                    )

    combined = _combine(absorbing_parts + switching_parts)
    scalar = combined["scalar"]
    event = combined["event"]
    diagnostic = combined["diagnostic"]
    failures = combined["failure"]
    scalar.to_csv(root / "scalar_replication_level.csv", index=False)
    event.to_csv(root / "event_study_replication_level.csv", index=False)
    diagnostic.to_csv(audit_root / "ipt_score_rank_identity_raw.csv", index=False)
    failures.to_csv(audit_root / "failures.csv", index=False)

    absorbing_main = _summarize_scalar(
        scalar.loc[scalar["experiment"].eq("absorbing_main")],
        ["scenario", "estimator"],
    )
    absorbing_pt = _summarize_scalar(
        scalar.loc[scalar["experiment"].eq("absorbing_pt")],
        ["delta", "estimator"],
    )
    absorbing_horizon = _summarize_event(
        event.loc[
            event["experiment"].eq("absorbing_main")
            & event["scenario"].eq("A_both_correct")
            & event["estimator"].isin(["DRLPDID-RA", "IPT", "DRLPDID-IPT"])
        ],
        ["horizon", "estimator"],
    )
    switching_main = _summarize_scalar(
        scalar.loc[scalar["experiment"].eq("switching_main")],
        ["N", "scenario", "estimator"],
    )
    switching_path = _summarize_event(
        event.loc[
            event["experiment"].eq("switching_main")
            & event["scenario"].eq("A_both_correct")
            & event["estimator"].eq("DRLPDID-IPT")
        ],
        ["N", "horizon"],
    )
    switching_pt = _summarize_scalar(
        scalar.loc[scalar["experiment"].eq("switching_pt")],
        ["N", "pt_mode", "delta", "estimator"],
    )

    if run_absorbing:
        _write_frame(absorbing_main, table_root / "table_absorbing_N500")
        _write_frame(absorbing_pt, table_root / "table_absorbing_pt_N500")
        _write_frame(
            absorbing_horizon,
            table_root / "table_absorbing_horizon_scenario_A_N500",
        )
    if run_switching:
        _write_frame(switching_main, table_root / "table_switching_main")
        _write_frame(switching_path, table_root / "table_switching_path")
        _write_frame(switching_pt, table_root / "table_switching_pt")

    pair_identity = _pair_identity(event)
    pair_identity.to_csv(
        audit_root / "ipt_dript_direct_event_identity.csv", index=False
    )
    diagnostic_summary = _diagnostic_summary(diagnostic)
    diagnostic_summary.to_csv(
        audit_root / "ipt_score_rank_identity_summary.csv", index=False
    )
    retained_columns = [
        "experiment", "design", "N", "replication", "scenario", "delta",
        "pt_mode", "horizon", "nested_basis", "ipt_retained_columns",
        "ipt_dropped_columns", "or_retained_columns", "or_dropped_columns",
    ]
    retained_basis = (
        diagnostic[retained_columns].drop_duplicates()
        if not diagnostic.empty
        else pd.DataFrame(columns=retained_columns)
    )
    retained_basis.to_csv(
        audit_root / "retained_basis_by_replication_horizon.csv", index=False
    )

    expected_abs = settings.absorbing_replications
    expected_sw = settings.switching_replications
    nested_diagnostics = diagnostic.loc[diagnostic["nested_basis"]].copy()
    identity_pass = bool(
        nested_diagnostics.empty
        or (
            nested_diagnostics["nested_difference"]
            <= nested_diagnostics["identity_tolerance"]
        ).all()
    )
    balance_pass = bool(
        diagnostic.empty
        or (diagnostic["ipt_balance_error_mean_score"] <= 1e-9).all()
    )
    absorbing_counts_complete = bool(
        not run_absorbing
        or scalar.loc[
            scalar["experiment"].eq("absorbing_main")
        ].groupby(["scenario", "estimator"])["replication"].nunique().eq(
            expected_abs
        ).all()
    )
    switching_counts_complete = bool(
        not run_switching
        or scalar.loc[
            scalar["experiment"].eq("switching_main")
        ].groupby(["N", "scenario", "estimator"])["replication"].nunique().eq(
            expected_sw
        ).all()
    )
    complete_counts = absorbing_counts_complete and switching_counts_complete
    paper_mode = settings.mode == "paper"
    absorbing_failures = failures.loc[
        failures.get("design", pd.Series(dtype=str)).eq("absorbing")
    ]
    switching_failures = failures.loc[
        failures.get("design", pd.Series(dtype=str)).eq("switching")
    ]
    absorbing_diag = diagnostic.loc[
        diagnostic.get("design", pd.Series(dtype=str)).eq("absorbing")
    ]
    switching_diag = diagnostic.loc[
        diagnostic.get("design", pd.Series(dtype=str)).eq("switching")
    ]

    def diagnostic_pass(frame: pd.DataFrame) -> tuple[bool, bool]:
        if frame.empty:
            return True, True
        balance_ok = bool(
            (frame["ipt_balance_error_mean_score"] <= 1e-9).all()
        )
        nested = frame.loc[frame["nested_basis"]]
        identity_ok = bool(
            nested.empty
            or (
                nested["nested_difference"] <= nested["identity_tolerance"]
            ).all()
        )
        return balance_ok, identity_ok

    abs_balance, abs_identity = diagnostic_pass(absorbing_diag)
    sw_balance, sw_identity = diagnostic_pass(switching_diag)
    certified_absorbing = bool(
        paper_mode
        and run_absorbing
        and absorbing_counts_complete
        and absorbing_failures.empty
        and abs_balance
        and abs_identity
    )
    certified_switching = bool(
        paper_mode
        and run_switching
        and switching_counts_complete
        and switching_failures.empty
        and sw_balance
        and sw_identity
    )
    certified_requested_scope = bool(
        (not run_absorbing or certified_absorbing)
        and (not run_switching or certified_switching)
    )
    certification = {
        "mode": settings.mode,
        "scope": settings.scope,
        "run_signature": run_signature,
        "package_version": __version__,
        "package_core_file": str(PACKAGE_SOURCE_FILE),
        "package_core_sha256": _hash_file(PACKAGE_SOURCE_FILE),
        "paper_replication_counts": paper_mode,
        "zero_failed_fits": bool(failures.empty),
        "complete_replication_counts": complete_counts,
        "absorbing_replication_counts_complete": absorbing_counts_complete,
        "switching_replication_counts_complete": switching_counts_complete,
        "all_ipt_mean_score_checks_pass": balance_pass,
        "all_nested_identity_checks_pass": identity_pass,
        "certified_absorbing": certified_absorbing,
        "certified_switching": certified_switching,
        "certified_for_requested_scope": certified_requested_scope,
        "certified_for_article": bool(
            settings.scope == "all"
            and certified_absorbing
            and certified_switching
        ),
        "maximum_ipt_mean_score_error": (
            float(diagnostic["ipt_balance_error_mean_score"].max())
            if not diagnostic.empty else np.nan
        ),
        "maximum_ipt_score_error_per_treated": (
            float(diagnostic["ipt_balance_error_per_treated"].max())
            if not diagnostic.empty else np.nan
        ),
        "maximum_nested_identity_difference": (
            float(nested_diagnostics["nested_difference"].max())
            if not nested_diagnostics.empty else np.nan
        ),
        "maximum_nested_identity_tolerance": (
            float(nested_diagnostics["identity_tolerance"].max())
            if not nested_diagnostics.empty else np.nan
        ),
        "failure_count": int(len(failures)),
        "runtime_seconds": float(time.time() - started),
    }
    (root / "certification.json").write_text(
        json.dumps(certification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    configuration = {
        **asdict(settings),
        "run_signature": run_signature,
        "package_version": __version__,
        "package_core_file": str(PACKAGE_SOURCE_FILE),
        "package_core_sha256": _hash_file(PACKAGE_SOURCE_FILE),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (root / "configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "checkpoints" not in path.parts:
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _hash_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(root / "SHA256SUMS.csv", index=False)

    if paper_mode and not certification["certified_for_requested_scope"]:
        raise RuntimeError(
            "The requested paper-mode scope was not certified. Inspect "
            "audit/failures.csv and certification.json. The corresponding "
            "article tables must not be updated."
        )
    return {
        "settings": settings,
        "scalar_raw": scalar,
        "event_raw": event,
        "diagnostic_raw": diagnostic,
        "failures": failures,
        "absorbing_main": absorbing_main,
        "absorbing_pt": absorbing_pt,
        "absorbing_horizon": absorbing_horizon,
        "switching_main": switching_main,
        "switching_path": switching_path,
        "switching_pt": switching_pt,
        "identity_audit": pair_identity,
        "diagnostic_summary": diagnostic_summary,
        "certification": certification,
    }
