"""Article Monte Carlo: absorbing adoption, N=500.

This is the dedicated replication driver for the absorbing-adoption experiment
in the article.  The public DRLPDID API intentionally uses one covariate list.
For this controlled misspecification experiment only, the script calls the
private replication hook ``_fit_core`` with distinct PS and OR bases.  This
does not add nuisance-model choices to the public estimator.

Run from the pydrlpdid-0.7.2 project root:

    python examples/monte_carlo_absorbing_N500.py

Environment variables:
    DRLPDID_MC_REPS       number of replications (default 500)
    DRLPDID_MC_SEED       first replication seed (default 20250301)
    DRLPDID_MC_OUTPUT     output directory (default mc_absorbing_N500_results)

No failed fit is silently discarded.  A nonzero failure count stops the final
article-table export.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, LPDID, __version__


N = 500
T = 17
H_POST = 6
REPLICATIONS = int(os.environ.get("DRLPDID_MC_REPS", "500"))
FIRST_SEED = int(os.environ.get("DRLPDID_MC_SEED", "20250301"))
OUTPUT_DIR = Path(
    os.environ.get("DRLPDID_MC_OUTPUT", "mc_absorbing_N500_results")
).resolve()
FULL = ["x1", "x2", "x3", "x4"]
PARTIAL = ["x1", "x2"]
SCENARIOS = {
    "A_both_correct": (FULL, FULL),
    "B_or_misspecified": (FULL, PARTIAL),
    "C_ps_misspecified": (PARTIAL, FULL),
    "D_both_misspecified": (PARTIAL, PARTIAL),
}
METHODS = {
    "DRLPDID-RA": "ra",
    "IPW": "ipw",
    "IPT": "ipt",
    "DRLPDID-IPW": "dr-ipw",
    "DRLPDID-IPT": "dr-ipt",
}


def make_panel(seed: int, *, drift: float = 0.0) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
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

    rows = []
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
                if g[i] > 0
                else 0.0
            )
            rows.append(
                (i, t, int(g[i]), y0 + tau, *x[i])
            )
    panel = pd.DataFrame(
        rows, columns=["id", "t", "g", "y", "x1", "x2", "x3", "x4"]
    )

    truths = []
    for h in range(H_POST + 1):
        valid = (g > 0) & (g + h <= T)
        truths.append(float(np.mean((h + 1) * (1.0 + 0.1 * x[valid, 0]))))
    return panel, float(np.mean(truths))


def scalar_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise RuntimeError("The prespecified post-window scalar is absent.")
    if "term" in frame.columns:
        preferred = frame.loc[
            frame["term"].astype(str).str.contains("policy-window|ATT avg", regex=True)
        ]
        if not preferred.empty:
            return preferred.iloc[0]
    return frame.iloc[0]


def fit_lpdid(
    panel: pd.DataFrame, covariates: list[str] | None
) -> tuple[float, float]:
    result = LPDID(
        target_estimand="rw",
        base_period=-1,
        clean_control="not_yet_treated",
        inference="cluster",
        max_pre=1,
        max_post=H_POST,
        support_policy="supported_subset",
    ).fit(
        panel,
        outcome="y",
        unit="id",
        time="t",
        first_treat="g",
        covariates=covariates,
    )
    row = scalar_row(result.scalars)
    return float(row["estimate"]), float(row["se"])


def fit_local_att(
    panel: pd.DataFrame,
    method: str,
    ps_basis: list[str],
    or_basis: list[str],
) -> tuple[float, float, dict]:
    estimator = DRLPDID(
        estimation_method=method,
        design="absorbing",
        control_group="not_yet_treated",
        horizons=(-1, H_POST),
        post_window=(0, H_POST),
        inference="cluster",
    )
    core = estimator._fit_core(
        panel,
        outcome="y",
        unit="id",
        time="t",
        first_treat="g",
        treatment=None,
        covariates=[],
        compute_influence=True,
        _ps_covariates=ps_basis,
        _or_covariates=or_basis,
    )
    row = scalar_row(core["scalars"])
    return (
        float(row["estimate"]),
        float(row["se"]),
        core["metadata"]["nuisance_diagnostics"],
    )


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw.loc[raw["status"].eq("ok")].copy()
    rows = []
    for (scenario, estimator), group in valid.groupby(
        ["scenario", "estimator"], sort=False
    ):
        error = group["estimate"] - group["truth"]
        covered = (
            (group["truth"] >= group["estimate"] - 1.959963984540054 * group["se"])
            & (group["truth"] <= group["estimate"] + 1.959963984540054 * group["se"])
        )
        rows.append(
            {
                "scenario": scenario,
                "estimator": estimator,
                "replications": int(len(group)),
                "bias": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "coverage": float(covered.mean()),
                "mean_se": float(group["se"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if __version__ != "0.7.2":
        raise RuntimeError(
            f"This script requires pydrlpdid 0.7.2; imported {__version__}."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for replication in range(REPLICATIONS):
        seed = FIRST_SEED + replication
        panel, truth = make_panel(seed)
        fit_cache: dict[tuple, tuple[float, float, dict]] = {}
        for scenario, (ps_basis, or_basis) in SCENARIOS.items():
            fits: list[tuple[str, tuple, callable]] = [
                (
                    "LPDID-RW",
                    ("LPDID-RW",),
                    lambda: (*fit_lpdid(panel, None), {}),
                ),
                (
                    "LPDID-RW+X",
                    ("LPDID-RW+X", tuple(or_basis)),
                    lambda: (*fit_lpdid(panel, or_basis), {}),
                ),
            ]
            fits.extend(
                (
                    label,
                    (
                        label,
                        tuple(ps_basis) if method in {"ipw", "ipt", "dr-ipw", "dr-ipt"} else (),
                        tuple(or_basis) if method in {"ra", "dr-ipw", "dr-ipt"} else (),
                    ),
                    lambda method=method: fit_local_att(
                        panel, method, ps_basis, or_basis
                    ),
                )
                for label, method in METHODS.items()
            )
            for label, cache_key, fit_callable in fits:
                record = {
                    "replication": replication,
                    "seed": seed,
                    "scenario": scenario,
                    "estimator": label,
                    "truth": truth,
                    "estimate": np.nan,
                    "se": np.nan,
                    "status": "error",
                    "error": "",
                }
                try:
                    if cache_key not in fit_cache:
                        fit_cache[cache_key] = fit_callable()
                    estimate, se, diagnostics = fit_cache[cache_key]
                    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
                        raise RuntimeError("Non-finite estimate or standard error.")
                    record.update(
                        estimate=estimate, se=se, status="ok", error=""
                    )
                    if label == "DRLPDID-IPT":
                        for horizon, values in diagnostics.items():
                            diagnostic_rows.append(
                                {
                                    "replication": replication,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "horizon": int(horizon),
                                    **values,
                                }
                            )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                raw_rows.append(record)

        if (replication + 1) % 10 == 0 or replication + 1 == REPLICATIONS:
            raw = pd.DataFrame(raw_rows)
            raw.to_csv(OUTPUT_DIR / "replication_level.csv", index=False)
            pd.DataFrame(diagnostic_rows).to_csv(
                OUTPUT_DIR / "ipt_rank_and_identity_audit.csv", index=False
            )
            print(
                f"Completed {replication + 1}/{REPLICATIONS}; "
                f"failures={(raw['status'] != 'ok').sum()}"
            )

    raw = pd.DataFrame(raw_rows)
    failures = raw.loc[raw["status"].ne("ok")].copy()
    failures.to_csv(OUTPUT_DIR / "failures.csv", index=False)
    if not failures.empty:
        raise RuntimeError(
            f"{len(failures)} fits failed. Inspect {OUTPUT_DIR / 'failures.csv'}; "
            "article summaries were not certified."
        )

    summary = summarize(raw)
    summary.to_csv(OUTPUT_DIR / "table_absorbing_N500.csv", index=False)
    config = {
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "N": N,
        "T": T,
        "replications": REPLICATIONS,
        "first_seed": FIRST_SEED,
        "horizons": [-1, H_POST],
        "post_window": [0, H_POST],
        "scenarios": SCENARIOS,
        "runtime_seconds": time.time() - started,
    }
    (OUTPUT_DIR / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nCertified outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
