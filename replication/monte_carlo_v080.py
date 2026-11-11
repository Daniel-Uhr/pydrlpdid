"""Reproducible Monte Carlo validation for pydrlpdid 0.8.0 release candidates.

The four scenarios have transparent nuisance truth:

* both_correct: correctly specified PS and OR bases;
* or_only: the OR includes the relevant nonlinear basis and the PS omits it;
* ps_only: the PS includes the relevant basis and the OR omits it;
* neither: both nuisance bases omit the relevant nonlinear basis.

All five estimators use the same horizon-specific stack and target the same
constant ATT (= TAU).  A placebo lead at h=-2 has population value zero.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, __version__


TAU = 2.0
TIMES = np.arange(0, 10)
METHODS = ("ra", "ipw", "ipt", "dr-ipw", "dr-ipt")
SCENARIOS = {
    "both_correct": {
        "selection_basis": "q", "outcome_basis": "q",
        "ps_covariates": ["x", "q"], "or_covariates": ["x", "q"],
    },
    "or_only": {
        "selection_basis": "r", "outcome_basis": "r",
        "ps_covariates": ["x", "q"], "or_covariates": ["x", "r"],
    },
    "ps_only": {
        "selection_basis": "q", "outcome_basis": "q",
        "ps_covariates": ["x", "q"], "or_covariates": ["x"],
    },
    "neither": {
        "selection_basis": "r", "outcome_basis": "r",
        "ps_covariates": ["x", "q"], "or_covariates": ["x"],
    },
}
DESIGN_CODES = {"absorbing": 10_000_000, "switching": 20_000_000}
SCENARIO_CODES = {
    "both_correct": 100_000,
    "or_only": 200_000,
    "ps_only": 300_000,
    "neither": 400_000,
}


def deterministic_seed(base_seed: int, design: str, scenario: str, rep: int) -> int:
    return int(base_seed + DESIGN_CODES[design] + SCENARIO_CODES[scenario] + rep)


def make_panel(
    rng: np.random.Generator, n_units: int, design: str, scenario: str
) -> pd.DataFrame:
    specification = SCENARIOS[scenario]
    x = rng.normal(size=n_units)
    q = x ** 2 - 1.0
    r = (q ** 2 - 2.0) / np.sqrt(56.0)
    basis = {"q": q, "r": r}
    alpha = rng.normal(size=n_units)
    n_periods = len(TIMES)
    d = np.zeros((n_units, n_periods), dtype=int)
    selection_index = -0.20 + 0.35 * basis[specification["selection_basis"]]
    event_probability = 1.0 / (1.0 + np.exp(-selection_index))
    event_unit = rng.uniform(size=n_units) < event_probability

    if design == "absorbing":
        d[event_unit, 4:] = 1
    elif design == "switching":
        # One certified sustained switch-in at t=4. Among non-event controls,
        # some switch at t=1 and form a stable-one comparison path by t=4;
        # the remainder are stable zero. Control type is independent of X.
        d[event_unit, 4:] = 1
        stable_one = (~event_unit) & (rng.uniform(size=n_units) < 0.30)
        d[stable_one, 1:] = 1
    else:
        raise ValueError(f"Unknown design: {design}")

    eps = rng.normal(scale=1.0, size=(n_units, n_periods))
    t_grid = TIMES[None, :]
    y0 = alpha[:, None] + 0.25 * t_grid + 0.20 * t_grid * x[:, None] + eps
    outcome_basis = basis[specification["outcome_basis"]]
    y0 = y0 + 0.70 * t_grid * outcome_basis[:, None]
    y = y0 + TAU * d

    if design == "absorbing":
        first = np.where(d.sum(axis=1) > 0, np.argmax(d == 1, axis=1), 0)
    else:
        first = np.zeros(n_units, dtype=int)

    frames = []
    for j, period in enumerate(TIMES):
        frames.append(pd.DataFrame({
            "id": np.arange(n_units),
            "time": int(period),
            "first_treat": first,
            "treatment": d[:, j],
            "x": x,
            "q": q,
            "r": r,
            "y": y[:, j],
        }))
    return pd.concat(frames, ignore_index=True)


def one_replication(task: tuple) -> list[dict]:
    design, scenario, rep, n_units, base_seed = task
    seed = deterministic_seed(base_seed, design, scenario, rep)
    data = make_panel(np.random.default_rng(seed), n_units, design, scenario)
    specification = SCENARIOS[scenario]
    rows: list[dict] = []
    for method in METHODS:
        started = time.perf_counter()
        record = {
            "design": design,
            "scenario": scenario,
            "replication": rep,
            "seed": seed,
            "method": method,
        }
        try:
            common = dict(
                estimation_method=method,
                horizons=(-2, 2),
                post_window=(0, 2),
                post_aggregation="mean_horizons",
                or_slope_structure="common",
                history_policy="observed",
                small_sample_correction="none",
                inference="cluster",
            )
            if design == "absorbing":
                estimator = DRLPDID(
                    design="absorbing", control_group="never_treated", **common
                )
                fit = estimator.fit(
                    data, outcome="y", unit="id", time="time",
                    first_treat="first_treat",
                    ps_covariates=specification["ps_covariates"],
                    or_covariates=specification["or_covariates"],
                )
            else:
                estimator = DRLPDID(
                    design="switching", stabilization_window=2, **common
                )
                fit = estimator.fit(
                    data, outcome="y", unit="id", time="time",
                    treatment="treatment",
                    ps_covariates=specification["ps_covariates"],
                    or_covariates=specification["or_covariates"],
                )
            event = fit.event_study.set_index("horizon")
            for horizon in (-2, 0, 1, 2):
                row = event.loc[horizon]
                truth = 0.0 if horizon < 0 else TAU
                rows.append({
                    **record,
                    "target": f"h={horizon}",
                    "truth": truth,
                    "estimate": float(row["estimate"]),
                    "se": float(row["se"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                    "failed": False,
                    "error_type": "",
                    "error_message": "",
                    "elapsed_seconds": time.perf_counter() - started,
                })
            scalar = fit.scalars.iloc[0]
            rows.append({
                **record,
                "target": "ATT[0,2]",
                "truth": TAU,
                "estimate": float(scalar["estimate"]),
                "se": float(scalar["se"]),
                "ci_lower": float(scalar["ci_lower"]),
                "ci_upper": float(scalar["ci_upper"]),
                "failed": False,
                "error_type": "",
                "error_message": "",
                "elapsed_seconds": time.perf_counter() - started,
            })
        except Exception as exc:  # failures are scientific outcomes, not hidden
            rows.append({
                **record,
                "target": "FIT",
                "truth": np.nan,
                "estimate": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "failed": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            })
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw.loc[~raw["failed"] & raw["estimate"].notna()].copy()
    valid["error"] = valid["estimate"] - valid["truth"]
    valid["squared_error"] = valid["error"] ** 2
    valid["covered"] = (
        (valid["ci_lower"] <= valid["truth"])
        & (valid["truth"] <= valid["ci_upper"])
    ).astype(float)
    valid["interval_length"] = valid["ci_upper"] - valid["ci_lower"]
    grouped = valid.groupby(
        ["design", "scenario", "method", "target"], sort=True
    )
    summary = grouped.agg(
        replications=("estimate", "size"),
        mean_estimate=("estimate", "mean"),
        bias=("error", "mean"),
        variance=("estimate", "var"),
        mse=("squared_error", "mean"),
        coverage=("covered", "mean"),
        mean_interval_length=("interval_length", "mean"),
        mean_se=("se", "mean"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"),
    ).reset_index()
    summary["rmse"] = np.sqrt(summary["mse"])

    attempts = raw.groupby(
        ["design", "scenario", "method"], sort=True
    ).agg(
        attempted_fits=("replication", "nunique"),
        failed_fits=("failed", "sum"),
    ).reset_index()
    attempts["failure_rate"] = attempts["failed_fits"] / attempts["attempted_fits"]
    return summary.merge(attempts, on=["design", "scenario", "method"], how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--units", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=8_031_977)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v080"))
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            # Windows readers can briefly hold a sharing lock on the target.
            time.sleep(0.25)


def main() -> None:
    args = parse_args()
    if (
        args.replications < 1 or args.units < 50 or args.jobs < 1
        or args.checkpoint_every < 1
    ):
        raise ValueError(
            "Use replications>=1, units>=50, jobs>=1, and checkpoint-every>=1."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = [
        (design, scenario, rep, args.units, args.base_seed)
        for design in DESIGN_CODES
        for scenario in SCENARIOS
        for rep in range(args.replications)
    ]
    checkpoint_path = args.output_dir / "monte_carlo_checkpoint.csv"
    completed_rows: list[dict] = []
    completed_keys: set[tuple[str, str, int]] = set()
    if args.resume and checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path)
        completed_rows = previous.to_dict("records")
        method_counts = previous.groupby(
            ["design", "scenario", "replication"]
        )["method"].nunique()
        completed_keys = {
            (str(design), str(scenario), int(rep))
            for (design, scenario, rep), count in method_counts.items()
            if int(count) == len(METHODS)
        }
        completed_rows = [
            row for row in completed_rows
            if (str(row["design"]), str(row["scenario"]), int(row["replication"]))
            in completed_keys
        ]
    tasks = [
        task for task in all_tasks
        if (task[0], task[1], int(task[2])) not in completed_keys
    ]
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if args.jobs == 1:
            iterator = map(one_replication, tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=args.jobs)
            iterator = executor.map(one_replication, tasks, chunksize=1)
        try:
            for completed, result_rows in enumerate(iterator, start=1):
                completed_rows.extend(result_rows)
                if completed % args.checkpoint_every == 0:
                    atomic_csv(pd.DataFrame(completed_rows), checkpoint_path)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
        if tasks:
            atomic_csv(pd.DataFrame(completed_rows), checkpoint_path)
    raw = pd.DataFrame(completed_rows)
    summary = summarize(raw)
    failures = raw.loc[raw["failed"]].copy()

    atomic_csv(raw, args.output_dir / "monte_carlo_raw.csv")
    atomic_csv(summary, args.output_dir / "monte_carlo_summary.csv")
    atomic_csv(failures, args.output_dir / "monte_carlo_failures.csv")
    configuration = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "replications": args.replications,
        "units": args.units,
        "jobs": args.jobs,
        "base_seed": args.base_seed,
        "checkpoint_every": args.checkpoint_every,
        "resumed_completed_tasks": len(completed_keys),
        "tau": TAU,
        "designs": list(DESIGN_CODES),
        "scenarios": SCENARIOS,
        "methods": list(METHODS),
        "last_run_elapsed_seconds": time.perf_counter() - started,
        "captured_parent_warnings": [str(w.message) for w in caught],
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSaved results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
