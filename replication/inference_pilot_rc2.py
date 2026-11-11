"""Focused inference pilot for the pydrlpdid 0.8.0rc2 release gate."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, __version__

from monte_carlo_v080 import SCENARIOS, TAU, make_panel


METHODS = ("ra", "ipw", "ipt", "dr-ipw", "dr-ipt")
DESIGNS = ("absorbing", "first_entry", "effect_stabilization")
INFERENCE_SPECS = (
    ("cluster", "none", None),
    ("cluster_cr1", "cr1", None),
    ("multiplier_rademacher", "none", "rademacher"),
    ("multiplier_mammen", "none", "mammen"),
    ("multiplier_webb", "none", "webb"),
    ("cluster_bootstrap", "none", None),
)
DESIGN_CODES = {
    "absorbing": 1_000_000,
    "first_entry": 2_000_000,
    "effect_stabilization": 3_000_000,
}
METHOD_CODES = {method: 10_000 * (index + 1) for index, method in enumerate(METHODS)}
INFERENCE_CODES = {
    name: 100 * (index + 1) for index, (name, _, _) in enumerate(INFERENCE_SPECS)
}


def fit_once(
    data: pd.DataFrame,
    design: str,
    method: str,
    inference_name: str,
    correction: str,
    multiplier_weights: Optional[str],
    bootstrap_seed: int,
    n_multiplier: int,
    n_cluster_bootstrap: int,
    min_bootstrap_success: float,
):
    if inference_name.startswith("multiplier"):
        inference = "multiplier"
        n_bootstrap = n_multiplier
    elif inference_name == "cluster_bootstrap":
        inference = "cluster_bootstrap"
        n_bootstrap = n_cluster_bootstrap
    else:
        inference = "cluster"
        n_bootstrap = n_multiplier

    kwargs = dict(
        estimation_method=method,
        pre_window=2,
        post_window=2,
        control_group="clean",
        cluster="id",
        post_aggregation="pooled",
        inference=inference,
        bootstrap_weights=multiplier_weights or "rademacher",
        n_bootstrap=n_bootstrap,
        min_bootstrap_success=min_bootstrap_success,
        small_sample_correction=correction,
        seed=bootstrap_seed,
    )
    if design == "first_entry":
        kwargs["non_absorbing"] = "first_entry"
    elif design == "effect_stabilization":
        kwargs["non_absorbing"] = "effect_stabilization"
        kwargs["stabilization_window"] = 2

    estimator = DRLPDID(**kwargs)
    specification = SCENARIOS["both_correct"]
    return estimator.fit(
        data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        ps_covariates=specification["ps_covariates"],
        or_covariates=specification["or_covariates"],
    )


def extract_rows(result, design, method, inference_name, replication, elapsed):
    rows = []
    event = result.event_study.set_index("horizon")
    for horizon, truth in ((-2, 0.0), (0, TAU), (1, TAU), (2, TAU)):
        estimate = float(event.loc[horizon, "estimate"])
        if inference_name.startswith("multiplier"):
            lower = float(event.loc[horizon, "multiplier_ci_lower"])
            upper = float(event.loc[horizon, "multiplier_ci_upper"])
            p_value = float(event.loc[horizon, "multiplier_p_value"])
        else:
            lower = float(event.loc[horizon, "ci_lower"])
            upper = float(event.loc[horizon, "ci_upper"])
            p_value = float(event.loc[horizon, "p_value"])
        rows.append({
            "design": design,
            "method": method,
            "inference": inference_name,
            "replication": replication,
            "target": f"h={horizon}",
            "truth": truth,
            "estimate": estimate,
            "se": float(event.loc[horizon, "se"]),
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_value,
            "covered": bool(lower <= truth <= upper),
            "interval_length": upper - lower,
            "rejected": bool(p_value < 0.05),
            "simultaneous_covered": np.nan,
            "elapsed_seconds": elapsed,
            "bootstrap_success_rate": result.metadata.get(
                "cluster_bootstrap", {}
            ).get("success_rate", np.nan),
            "failed": False,
            "error_type": "",
            "error_message": "",
        })

    scalar = result.pooled.loc[result.pooled["window"].eq("post")].iloc[0]
    if inference_name.startswith("multiplier"):
        lower = float(scalar["multiplier_ci_lower"])
        upper = float(scalar["multiplier_ci_upper"])
        p_value = float(scalar["multiplier_p_value"])
    else:
        lower = float(scalar["ci_lower"])
        upper = float(scalar["ci_upper"])
        p_value = float(scalar["p_value"])
    rows.append({
        "design": design,
        "method": method,
        "inference": inference_name,
        "replication": replication,
        "target": "ATT_post",
        "truth": TAU,
        "estimate": float(scalar["estimate"]),
        "se": float(scalar["se"]),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "covered": bool(lower <= TAU <= upper),
        "interval_length": upper - lower,
        "rejected": bool(p_value < 0.05),
        "simultaneous_covered": np.nan,
        "elapsed_seconds": elapsed,
        "bootstrap_success_rate": result.metadata.get(
            "cluster_bootstrap", {}
        ).get("success_rate", np.nan),
        "failed": False,
        "error_type": "",
        "error_message": "",
    })

    if inference_name.startswith("multiplier"):
        truths = {-2: 0.0, 0: TAU, 1: TAU, 2: TAU}
        simultaneous_covered = all(
            float(event.loc[h, "sim_ci_lower"]) <= truth
            <= float(event.loc[h, "sim_ci_upper"])
            for h, truth in truths.items()
        )
        for row in rows:
            row["simultaneous_covered"] = bool(simultaneous_covered)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    successful = raw.loc[~raw["failed"]].copy()
    if successful.empty:
        return pd.DataFrame()
    successful["bias"] = successful["estimate"] - successful["truth"]
    summary = successful.groupby(
        ["design", "method", "inference", "target"], sort=True
    ).agg(
        replications=("estimate", "size"),
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        coverage=("covered", "mean"),
        rejection_rate=("rejected", "mean"),
        mean_interval_length=("interval_length", "mean"),
        simultaneous_coverage=("simultaneous_covered", "mean"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"),
        mean_bootstrap_success=("bootstrap_success_rate", "mean"),
    ).reset_index()
    attempts = raw.groupby(
        ["design", "method", "inference"], sort=True
    ).agg(
        attempted_rows=("failed", "size"),
        failed_rows=("failed", "sum"),
    ).reset_index()
    return summary.merge(attempts, on=["design", "method", "inference"], how="left")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--units", type=int, default=120)
    parser.add_argument("--multiplier-draws", type=int, default=99)
    parser.add_argument("--cluster-bootstrap-draws", type=int, default=5)
    parser.add_argument("--min-bootstrap-success", type=float, default=0.60)
    parser.add_argument("--base-seed", type=int, default=90210)
    parser.add_argument(
        "--methods", default=",".join(METHODS),
        help="Comma-separated subset of estimation methods.",
    )
    parser.add_argument(
        "--designs", default=",".join(DESIGNS),
        help="Comma-separated subset of treatment designs.",
    )
    parser.add_argument(
        "--inferences",
        default=",".join(name for name, _, _ in INFERENCE_SPECS),
        help="Comma-separated subset of inference specifications.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/inference_pilot_rc2")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replications < 1 or args.units < 40:
        raise ValueError("Use replications>=1 and units>=40.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    selected_designs = tuple(item.strip() for item in args.designs.split(",") if item.strip())
    selected_inferences = tuple(
        item.strip() for item in args.inferences.split(",") if item.strip()
    )
    unknown_methods = sorted(set(selected_methods) - set(METHODS))
    unknown_designs = sorted(set(selected_designs) - set(DESIGNS))
    known_inferences = {name for name, _, _ in INFERENCE_SPECS}
    unknown_inferences = sorted(set(selected_inferences) - known_inferences)
    if unknown_methods or unknown_designs or unknown_inferences:
        raise ValueError(
            "Unknown selections: "
            f"methods={unknown_methods}, designs={unknown_designs}, "
            f"inferences={unknown_inferences}."
        )
    inference_specs = tuple(
        spec for spec in INFERENCE_SPECS if spec[0] in selected_inferences
    )
    rows = []
    started_all = time.perf_counter()
    for replication in range(args.replications):
        for design in selected_designs:
            dgp_design = "absorbing" if design == "absorbing" else "switching"
            data_seed = args.base_seed + DESIGN_CODES[design] + replication
            data = make_panel(
                np.random.default_rng(data_seed), args.units, dgp_design, "both_correct"
            )
            for method in selected_methods:
                for inference_name, correction, weights in inference_specs:
                    bootstrap_seed = (
                        args.base_seed
                        + DESIGN_CODES[design]
                        + METHOD_CODES[method]
                        + INFERENCE_CODES[inference_name]
                        + replication
                    )
                    started = time.perf_counter()
                    try:
                        result = fit_once(
                            data, design, method, inference_name, correction,
                            weights, bootstrap_seed, args.multiplier_draws,
                            args.cluster_bootstrap_draws,
                            args.min_bootstrap_success,
                        )
                        elapsed = time.perf_counter() - started
                        rows.extend(extract_rows(
                            result, design, method, inference_name,
                            replication, elapsed,
                        ))
                    except Exception as exc:
                        elapsed = time.perf_counter() - started
                        rows.append({
                            "design": design,
                            "method": method,
                            "inference": inference_name,
                            "replication": replication,
                            "target": "fit",
                            "truth": np.nan,
                            "estimate": np.nan,
                            "se": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "p_value": np.nan,
                            "covered": False,
                            "interval_length": np.nan,
                            "rejected": False,
                            "simultaneous_covered": np.nan,
                            "elapsed_seconds": elapsed,
                            "bootstrap_success_rate": np.nan,
                            "failed": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        })
        pd.DataFrame(rows).to_csv(args.output_dir / "raw.csv", index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "raw.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    failures = raw.loc[raw["failed"], [
        "design", "method", "inference", "replication",
        "error_type", "error_message",
    ]]
    failures.to_csv(args.output_dir / "failures.csv", index=False)
    manifest = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "replications": args.replications,
        "units": args.units,
        "multiplier_draws": args.multiplier_draws,
        "cluster_bootstrap_draws": args.cluster_bootstrap_draws,
        "min_bootstrap_success": args.min_bootstrap_success,
        "base_seed": args.base_seed,
        "methods": selected_methods,
        "designs": selected_designs,
        "inference_specs": inference_specs,
        "elapsed_seconds": time.perf_counter() - started_all,
        "failed_fits": int(raw["failed"].sum()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if not failures.empty:
        print(failures.to_string(index=False))


if __name__ == "__main__":
    main()

