"""Pilot comparison on the no-fault-divorce (Bacon) panel.

The script separates two checks:

1. implementation parity between pydrlpdid and diff-diff LP-DiD; and
2. a common fixed-composition post-treatment target over h=0,...,19.

It deliberately does not compare package-native overall ATTs that use
different aggregation rules.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff import CallawaySantAnna, LPDiD as DiffDiffLPDiD
from pydrlpdid import DRLPDID
from pydrlpdid.lpdid import _LPDIDEngine


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bacon_example.dta"
OUTPUT_DIR = ROOT / "results" / "bacon_common_att_pilot"
H_PRE = 10
H_POST = 19
ALPHA = 0.05
SEED = 123


def _prepare_data() -> tuple[pd.DataFrame, dict]:
    data = pd.read_stata(DATA_PATH)
    unit_status = data.groupby("stfips", sort=True)["post"].agg(["min", "max"])
    always_ids = unit_status.index[
        unit_status["min"].eq(1) & unit_status["max"].eq(1)
    ]
    never_ids = unit_status.index[
        unit_status["min"].eq(0) & unit_status["max"].eq(0)
    ]

    estimation = data.loc[~data["stfips"].isin(always_ids)].copy()
    estimation["id"] = estimation["stfips"].astype(int)
    estimation["year"] = estimation["year"].astype(int)
    estimation["y"] = estimation["asmrs"].astype(float)
    estimation["g"] = estimation["_nfd"].fillna(0).astype(int)
    estimation["treated"] = estimation["post"].astype(int)

    unit = estimation.groupby("id", sort=True).agg(
        g=("g", "first"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    switchers = unit.loc[unit["g"].gt(0)].copy()
    switchers["full_h19"] = (
        switchers["g"].gt(switchers["first_year"])
        & switchers["g"].add(H_POST).le(switchers["last_year"])
    )
    target = switchers.loc[switchers["full_h19"]]
    late = switchers.loc[~switchers["full_h19"]]

    support = {
        "source_rows": int(len(data)),
        "source_units": int(data["stfips"].nunique()),
        "always_treated_excluded": int(len(always_ids)),
        "never_treated_units": int(len(never_ids)),
        "switcher_units": int(len(switchers)),
        "full_support_target_units_h0_h19": int(len(target)),
        "full_support_target_cohorts": [int(x) for x in sorted(target["g"].unique())],
        "late_cohorts_not_in_target": [int(x) for x in sorted(late["g"].unique())],
        "estimation_units": int(estimation["id"].nunique()),
        "horizon_window": [0, H_POST],
    }
    return estimation, support


def _summary_row(
    estimator: str,
    estimate: float,
    se: float,
    df: float,
    aggregation: str,
    implementation: str,
) -> dict:
    critical = float(stats.t.ppf(1 - ALPHA / 2, df))
    t_stat = float(estimate / se)
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    return {
        "estimator": estimator,
        "implementation": implementation,
        "target": "fixed-composition ATT, h=0,...,19",
        "aggregation": aggregation,
        "estimate": float(estimate),
        "se": float(se),
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": float(estimate - critical * se),
        "ci_upper": float(estimate + critical * se),
        "df": float(df),
    }


def _path(frame: pd.DataFrame, estimator: str, estimate_column: str) -> pd.DataFrame:
    result = frame.loc[
        frame["horizon"].between(-H_PRE, H_POST),
        ["horizon", estimate_column],
    ].copy()
    result.rename(columns={estimate_column: "estimate"}, inplace=True)
    result["estimator"] = estimator
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, support = _prepare_data()
    summaries: list[dict] = []
    paths: list[pd.DataFrame] = []

    pyd_lpdid = _LPDIDEngine(
        target_estimand="ra",
        max_pre=H_PRE,
        max_post=H_POST,
        fixed_composition=True,
        small_sample_correction="none",
    ).fit(
        data,
        outcome="y",
        unit="id",
        time="year",
        first_treat="g",
        covariates=["asmrh"],
    )
    pyd_scalar = pyd_lpdid.scalars.loc[
        pyd_lpdid.scalars["term"].eq("ATT avg")
    ].iloc[0]
    summaries.append(
        _summary_row(
            "LPDID-RA",
            pyd_scalar["estimate"],
            pyd_scalar["se"],
            pyd_scalar["df"],
            "equal horizon; fixed treated composition",
            "pydrlpdid 0.8.0rc3",
        )
    )
    paths.append(_path(pyd_lpdid.event_study, "LPDID-RA (pydrlpdid)", "estimate"))

    dd_lpdid = DiffDiffLPDiD(
        pre_window=H_PRE,
        post_window=H_POST,
        control_group="clean",
        reweight=True,
        no_composition=True,
        cluster="id",
        df_convention="cluster",
    ).fit(
        data,
        outcome="y",
        unit="id",
        time="year",
        treatment="treated",
        covariates=["asmrh"],
    )
    dd_scalar = dd_lpdid.pooled.loc[dd_lpdid.pooled["window"].eq("post")].iloc[0]
    summaries.append(
        _summary_row(
            "LPDID-RA",
            dd_scalar["coefficient"],
            dd_scalar["se"],
            int(dd_scalar["n_clusters"]) - 1,
            "equal horizon; fixed treated composition",
            "diff-diff 3.12.0",
        )
    )
    paths.append(_path(dd_lpdid.event_study, "LPDID-RA (diff-diff)", "coefficient"))

    event_parity = pyd_lpdid.event_study[
        ["horizon", "estimate", "se"]
    ].merge(
        dd_lpdid.event_study[["horizon", "coefficient", "se"]],
        on="horizon",
        suffixes=("_pydrlpdid", "_diff_diff"),
        validate="one_to_one",
    )
    post_event_parity = event_parity.loc[event_parity["horizon"].ge(0)]
    pre_event_parity = event_parity.loc[event_parity["horizon"].lt(-1)]
    parity = {
        "estimate_abs_difference": float(
            abs(pyd_scalar["estimate"] - dd_scalar["coefficient"])
        ),
        "se_abs_difference": float(abs(pyd_scalar["se"] - dd_scalar["se"])),
        "post_event_estimate_max_abs_difference": float(
            (
                post_event_parity["estimate"]
                - post_event_parity["coefficient"]
            ).abs().max()
        ),
        "post_event_se_max_abs_difference": float(
            (
                post_event_parity["se_pydrlpdid"]
                - post_event_parity["se_diff_diff"]
            ).abs().max()
        ),
        "pre_event_estimate_max_abs_difference": float(
            (
                pre_event_parity["estimate"]
                - pre_event_parity["coefficient"]
            ).abs().max()
        ),
        "pre_event_se_max_abs_difference": float(
            (
                pre_event_parity["se_pydrlpdid"]
                - pre_event_parity["se_diff_diff"]
            ).abs().max()
        ),
    }
    certified_keys = [
        "estimate_abs_difference",
        "se_abs_difference",
        "post_event_estimate_max_abs_difference",
        "post_event_se_max_abs_difference",
    ]
    parity["post_and_scalar_passed_1e_8"] = bool(
        max(parity[key] for key in certified_keys) <= 1e-8
    )
    if not parity["post_and_scalar_passed_1e_8"]:
        raise RuntimeError(f"LP-DiD parity gate failed: {parity}")

    with warnings.catch_warnings(record=True) as cs_warnings:
        warnings.simplefilter("always")
        cs_base = CallawaySantAnna(
            control_group="not_yet_treated",
            base_period="universal",
            estimation_method="dr",
            cluster="id",
            n_bootstrap=0,
            cband=False,
            seed=SEED,
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="year",
            first_treat="g",
            covariates=["asmrh"],
        )
        cs_event = cs_base.aggregate("event_study", balance_e=H_POST)

    cs_frame = cs_event.to_dataframe().rename(
        columns={"event_time": "horizon", "att": "estimate"}
    )
    post = cs_frame.loc[cs_frame["horizon"].between(0, H_POST)].copy()
    if len(post) != H_POST + 1 or post["n"].nunique() != 1:
        raise RuntimeError("Callaway-Sant'Anna fixed-composition support is incomplete.")
    vcov_index = list(cs_event.vcov_index)
    positions = [vcov_index.index(h) for h in range(H_POST + 1)]
    vcov = np.asarray(cs_event.vcov, dtype=float)[np.ix_(positions, positions)]
    horizon_weights = np.full(H_POST + 1, 1.0 / (H_POST + 1))
    cs_estimate = float(horizon_weights @ post["estimate"].to_numpy(dtype=float))
    cs_se = float(np.sqrt(horizon_weights @ vcov @ horizon_weights))
    cs_df = float(np.asarray(cs_event.df).reshape(-1)[0])
    summaries.append(
        _summary_row(
            "Callaway-Sant'Anna DR",
            cs_estimate,
            cs_se,
            cs_df,
            "equal horizon; balance_e=19",
            "diff-diff 3.12.0",
        )
    )
    paths.append(
        cs_frame.loc[
            cs_frame["horizon"].between(-H_PRE, H_POST),
            ["horizon", "estimate"],
        ].assign(estimator="Callaway-Sant'Anna DR")
    )

    epv = cs_base.epv_summary()
    cs_diagnostics = {
        "warning_count": len(cs_warnings),
        "warning_messages": sorted({str(item.message) for item in cs_warnings}),
        "epv_cells": int(len(epv)),
        "epv_low_cells": int(epv["is_low"].sum()),
        "epv_min": float(epv["epv"].min()),
        "epv_median": float(epv["epv"].median()),
        "fixed_composition_group_count": int(post["n"].iloc[0]),
        "native_overall_att_not_used": float(cs_base.overall_att),
    }

    # Regression-only CS sensitivity: same target and covariate information,
    # but no propensity-score fit in the very small adoption cohorts.
    cs_reg_base = CallawaySantAnna(
        control_group="not_yet_treated",
        base_period="universal",
        estimation_method="reg",
        cluster="id",
        n_bootstrap=0,
        cband=False,
        seed=SEED,
    ).fit(
        data,
        outcome="y",
        unit="id",
        time="year",
        first_treat="g",
        covariates=["asmrh"],
    )
    cs_reg_event = cs_reg_base.aggregate("event_study", balance_e=H_POST)
    cs_reg_frame = cs_reg_event.to_dataframe().rename(
        columns={"event_time": "horizon", "att": "estimate"}
    )
    cs_reg_post = cs_reg_frame.loc[
        cs_reg_frame["horizon"].between(0, H_POST)
    ].copy()
    cs_reg_index = list(cs_reg_event.vcov_index)
    cs_reg_positions = [cs_reg_index.index(h) for h in range(H_POST + 1)]
    cs_reg_vcov = np.asarray(cs_reg_event.vcov, dtype=float)[
        np.ix_(cs_reg_positions, cs_reg_positions)
    ]
    cs_reg_estimate = float(
        horizon_weights @ cs_reg_post["estimate"].to_numpy(dtype=float)
    )
    cs_reg_se = float(np.sqrt(horizon_weights @ cs_reg_vcov @ horizon_weights))
    cs_reg_df = float(np.asarray(cs_reg_event.df).reshape(-1)[0])
    summaries.append(
        _summary_row(
            "Callaway-Sant'Anna REG",
            cs_reg_estimate,
            cs_reg_se,
            cs_reg_df,
            "equal horizon; balance_e=19",
            "diff-diff 3.12.0",
        )
    )
    paths.append(
        cs_reg_frame.loc[
            cs_reg_frame["horizon"].between(-H_PRE, H_POST),
            ["horizon", "estimate"],
        ].assign(estimator="Callaway-Sant'Anna REG")
    )

    with warnings.catch_warnings(record=True) as dr_warnings:
        warnings.simplefilter("always")
        dr_result = DRLPDID(
            estimation_method="dr-ipt",
            pre_window=H_PRE,
            post_window=H_POST,
            control_group="clean",
            cluster="id",
            post_aggregation="pooled",
            inference="cluster",
            small_sample_correction="cr1",
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="year",
            first_treat="g",
            covariates=["asmrh"],
        )
    dr_scalar = dr_result.pooled.loc[dr_result.pooled["window"].eq("post")].iloc[0]
    summaries.append(
        _summary_row(
            "DRLPDID-IPT",
            dr_scalar["estimate"],
            dr_scalar["se"],
            dr_scalar["df"],
            "pooled common post window",
            "pydrlpdid 0.8.0rc3",
        )
    )
    paths.append(_path(dr_result.event_study, "DRLPDID-IPT", "estimate"))

    summary = pd.DataFrame(summaries)
    event_paths = pd.concat(paths, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "common_att_summary.csv", index=False)
    event_paths.to_csv(OUTPUT_DIR / "event_study_paths.csv", index=False)
    pd.DataFrame([support]).to_json(
        OUTPUT_DIR / "support_summary.json", orient="records", indent=2
    )
    with (OUTPUT_DIR / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "environment": {
                    "python": platform.python_version(),
                    "pydrlpdid": importlib.metadata.version("pydrlpdid"),
                    "diff-diff": importlib.metadata.version("diff-diff"),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
                "lpdid_parity": parity,
                "callaway_santanna": cs_diagnostics,
                "drlpdid_warning_messages": sorted(
                    {str(item.message) for item in dr_warnings}
                ),
            },
            handle,
            indent=2,
        )

    print(summary.to_string(index=False))
    print("\nLP-DiD parity:", parity)
    print("Support:", support)
    print(
        "CS overlap diagnostics:",
        {key: value for key, value in cs_diagnostics.items() if key != "warning_messages"},
    )
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
