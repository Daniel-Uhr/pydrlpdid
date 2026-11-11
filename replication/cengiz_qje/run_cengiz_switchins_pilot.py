"""Cengiz et al. (2019) county-level switch-in pilot.

The script uses the official QJE replication files and keeps counties as the
physical panel units while clustering influence functions at the state level.
It deliberately separates two binary treatments:

1. ``event_spell``: the documented four-quarter window following a prominent
   state minimum-wage increase (post horizons are therefore limited to 0--3);
2. ``above_federal``: a state's minimum wage exceeds the federal floor, which
   defines economically interpretable recurrent sustained switch-ins.

Neither target is described as a reproduction of the paper's continuous
log-minimum-wage regressions in Appendix Table G8.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pydrlpdid import DRLPDID  # noqa: E402
from pydrlpdid.lpdid import _LPDIDEngine  # noqa: E402
from pydrlpdid._inference import se_from_influence  # noqa: E402


DATA = Path(
    os.environ.get(
        "PYDRLPDID_CENGIZ_DATA_DIR",
        REPO / "data" / "cengiz_qje",
    )
).expanduser().resolve()
OUTPUT = REPO / "results" / "cengiz_qje"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analysis_panel() -> tuple[pd.DataFrame, dict]:
    qcew_path = DATA / "QCEWindustry_minwage_all.dta"
    state_path = DATA / "quarterly_state_panel.dta"
    qcew = pd.read_stata(qcew_path, convert_categoricals=False)
    state = pd.read_stata(state_path, convert_categoricals=False)

    # Stata quarterly dates count from 1960q1=0.
    qcew["quarterdate"] = (
        (qcew["year"].astype(int) - 1960) * 4
        + qcew["quarter"].astype(int) - 1
    )
    state_vars = [
        "statenum", "quarterdate", "treat", "DMW_real", "fedincrease",
        "overallcountgroup", "cleansample",
    ]
    panel = qcew.merge(
        state[state_vars],
        left_on=["state_fips", "quarterdate"],
        right_on=["statenum", "quarterdate"],
        how="left",
        validate="many_to_one",
    )

    # Appendix Table G8's full-county restaurant sample.
    panel = panel.loc[panel["nonmissing_rest_both"].eq(66)].copy()
    panel["event_spell"] = panel["treat"].fillna(0).gt(0).astype(int)
    panel["above_federal"] = (
        panel["stminwage"] > panel["federalmin"] + 1e-6
    ).astype(int)
    panel["county_id"] = panel["countyreal"].astype(int)
    panel["state_id"] = panel["state_fips"].astype(int)
    panel = panel.sort_values(["county_id", "quarterdate"])
    panel["lnpop_l1"] = panel.groupby("county_id", sort=False)["lnpop"].shift(1)
    panel = panel.dropna(
        subset=[
            "lnemp_rest_both", "lnpop_l1", "county_id", "state_id",
            "quarterdate",
        ]
    ).copy()

    mapping_count = panel.groupby("county_id")["state_id"].nunique()
    if not mapping_count.eq(1).all():
        raise RuntimeError("A county maps to more than one state.")
    if panel.duplicated(["county_id", "quarterdate"]).any():
        raise RuntimeError("Duplicate county-quarter observations.")

    diagnostics = {
        "qcew_sha256": sha256(qcew_path),
        "state_panel_sha256": sha256(state_path),
        "n_rows": int(len(panel)),
        "n_counties": int(panel["county_id"].nunique()),
        "n_states": int(panel["state_id"].nunique()),
        "first_quarterdate": int(panel["quarterdate"].min()),
        "last_quarterdate": int(panel["quarterdate"].max()),
    }
    for treatment in ("event_spell", "above_federal"):
        previous = panel.groupby("county_id", sort=False)[treatment].shift(1)
        onset = panel[treatment].eq(1) & previous.eq(0)
        diagnostics[f"{treatment}_state_onsets"] = int(
            panel.loc[onset, ["state_id", "quarterdate"]].drop_duplicates().shape[0]
        )
        diagnostics[f"{treatment}_left_censored_states"] = int(
            panel.groupby("state_id", sort=False)[treatment].first().eq(1).sum()
        )
    return panel, diagnostics


def model_specs(treatment: str, post: int):
    common_lp = dict(
        max_pre=4,
        max_post=post,
        clean_control="stabilized",
        nonabsorbing=True,
        effect_stabilization=4,
        include_lagged_outcome_change=True,
        n_lagged_outcome_changes=1,
        inference="cluster",
        cluster="state_id",
        left_censoring="keep_as_control_after_l",
        small_sample_correction="cr1",
    )
    common_dr = dict(
        pre_window=4,
        post_window=post,
        control_group="clean",
        non_absorbing="effect_stabilization",
        stabilization_window=4,
        outcome_change_lags=1,
        inference="cluster",
        cluster="state_id",
        left_censoring="keep_as_control_after_l",
        small_sample_correction="cr1",
    )
    return {
        "LPDID-RA": _LPDIDEngine(target_estimand="ra", **common_lp),
        "DRLPDID-DRIPW": DRLPDID(estimation_method="dr-ipw", **common_dr),
        "DRLPDID-DRIPT": DRLPDID(estimation_method="dr-ipt", **common_dr),
    }


def run_design(panel: pd.DataFrame, treatment: str, post: int):
    event_frames = []
    scalar_frames = []
    fit_kwargs = dict(
        outcome="lnemp_rest_both",
        unit="county_id",
        time="quarterdate",
        treatment=treatment,
        covariates=["lnpop_l1"],
    )
    for label, estimator in model_specs(treatment, post).items():
        print(f"Fitting {treatment}: {label}", flush=True)
        result = estimator.fit(panel, **fit_kwargs)
        event = result.event_study.copy()
        event.insert(0, "estimator", label)
        event.insert(0, "treatment_design", treatment)
        event_frames.append(event)
        scalars = result.scalars.copy()
        scalars.insert(0, "estimator", label)
        scalars.insert(0, "treatment_design", treatment)
        scalar_frames.append(scalars)
    return pd.concat(event_frames, ignore_index=True), pd.concat(
        scalar_frames, ignore_index=True
    )


def validate_state_influence_aggregation(panel: pd.DataFrame) -> dict:
    """Application-specific equality: direct state IF = sum of county IFs."""
    common = dict(
        target_estimand="ra", max_pre=2, max_post=1,
        clean_control="stabilized", nonabsorbing=True,
        effect_stabilization=4, include_lagged_outcome_change=True,
        n_lagged_outcome_changes=1, inference="cluster",
        left_censoring="keep_as_control_after_l",
        small_sample_correction="none",
    )
    fit_args = (
        panel, "lnemp_rest_both", "county_id", "quarterdate",
        None, "event_spell", ["lnpop_l1"],
    )
    county_core = _LPDIDEngine(**common)._fit_core(*fit_args)
    state_core = _LPDIDEngine(**common, cluster="state_id")._fit_core(*fit_args)
    county_to_state = (
        panel[["county_id", "state_id"]]
        .drop_duplicates()
        .set_index("county_id")["state_id"]
    )
    comparisons = {}
    for horizon in (-2, 0, 1):
        county_psi = county_core["psi_by_h"][horizon]
        manual = county_psi.groupby(
            county_to_state.reindex(county_psi.index).to_numpy()
        ).sum()
        direct = state_core["psi_by_h"][horizon]
        aligned = manual.reindex(direct.index)
        comparisons[str(horizon)] = {
            "max_abs_if_difference": float(
                np.max(np.abs(direct.to_numpy() - aligned.to_numpy()))
            ),
            "direct_state_se": float(se_from_influence(direct.to_numpy())),
            "manual_state_se": float(se_from_influence(aligned.to_numpy())),
        }
    return comparisons


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, diagnostics = load_analysis_panel()
    print(json.dumps(diagnostics, indent=2), flush=True)

    influence_check = validate_state_influence_aggregation(panel)
    events = []
    scalars = []
    for treatment, post in (("event_spell", 3), ("above_federal", 8)):
        event, scalar = run_design(panel, treatment, post)
        events.append(event)
        scalars.append(scalar)

    event_results = pd.concat(events, ignore_index=True)
    scalar_results = pd.concat(scalars, ignore_index=True)
    event_results.to_csv(OUTPUT / "event_study.csv", index=False)
    scalar_results.to_csv(OUTPUT / "scalar_results.csv", index=False)
    with (OUTPUT / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "diagnostics": diagnostics,
                "influence_aggregation_validation": influence_check,
                "interpretation": {
                    "event_spell": "Entry into the paper's four-quarter prominent-increase window; post h=0..3 only.",
                    "above_federal": "Entry into a sustained state minimum-wage regime above the federal floor.",
                    "outcome": "Log restaurant employment (lnemp_rest_both).",
                    "cluster": "State; county influence functions summed within state.",
                },
            },
            handle,
            indent=2,
        )
    print(event_results.to_string(index=False), flush=True)
    print(scalar_results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
