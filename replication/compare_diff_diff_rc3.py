"""External numerical equivalence gate against diff-diff LPDiD 3.12.0.

The comparison uses only the two public small-sample choices.  diff-diff's
native VW/RW paths use CR1, whereas its influence-function RA path uses the
uncorrected sandwich; pydrlpdid is configured accordingly for replication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diff_diff import LPDiD as DiffDiffLPDiD
from pydrlpdid import LPDID, __version__

from monte_carlo_v080 import make_panel


def _max_abs(left, right) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    finite = np.isfinite(delta)
    return float(np.max(np.abs(delta[finite]))) if finite.any() else 0.0


def _one_comparison(design: str, estimand: str) -> dict:
    rng = np.random.default_rng(20261110)
    panel_design = "absorbing" if design == "absorbing" else "switching"
    data = make_panel(rng, 240, panel_design, "both_correct")
    covariates = ["x", "q"] if estimand == "ra" else None
    reweight = estimand in {"rw", "ra"}
    correction = "none" if estimand == "ra" else "cr1"

    pyd_kwargs = dict(
        target_estimand=estimand,
        max_pre=2,
        max_post=2,
        history_policy="diff_diff",
        small_sample_correction=correction,
    )
    dd_kwargs = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=reweight,
        cluster="id",
        df_convention="cluster",
    )
    fit_kwargs = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=covariates,
    )

    if design == "absorbing":
        pyd_kwargs["clean_control"] = "not_yet_treated"
    elif design == "first_entry":
        pyd_kwargs["clean_control"] = "first_entry"
        dd_kwargs["non_absorbing"] = "first_entry"
    elif design == "effect_stabilization":
        pyd_kwargs.update(
            clean_control="stabilized", effect_stabilization=2
        )
        dd_kwargs.update(
            non_absorbing="effect_stabilization", stabilization_window=2
        )
    else:  # pragma: no cover - guarded by the fixed design matrix below
        raise ValueError(design)

    pyd = LPDID(**pyd_kwargs).fit(**fit_kwargs)
    dd = DiffDiffLPDiD(**dd_kwargs).fit(**fit_kwargs)

    event = pyd.event_study.merge(dd.event_study, on="horizon", suffixes=("_pyd", "_dd"))
    pooled_pyd = pyd.scalars.loc[pyd.scalars["term"].eq("ATT pooled")].iloc[0]
    pooled_dd = dd.pooled.loc[dd.pooled["window"].eq("post")].iloc[0]

    row = {
        "pydrlpdid_version": __version__,
        "design": design,
        "estimand": estimand,
        "small_sample_correction": correction,
        "event_max_abs_estimate": _max_abs(event["estimate"], event["coefficient"]),
        "event_max_abs_se": _max_abs(event["se_pyd"], event["se_dd"]),
        "event_max_abs_p_value": _max_abs(event["p_value_pyd"], event["p_value_dd"]),
        "event_max_abs_ci_lower": _max_abs(event["ci_lower"], event["conf_low"]),
        "event_max_abs_ci_upper": _max_abs(event["ci_upper"], event["conf_high"]),
        "pooled_abs_estimate": abs(
            float(pooled_pyd["estimate"]) - float(pooled_dd["coefficient"])
        ),
        "pooled_abs_se": abs(float(pooled_pyd["se"]) - float(pooled_dd["se"])),
        "pooled_abs_p_value": abs(
            float(pooled_pyd["p_value"]) - float(pooled_dd["p_value"])
        ),
        "pooled_abs_ci_lower": abs(
            float(pooled_pyd["ci_lower"]) - float(pooled_dd["conf_low"])
        ),
        "pooled_abs_ci_upper": abs(
            float(pooled_pyd["ci_upper"]) - float(pooled_dd["conf_high"])
        ),
    }
    numeric = [value for key, value in row.items() if "abs_" in key]
    row["passed_1e_8"] = bool(max(numeric) <= 1e-8)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        _one_comparison(design, estimand)
        for design in ("absorbing", "first_entry", "effect_stabilization")
        for estimand in ("vw", "rw", "ra")
    ]
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    if not result["passed_1e_8"].all():
        raise SystemExit("External equivalence tolerance failed.")


if __name__ == "__main__":
    main()
