"""County-year companion aggregate that excludes the one-cell h=4 tail."""

import json

import pandas as pd
from scipy import stats

from run_received_minwage_v080 import (
    OUTPUT,
    RECEIVED,
    county_year_filter,
    estimators,
    fit_model,
)
from pydrlpdid._inference import se_from_influence


def main() -> None:
    data_path = RECEIVED / "data" / "clean" / "county_year.parquet"
    panel = pd.read_parquet(data_path)
    panel["year"] = panel["year"].astype(int)
    fit_kwargs = dict(
        outcome="y_emp722", unit="unit", time="year", treatment="d"
    )
    window = [0, 1, 2]
    leads = [-4, -3, -2]
    scalar_rows = []
    scalar_psi = {}
    for label, (model, covariates) in estimators("county_year").items():
        _, scalar, psi = fit_model(
            panel, "county_year_short", label, model, covariates,
            county_year_filter(), fit_kwargs, window, leads,
        )
        scalar_rows.append(scalar)
        scalar_psi[label] = psi

    scalars = pd.DataFrame(scalar_rows)
    a, b = "LPDID-RW+X1", "DRLPDID-DR-IPT|X1"
    union = scalar_psi[a].index.union(scalar_psi[b].index)
    gap_psi = scalar_psi[a].reindex(union, fill_value=0.0) - scalar_psi[b].reindex(
        union, fill_value=0.0
    )
    indexed = scalars.set_index("estimator")
    gap = float(indexed.loc[a, "estimate"] - indexed.loc[b, "estimate"])
    gap_se = se_from_influence(gap_psi.to_numpy())
    gap_df = len(gap_psi) - 1
    gap_p = float(2 * stats.t.sf(abs(gap / gap_se), gap_df))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    scalars.to_csv(OUTPUT / "county_year_scalars_h0_h2.csv", index=False)
    with (OUTPUT / "county_year_assessment_h0_h2.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "window": window,
                "linear_minus_dript": gap,
                "linear_minus_dript_se": gap_se,
                "linear_minus_dript_p_value": gap_p,
                "dript_smaller_standard_error_than_linear": bool(
                    indexed.loc[b, "se"] < indexed.loc[a, "se"]
                ),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
