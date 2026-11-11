"""Minimal diff-diff-style DRLPDID example with generated panel data."""

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, LPDID


def make_panel(seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    units = np.arange(160)
    periods = np.arange(1, 11)
    first_treat = np.repeat([4, 6, 0, 0], 40)
    unit_effect = rng.normal(size=len(units))
    covariate = rng.normal(size=len(units))
    rows = []
    for unit in units:
        for year in periods:
            treated = int(first_treat[unit] > 0 and year >= first_treat[unit])
            outcome = (
                unit_effect[unit]
                + 0.3 * year
                + 0.15 * covariate[unit] * year
                + 2.0 * treated
                + rng.normal(scale=0.5)
            )
            rows.append((unit, year, treated, covariate[unit], outcome))
    return pd.DataFrame(rows, columns=["id", "year", "treated", "x", "y"])


if __name__ == "__main__":
    df = make_panel()

    benchmark = LPDID(
        pre_window=3,
        post_window=3,
        control_group="clean",
        reweight=True,
        cluster="id",
        pre_aggregation="all",
        post_aggregation="all",
    ).fit(
        df,
        outcome="y",
        unit="id",
        time="year",
        treatment="treated",
        covariates=["x"],
    )
    print(benchmark.summary())
    print("\nLP-DiD pre/post aggregates:")
    print(benchmark.aggregates)

    result = DRLPDID(
        estimation_method="dr-ipt",
        pre_window=3,
        post_window=3,
        control_group="clean",
        cluster="id",
        inference="cluster",
        pre_aggregation="all",
        post_aggregation="all",
    ).fit(
        df,
        outcome="y",
        unit="id",
        time="year",
        treatment="treated",
        covariates=["x"],
    )

    print("\n" + result.summary())
    print("\nEvent-study effects:")
    print(result.to_dataframe(level="event_study"))
    print("\nPre/post aggregates:")
    print(result.to_dataframe(level="aggregates"))
    comparison = result.aggregate("post", "event_weighted")
    print(f"\nPooled post-treatment ATT: {result.att:.4f}")
    print(f"Event-weighted comparison ATT: {comparison['estimate']:.4f}")

