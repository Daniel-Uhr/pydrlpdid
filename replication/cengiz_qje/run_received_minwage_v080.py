"""Re-estimate the received minimum-wage application with pydrlpdid 0.8.

This driver leaves the received archive and extracted project unchanged.  It
uses their frozen clean panels and horizon-specific sample filters, but asks
the current package to aggregate county influence functions natively at the
state level through ``cluster='state_fips'``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

RECEIVED = Path(
    os.environ.get(
        "PYDRLPDID_RECEIVED_MINWAGE_DIR",
        REPO / "replication" / "cengiz_qje" / "received_project",
    )
).expanduser().resolve()
sys.path.insert(1, str(RECEIVED / "code"))

from diagnostics import StackFilter, _patched_builder  # noqa: E402
from pydrlpdid import DRLPDID  # noqa: E402
from pydrlpdid.lpdid import _LPDIDEngine  # noqa: E402
from pydrlpdid._inference import se_from_influence  # noqa: E402


OUTPUT = REPO / "results" / "cengiz_qje" / "received_v080"
X1 = [f"div{k}" for k in range(2, 10)]
X2_SQ = X1 + ["x_unemp_l4", "x_manuf_l4", "x_lpop_l4"]
X2_CY = X1 + ["x_unemp_l1", "x_manuf_l1", "x_lpop_l1"]


class CaptureLPDID(_LPDIDEngine):
    def _fit_core(self, *args, **kwargs):
        core = super()._fit_core(*args, **kwargs)
        self.captured_core = core
        return core


class CaptureDRLPDID(DRLPDID):
    def _fit_core(self, *args, **kwargs):
        core = super()._fit_core(*args, **kwargs)
        self.captured_core = core
        return core


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_quarter_filter() -> StackFilter:
    clean = RECEIVED / "data" / "clean"
    federal = pd.read_csv(clean / "federal_increase_tq.csv")["tq"].astype(int)
    events = pd.read_csv(clean / "state_quarter_events.csv")[["state_fips", "tq"]]
    return StackFilter(
        "state_fips", "tq", L=8, federal_dates=federal.tolist(),
        control_events=events, require_complete=X2_SQ,
        zero_entrant_group="division",
    )


def county_year_filter() -> StackFilter:
    clean = RECEIVED / "data" / "clean"
    federal = pd.read_csv(clean / "federal_increase_year.csv")["year"].astype(int)
    events = pd.read_csv(clean / "county_year_events.csv")[["unit", "year"]]
    return StackFilter(
        "unit", "year", L=2, federal_dates=federal.tolist(),
        control_events=events, require_complete=X2_CY,
        zero_entrant_group="division",
    )


def scalar_from_core(result, core, window: list[int], leads: list[int]) -> tuple[dict, pd.Series]:
    es = result.event_study.set_index("horizon")
    missing = [h for h in window if h not in es.index or not np.isfinite(es.loc[h, "estimate"])]
    if missing:
        raise RuntimeError(f"Unsupported scalar horizon(s): {missing}")
    psi = pd.concat(
        [core["psi_by_h"][h].rename(h) for h in window], axis=1
    ).fillna(0.0)
    psi_bar = psi.mean(axis=1)
    estimate = float(es.loc[window, "estimate"].mean())
    se = se_from_influence(psi_bar.to_numpy())
    g = int(len(psi_bar))
    df = g - 1
    t_stat = estimate / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    critical = float(stats.t.ppf(0.975, df))

    valid_leads = [h for h in leads if h in es.index and np.isfinite(es.loc[h, "estimate"])]
    pre_b = es.loc[valid_leads, "estimate"].to_numpy(dtype=float)
    pre_psi = pd.concat(
        [core["psi_by_h"][h].rename(h) for h in valid_leads], axis=1
    ).fillna(0.0).to_numpy(dtype=float)
    covariance = pre_psi.T @ pre_psi
    rank = int(np.linalg.matrix_rank(covariance))
    wald = float(pre_b @ np.linalg.pinv(covariance) @ pre_b)
    pre_p = float(stats.chi2.sf(wald, rank))
    return {
        "estimate": estimate,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": estimate - critical * se,
        "ci_upper": estimate + critical * se,
        "n_clusters": g,
        "pretrend_wald": wald,
        "pretrend_df": rank,
        "pretrend_p": pre_p,
    }, psi_bar


def fit_model(
    panel: pd.DataFrame,
    design: str,
    label: str,
    estimator,
    covariates: list[str],
    stack_filter: StackFilter,
    fit_kwargs: dict,
    window: list[int],
    leads: list[int],
) -> tuple[pd.DataFrame, dict, pd.Series]:
    print(f"Fitting {design}: {label}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _patched_builder(stack_filter):
            result = estimator.fit(
                panel, covariates=(covariates or None), **fit_kwargs
            )
    event = result.event_study.copy()
    event.insert(0, "estimator", label)
    event.insert(0, "design", design)
    scalar, psi = scalar_from_core(
        result, estimator.captured_core, window, leads
    )
    scalar.update({"design": design, "estimator": label})
    return event, scalar, psi


def estimators(design: str):
    if design == "state_quarter":
        pre, post, L, cluster = 8, 12, 8, "state_fips"
    else:
        pre, post, L, cluster = 4, 6, 2, "state_fips"
    lp_common = dict(
        max_pre=pre, max_post=post, clean_control="stabilized",
        nonabsorbing=True, effect_stabilization=L, inference="cluster",
        left_censoring="drop", cluster=cluster,
        small_sample_correction="cr1",
    )
    dr_common = dict(
        pre_window=pre, post_window=post, design="switching",
        stabilization_window=L, inference="cluster", cluster=cluster,
        left_censoring="drop",
        small_sample_correction="cr1",
    )
    return {
        "LPDID-RW": (CaptureLPDID(target_estimand="rw", **lp_common), []),
        "LPDID-RW+X1": (CaptureLPDID(target_estimand="rw", **lp_common), X1),
        "DRLPDID-RA|X1": (
            CaptureDRLPDID(estimation_method="ra", **dr_common), X1
        ),
        "DRLPDID-DR-IPW|X1": (
            CaptureDRLPDID(estimation_method="dr-ipw", **dr_common), X1
        ),
        "DRLPDID-IPT|X1": (
            CaptureDRLPDID(estimation_method="ipt", **dr_common), X1
        ),
        "DRLPDID-DR-IPT|X1": (
            CaptureDRLPDID(estimation_method="dr-ipt", **dr_common), X1
        ),
    }


def run_design(design: str):
    clean = RECEIVED / "data" / "clean"
    if design == "state_quarter":
        data_path = clean / "state_quarter.parquet"
        panel = pd.read_parquet(data_path)
        panel["tq"] = panel["tq"].astype(int)
        fit_kwargs = dict(
            outcome="y_emp722", unit="state_fips", time="tq", treatment="d"
        )
        window, leads = list(range(0, 9)), list(range(-8, -1))
        make_filter = state_quarter_filter
        old_path = RECEIVED / "output" / "tables" / "main_scalars.csv"
    else:
        data_path = clean / "county_year.parquet"
        panel = pd.read_parquet(data_path)
        panel["year"] = panel["year"].astype(int)
        fit_kwargs = dict(
            outcome="y_emp722", unit="unit", time="year", treatment="d"
        )
        window, leads = list(range(0, 5)), list(range(-4, -1))
        make_filter = county_year_filter
        old_path = RECEIVED / "output" / "tables" / "cy_main_scalars.csv"

    event_rows, scalar_rows, psi = [], [], {}
    for label, (model, covariates) in estimators(design).items():
        event, scalar, scalar_psi = fit_model(
            panel, design, label, model, covariates, make_filter(),
            fit_kwargs, window, leads,
        )
        event_rows.append(event)
        scalar_rows.append(scalar)
        psi[label] = scalar_psi

    events = pd.concat(event_rows, ignore_index=True)
    scalars = pd.DataFrame(scalar_rows)
    old = pd.read_csv(old_path)
    old = old.loc[old["outcome"].eq("y_emp722")]
    comparison = scalars.merge(
        old[["estimator", "estimate", "se"]], on="estimator", how="left",
        suffixes=("_v080", "_received_v072"),
    )
    comparison["estimate_difference"] = (
        comparison["estimate_v080"] - comparison["estimate_received_v072"]
    )
    comparison["se_difference"] = (
        comparison["se_v080"] - comparison["se_received_v072"]
    )

    a, b = "LPDID-RW+X1", "DRLPDID-DR-IPT|X1"
    gap_psi = psi[a].reindex(psi[a].index.union(psi[b].index), fill_value=0.0) - psi[b].reindex(
        psi[a].index.union(psi[b].index), fill_value=0.0
    )
    gap = float(
        scalars.set_index("estimator").loc[a, "estimate"]
        - scalars.set_index("estimator").loc[b, "estimate"]
    )
    gap_se = se_from_influence(gap_psi.to_numpy())
    gap_df = len(gap_psi) - 1
    gap_p = float(2 * stats.t.sf(abs(gap / gap_se), gap_df))
    assessment = {
        "design": design,
        "data_sha256": sha256(data_path),
        "window": window,
        "linear_minus_dript": gap,
        "linear_minus_dript_se": gap_se,
        "linear_minus_dript_p_value": gap_p,
        "dript_smaller_standard_error_than_linear": bool(
            scalars.set_index("estimator").loc[b, "se"]
            < scalars.set_index("estimator").loc[a, "se"]
        ),
        "ipt_dript_max_abs_event_difference": float(
            np.nanmax(np.abs(
                events.loc[events.estimator.eq("DRLPDID-IPT|X1"), "estimate"].to_numpy()
                - events.loc[events.estimator.eq("DRLPDID-DR-IPT|X1"), "estimate"].to_numpy()
            ))
        ),
    }
    return events, scalars, comparison, assessment, panel, make_filter, fit_kwargs


def validate_county_state_clustering(panel, make_filter, fit_kwargs) -> dict:
    """Validate direct state clustering against manual county-IF aggregation."""
    common = dict(
        target_estimand="rw", max_pre=2, max_post=1,
        clean_control="stabilized", nonabsorbing=True,
        effect_stabilization=2, inference="cluster",
        left_censoring="drop", small_sample_correction="none",
    )
    county = CaptureLPDID(**common)
    with warnings.catch_warnings(), _patched_builder(make_filter()):
        warnings.simplefilter("ignore")
        county.fit(panel, covariates=X1, **fit_kwargs)
    state = CaptureLPDID(**common, cluster="state_fips")
    with warnings.catch_warnings(), _patched_builder(make_filter()):
        warnings.simplefilter("ignore")
        state.fit(panel, covariates=X1, **fit_kwargs)
    mapping = panel.drop_duplicates("unit").set_index("unit")["state_fips"]
    checks = {}
    for horizon in (-2, 0, 1):
        county_psi = county.captured_core["psi_by_h"][horizon]
        manual = county_psi.groupby(mapping.reindex(county_psi.index).to_numpy()).sum()
        direct = state.captured_core["psi_by_h"][horizon]
        manual = manual.reindex(direct.index, fill_value=0.0)
        checks[str(horizon)] = {
            "max_abs_if_difference": float(
                np.max(np.abs(manual.to_numpy() - direct.to_numpy()))
            ),
            "manual_se": se_from_influence(manual.to_numpy()),
            "direct_se": se_from_influence(direct.to_numpy()),
        }
    return checks


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_assessments = []
    county_validation = None
    for design in ("state_quarter", "county_year"):
        events, scalars, comparison, assessment, panel, make_filter, fit_kwargs = run_design(design)
        events.to_csv(OUTPUT / f"{design}_event_study.csv", index=False)
        scalars.to_csv(OUTPUT / f"{design}_scalars.csv", index=False)
        comparison.to_csv(OUTPUT / f"{design}_comparison_v072_v080.csv", index=False)
        all_assessments.append(assessment)
        if design == "county_year":
            county_validation = validate_county_state_clustering(
                panel, make_filter, fit_kwargs
            )
        print(comparison.to_string(index=False), flush=True)

    with (OUTPUT / "assessment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "assessments": all_assessments,
                "county_state_cluster_validation": county_validation,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
