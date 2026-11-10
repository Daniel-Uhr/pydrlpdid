import numpy as np
import pandas as pd
import pytest

from pydrlpdid import DRLPDID, LPDID
from pydrlpdid._errors import JacobianError
from pydrlpdid._inference import se_from_influence, stacked_influence
from pydrlpdid._panel_utils import prepare_panel, precompute_ccs
from pydrlpdid._panel_utils import build_local_sample
from pydrlpdid.lpdid import _fit_ra, _influence_ra

from test_formal_api import absorbing_panel, switching_panel


FIT = dict(outcome="y", unit="id", time="time", first_treat="g")


def test_post_window_requires_an_explicit_estimand():
    with pytest.raises(ValueError, match="post_aggregation"):
        DRLPDID(horizons=(-2, 2), post_window=(0, 2))


def test_scalar_multiplier_inference_is_complete():
    result = DRLPDID(
        estimation_method="dr-ipw",
        horizons=(-2, 2),
        post_window=(0, 2),
        post_aggregation="mean_horizons",
        inference="multiplier",
        n_bootstrap=49,
        seed=91,
    ).fit(absorbing_panel(), covariates=["x"], **FIT)
    required = {
        "multiplier_ci_lower", "multiplier_ci_upper", "multiplier_p_value"
    }
    assert required.issubset(result.scalars.columns)
    assert np.isfinite(result.scalars[list(required)].to_numpy()).all()


def test_pooled_and_mean_horizons_are_distinct_named_estimands():
    data = absorbing_panel()
    common = dict(
        estimation_method="ra", horizons=(-2, 2), post_window=(0, 2),
        inference="cluster",
    )
    mean = DRLPDID(
        **common, post_aggregation="mean_horizons"
    ).fit(data, covariates=["x"], **FIT)
    pooled = DRLPDID(
        **common, post_aggregation="pooled"
    ).fit(data, covariates=["x"], **FIT)
    assert "policy-window" in mean.scalars.iloc[0]["term"]
    assert "pooled" in pooled.scalars.iloc[0]["term"]
    assert mean.metadata["post_aggregation"] == "mean_horizons"
    assert pooled.metadata["post_aggregation"] == "pooled"


def test_cr1_scales_the_full_influence_representation():
    data = absorbing_panel()
    base = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1),
        small_sample_correction="none", inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    cr1 = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1),
        small_sample_correction="cr1", inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    supported = base.event_study.query("horizon != -1").set_index("horizon")
    corrected = cr1.event_study.query("horizon != -1").set_index("horizon")
    for h in supported.index:
        factor = cr1.metadata["nuisance_diagnostics"][int(h)]["small_sample_factor"]
        assert factor >= 1.0
        assert corrected.loc[h, "se"] == pytest.approx(
            factor * supported.loc[h, "se"], rel=1e-8, abs=1e-10
        )


def test_condition_limit_is_enforced():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    theta = np.array([0.0, 0.0])

    def moments(th):
        return np.column_stack([x - th[0], 1e-9 * (x - th[1])])

    with pytest.raises(JacobianError, match="ill-conditioned"):
        stacked_influence(
            theta, moments, np.arange(4), condition_limit=1e6
        )


def test_observed_history_never_creates_a_first_row_switch():
    data = switching_panel()
    observed = prepare_panel(
        data, "y", "id", "time", None, "w", nonabsorbing=True,
        left_censoring="keep_as_control_after_L", history_policy="observed",
    )
    compatible = prepare_panel(
        data, "y", "id", "time", None, "w", nonabsorbing=True,
        left_censoring="keep_as_control_after_L", history_policy="diff_diff",
    )
    first_observed = observed.groupby("id", sort=False).head(1)
    assert first_observed["D_treat"].isna().all()
    first_compatible = compatible.groupby("id", sort=False).head(1)
    assert first_compatible.loc[first_compatible["id"].eq(6), "D_treat"].iloc[0] == 1


def test_left_censored_unit_enters_stable_one_pool_only_after_observed_history():
    prepared = prepare_panel(
        switching_panel(), "y", "id", "time", None, "w",
        nonabsorbing=True, left_censoring="keep_as_control_after_L",
        history_policy="observed",
    )
    ccs = precompute_ccs(prepared, "id", 2, 1, 1)
    unit = ccs.loc[ccs["id"].eq(6)].set_index("time")
    assert not bool(unit.loc[1, "CCS_0"])
    assert not bool(unit.loc[2, "CCS_0"])
    assert not bool(unit.loc[3, "CCS_0"])
    assert bool(unit.loc[4, "CCS_0"])


def test_lpdid_ra_point_and_influence_share_one_design():
    data = absorbing_panel()
    data["x_duplicate"] = data["x"]
    result = LPDID(
        target_estimand="ra", max_pre=2, max_post=2,
        inference="cluster",
    ).fit(
        data, covariates=["x", "x_duplicate"], **FIT
    )
    supported = result.event_study.query("horizon != -1")
    assert np.isfinite(supported[["estimate", "se"]].to_numpy()).all()

    prepared = prepare_panel(
        data, "y", "id", "time", "g", "w", nonabsorbing=False
    )
    local = build_local_sample(
        prepared, "y", "id", "time", 0, -1, "not_yet_treated", None,
        ["x", "x_duplicate"], user_covariates=["x", "x_duplicate"],
    )
    point = _fit_ra(local, "id", "time", ["x", "x_duplicate"])
    influence = _influence_ra(local, "id", "time", ["x", "x_duplicate"])
    assert influence["estimate"] == pytest.approx(point["estimate"], abs=1e-12)


def test_lpdid_multiplier_covers_native_scalars():
    result = LPDID(
        target_estimand="ra", max_pre=2, max_post=2,
        inference="multiplier", n_bootstrap=49, seed=17,
    ).fit(absorbing_panel(), covariates=["x"], **FIT)
    columns = [
        "multiplier_ci_lower", "multiplier_ci_upper", "multiplier_p_value"
    ]
    assert len(result.scalars) == 2
    assert np.isfinite(result.scalars[columns].to_numpy()).all()


def test_default_rw_pre_weighting_is_horizon_specific():
    estimator = LPDID(target_estimand="rw")
    assert estimator.pre_weighting == "horizon_specific"


def test_placebo_does_not_condition_on_its_own_outcome_change():
    result = DRLPDID(
        estimation_method="ra", design="absorbing", horizons=(-2, 1),
        outcome_change_lags=1, inference="cluster",
    )
    fitted = result.fit(absorbing_panel(), covariates=["x"], **FIT)
    assert fitted.metadata["omitted_overlapping_outcome_controls"][-2] == ("ldy1",)


def test_overlap_diagnostics_are_reported():
    result = DRLPDID(
        estimation_method="dr-ipt", horizons=(-2, 1), inference="cluster"
    ).fit(absorbing_panel(), covariates=["x"], **FIT)
    for h, diag in result.metadata["nuisance_diagnostics"].items():
        assert 0 <= diag["propensity_min"] <= diag["propensity_max"] <= 1
        assert diag["control_weight_ess"] > 0


def test_separate_nuisance_bases_are_public_and_recorded():
    data = absorbing_panel()
    data["x2"] = data["x"] ** 2
    result = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1), inference="cluster"
    ).fit(
        data, ps_covariates=["x"], or_covariates=["x", "x2"], **FIT
    )
    assert result.metadata["ps_covariates"] == ["x"]
    assert result.metadata["or_covariates"] == ["x", "x2"]

