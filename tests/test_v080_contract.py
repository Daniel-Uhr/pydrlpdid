import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from pydrlpdid import DRLPDID, LPDID
from pydrlpdid.lpdid import _LPDIDEngine
from pydrlpdid._errors import JacobianError
from pydrlpdid._inference import se_from_influence, stacked_influence
from pydrlpdid._panel_utils import prepare_panel, precompute_ccs
from pydrlpdid._panel_utils import build_local_sample
from pydrlpdid.lpdid import _fit_ra, _influence_ra

from test_formal_api import absorbing_panel, switching_panel


FIT = dict(outcome="y", unit="id", time="time", first_treat="g")


def large_first_entry_panel(seed=321):
    rng = np.random.default_rng(seed)
    paths = [
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ]
    rows = []
    unit = 0
    for _ in range(20):
        for path in paths:
            x = rng.normal()
            alpha = rng.normal()
            for time, treatment in enumerate(path, start=1):
                y = alpha + 0.3 * time + 0.2 * x * time + 1.5 * treatment
                y += rng.normal(scale=0.4)
                rows.append((unit, time, treatment, x, y))
            unit += 1
    return pd.DataFrame(rows, columns=["id", "time", "w", "x", "y"])


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
    assert mean.metadata["post_aggregation"] == "equal_horizon"
    assert pooled.metadata["post_aggregation"] == "pooled"


@pytest.mark.parametrize("estimator", ["lpdid", "drlpdid"])
def test_all_pre_post_aggregations_are_explicit_and_auditable(estimator):
    data = absorbing_panel()
    if estimator == "lpdid":
        result = LPDID(
            pre_window=3,
            post_window=4,
            reweight=True,
            pre_aggregation="all",
            post_aggregation="all",
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="time",
            treatment="w",
            covariates=["x"],
        )
    else:
        result = DRLPDID(
            estimation_method="dr-ipt",
            pre_window=3,
            post_window=4,
            pre_aggregation="all",
            post_aggregation="all",
        ).fit(data, treatment="w", covariates=["x"],
              outcome="y", unit="id", time="time")

    assert set(result.aggregates["window"]) == {"pre", "post"}
    assert set(result.aggregates["aggregation"]) == {
        "pooled", "equal_horizon", "event_weighted"
    }
    assert len(result.aggregates) == 6
    assert len(result.pooled) == 2
    assert result.att == pytest.approx(
        result.aggregate("post", "pooled")["estimate"]
    )

    event_row = result.aggregate("post", "event_weighted")
    horizons = list(event_row["horizons"])
    path = result.event_study.set_index("horizon").loc[horizons]
    weights = path["n_event_rows"].to_numpy(dtype=float)
    weights /= weights.sum()
    assert not np.allclose(weights, np.full_like(weights, 1.0 / len(weights)))
    assert tuple(event_row["weights"]) == pytest.approx(tuple(weights))
    assert event_row["estimate"] == pytest.approx(
        float(np.dot(weights, path["estimate"].to_numpy(dtype=float)))
    )
    assert np.isfinite(event_row["se"])


def test_mean_horizons_remains_a_legacy_alias():
    result = DRLPDID(
        estimation_method="ra",
        pre_window=2,
        post_window=2,
        post_aggregation="mean_horizons",
    ).fit(absorbing_panel(), treatment="w", covariates=["x"],
          outcome="y", unit="id", time="time")
    assert result.metadata["post_aggregation"] == "equal_horizon"
    assert result.aggregate("post", "mean_horizons")["aggregation"] == "equal_horizon"
    assert result.pooled.query("window == 'post'").empty


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


def test_cr1_is_the_default_and_none_is_the_only_alternative():
    data = absorbing_panel()
    default = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1), inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    explicit = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1),
        small_sample_correction="cr1", inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    uncorrected = DRLPDID(
        estimation_method="dr-ipw", horizons=(-2, 1),
        small_sample_correction="none", inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    np.testing.assert_allclose(
        default.event_study[["estimate", "se"]],
        explicit.event_study[["estimate", "se"]],
        equal_nan=True,
    )
    supported = default.event_study["horizon"] != -1
    assert np.all(
        default.event_study.loc[supported, "se"].to_numpy()
        > uncorrected.event_study.loc[supported, "se"].to_numpy()
    )
    assert default.metadata["small_sample_correction"] == "cr1"
    with pytest.raises(ValueError, match="small_sample_correction"):
        DRLPDID(small_sample_correction="diff_diff")


def test_analytic_cluster_inference_uses_t_G_minus_1():
    result = DRLPDID(
        estimation_method="ra", horizons=(-2, 1), inference="cluster",
    ).fit(absorbing_panel(), covariates=["x"], **FIT)
    row = result.event_study.query("horizon == 0").iloc[0]
    assert row["df"] == pytest.approx(row["n_clusters"] - 1)
    expected_p = 2.0 * student_t.sf(abs(row["t_stat"]), row["df"])
    expected_crit = student_t.ppf(0.975, row["df"])
    assert row["p_value"] == pytest.approx(expected_p)
    assert row["critical_value"] == pytest.approx(expected_crit)
    assert row["ci_lower"] == pytest.approx(
        row["estimate"] - expected_crit * row["se"]
    )


def test_lpdid_exposes_cr1_and_none_with_invariant_point_estimates():
    data = absorbing_panel()
    corrected = _LPDIDEngine(
        target_estimand="ra", max_pre=2, max_post=1,
        small_sample_correction="cr1",
    ).fit(data, outcome="y", unit="id", time="time",
          first_treat="g", covariates=["x"])
    uncorrected = _LPDIDEngine(
        target_estimand="ra", max_pre=2, max_post=1,
        small_sample_correction="none",
    ).fit(data, outcome="y", unit="id", time="time",
          first_treat="g", covariates=["x"])
    supported = corrected.event_study["horizon"] != -1
    np.testing.assert_allclose(
        corrected.event_study.loc[supported, "estimate"],
        uncorrected.event_study.loc[supported, "estimate"],
    )
    assert np.all(
        corrected.event_study.loc[supported, "se"].to_numpy()
        > uncorrected.event_study.loc[supported, "se"].to_numpy()
    )
    assert corrected.metadata["small_sample_correction"] == "cr1"
    with pytest.raises(ValueError, match="small_sample_correction"):
        _LPDIDEngine(small_sample_correction="diff_diff")


def test_condition_limit_is_enforced():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    theta = np.array([0.0, 0.0])

    def moments(th):
        return np.column_stack([x - th[0], 1e-9 * (x - th[1])])

    with pytest.raises(JacobianError, match="ill-conditioned"):
        stacked_influence(
            theta, moments, np.arange(4), condition_limit=1e6
        )


def test_stacked_influence_exposes_the_manual_cluster_sandwich():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    theta = np.array([x.mean()])

    def moments(th):
        return (x - th[0])[:, None]

    influence = stacked_influence(
        theta, moments, groups, target_grad=lambda th: np.array([1.0])
    )
    scores = np.array([
        (x[groups == group] - theta[0]).sum()
        for group in np.unique(groups)
    ])
    expected_variance = np.sum(scores ** 2) / len(x) ** 2
    assert influence["bread"][0, 0] == pytest.approx(-1.0)
    assert influence["meat"][0, 0] == pytest.approx(expected_variance)
    assert influence["covariance"][0, 0] == pytest.approx(expected_variance)
    assert np.sum(influence["psi"].reshape(-1) ** 2) == pytest.approx(
        expected_variance
    )
    assert influence["n_clusters"] == 3


@pytest.mark.parametrize("correction", ["none", "cr1"])
def test_drlpdid_ra_uses_the_same_stacked_sandwich_on_one_local_stack(correction):
    data = absorbing_panel()
    prepared = prepare_panel(
        data, "y", "id", "time", "g", "w", nonabsorbing=False
    )
    local = build_local_sample(
        prepared, "y", "id", "time", 0, -1, "not_yet_treated", None,
        ["x"], user_covariates=["x"],
    )
    model = DRLPDID(
        estimation_method="ra",
        small_sample_correction=correction,
        rank_deficient_action="silent",
    )
    actual = model._fit_one(local, "id", "time", ["x"], ["x"], True)
    expected = _influence_ra(local, "id", "time", ["x"], correction)
    assert actual["covariance_estimator"] == "cluster_sandwich"
    assert actual["estimate"] == pytest.approx(expected["estimate"], abs=1e-12)
    assert actual["se"] == pytest.approx(expected["se"], rel=1e-8, abs=1e-10)
    if correction == "cr1":
        assert actual["cr1_parameters"] == actual["n_moment_parameters"]


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
    result = _LPDIDEngine(
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
    result = _LPDIDEngine(
        target_estimand="ra", max_pre=2, max_post=2,
        inference="multiplier", n_bootstrap=49, seed=17,
    ).fit(absorbing_panel(), covariates=["x"], **FIT)
    columns = [
        "multiplier_ci_lower", "multiplier_ci_upper", "multiplier_p_value"
    ]
    assert len(result.scalars) == 2
    assert np.isfinite(result.scalars[columns].to_numpy()).all()


def test_default_rw_pre_weighting_is_horizon_specific():
    estimator = _LPDIDEngine(target_estimand="rw")
    assert estimator.pre_weighting == "horizon_specific"


def test_placebo_does_not_condition_on_its_own_outcome_change():
    result = DRLPDID(
        estimation_method="dr-ipw", design="absorbing", horizons=(-2, 1),
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


def test_drlpdid_literature_vocabulary_for_lags_absorb_and_result_selection():
    data = absorbing_panel().copy()
    data["region"] = data["id"] % 6
    common = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="w",
        covariates=["x"],
        ylags=1,
        dylags=1,
        absorb=["region"],
    )
    event_only = DRLPDID(
        estimation_method="ra",
        pre_window=2,
        post_window=1,
        cluster="id",
        rank_deficient_action="silent",
    ).fit(**common, only_event=True)
    assert not event_only.event_study.empty
    assert event_only.scalars.empty
    assert event_only.metadata["outcome_level_lags"] == 1
    assert event_only.metadata["outcome_change_lags"] == 1
    assert event_only.metadata["absorb"] == ("region",)
    assert any(
        "ly1" in str(column)
        for column in event_only.metadata["nuisance_diagnostics"][0][
            "or_retained_columns"
        ]
    )

    pooled_only = DRLPDID(
        estimation_method="ra",
        pre_window=2,
        post_window=1,
        cluster="id",
        rank_deficient_action="silent",
    ).fit(**common, only_pooled=True)
    assert pooled_only.event_study.empty
    assert set(pooled_only.pooled["window"]) == {"pre", "post"}
    assert set(pooled_only.pooled["aggregation"]) == {"pooled"}

    custom = DRLPDID(
        estimation_method="ra",
        pre_window=3,
        post_window=3,
        cluster="id",
        rank_deficient_action="silent",
    ).fit(
        data,
        outcome="y",
        unit="id",
        time="time",
        treatment="w",
        covariates=["x"],
        pre_pooled=(-3, -2),
        post_pooled=(1, 2),
    )
    assert tuple(custom.pooled.query("window == 'pre'").iloc[0]["horizons"]) == (-3, -2)
    assert tuple(custom.pooled.query("window == 'post'").iloc[0]["horizons"]) == (1, 2)


def test_drlpdid_rank_action_and_df_convention_are_explicit():
    data = absorbing_panel().copy()
    data["x_duplicate"] = data["x"]
    with pytest.raises(ValueError, match="Rank-deficient DRLPDID"):
        DRLPDID(
            estimation_method="ra",
            pre_window=2,
            post_window=0,
            rank_deficient_action="error",
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="time",
            treatment="w",
            covariates=["x", "x_duplicate"],
        )
    with pytest.raises(NotImplementedError, match="df_convention='cluster'"):
        DRLPDID(df_convention="residual")


def test_diff_diff_style_absorbing_api_and_results_surface():
    data = absorbing_panel()
    result = DRLPDID(
        estimation_method="dr-ipt",
        pre_window=2,
        post_window=2,
        control_group="clean",
        cluster="id",
        inference="cluster",
    ).fit(
        data,
        outcome="y",
        unit="id",
        time="time",
        treatment="w",
        covariates=["x"],
    )
    assert set(result.pooled["window"]) == {"pre", "post"}
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)
    assert np.isfinite(result.p_value)
    assert np.isfinite(result.conf_int).all()
    assert "coefficient" in result.event_study
    assert "coefficient" in result.to_dataframe(level="pooled")


def test_default_windows_match_literature_api_and_none_requests_full_grid():
    result = DRLPDID(
        estimation_method="ra", inference="cluster"
    ).fit(
        absorbing_panel(), outcome="y", unit="id", time="time",
        treatment="w", covariates=["x"],
    )
    assert result.metadata["pre_window"] == 2
    assert result.metadata["post_window"] == 0
    assert result.event_study["horizon"].min() == -2
    assert result.event_study["horizon"].max() == 0

    full = DRLPDID(
        estimation_method="ra", pre_window=None, post_window=None,
        inference="cluster",
    ).fit(
        absorbing_panel(), outcome="y", unit="id", time="time",
        treatment="w", covariates=["x"],
    )
    assert full.metadata["pre_window"] == 4
    assert full.metadata["post_window"] == 5
    assert full.event_study["horizon"].min() == -4
    assert full.event_study["horizon"].max() == 5


@pytest.mark.parametrize("method", ["ra", "ipw", "ipt", "dr-ipw", "dr-ipt"])
def test_new_and_legacy_absorbing_surfaces_are_numerically_identical(method):
    data = absorbing_panel()
    legacy = DRLPDID(
        estimation_method=method,
        horizons=(-2, 2),
        post_window=(0, 2),
        post_aggregation="pooled",
        inference="cluster",
    ).fit(data, covariates=["x"], **FIT)
    modern = DRLPDID(
        estimation_method=method,
        pre_window=2,
        post_window=2,
        post_aggregation="pooled",
        control_group="clean",
        cluster="id",
        inference="cluster",
    ).fit(
        data, outcome="y", unit="id", time="time",
        treatment="w", covariates=["x"],
    )
    pd.testing.assert_frame_equal(
        legacy.event_study[["horizon", "estimate", "se"]].reset_index(drop=True),
        modern.event_study[["horizon", "estimate", "se"]].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert legacy.att == pytest.approx(modern.att, rel=1e-12, abs=1e-12)
    assert legacy.se == pytest.approx(modern.se, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("weights", ["rademacher", "mammen", "webb"])
def test_all_multiplier_weights_are_exposed(weights):
    result = DRLPDID(
        estimation_method="ra",
        pre_window=2,
        post_window=1,
        inference="multiplier",
        bootstrap_weights=weights,
        n_bootstrap=19,
        seed=7,
    ).fit(absorbing_panel(), treatment="w", covariates=["x"],
          outcome="y", unit="id", time="time")
    assert result.metadata["multiplier"]["weight_type"] == weights


def test_first_entry_is_public_for_nonabsorbing_treatment():
    result = DRLPDID(
        estimation_method="ra",
        pre_window=2,
        post_window=1,
        non_absorbing="first_entry",
        control_group="clean",
        left_censoring="drop",
        inference="cluster",
    ).fit(
        switching_panel(), outcome="y", unit="id", time="time", treatment="w"
    )
    assert result.metadata["design"] == "switching"
    assert result.clean_control == "first_entry"
    assert np.isfinite(result.event_study.query("horizon != -1")["estimate"]).any()


def test_first_entry_excludes_prior_treatment_and_intermediate_reversal():
    paths = {
        # Eligible first entry at t=3, sustained through h=2.
        "eligible_event": [0, 0, 1, 1, 1],
        # Valid never-treated clean control.
        "clean_control": [0, 0, 0, 0, 0],
        # Previously treated, then switched off before t=3: never a clean
        # first-entry control even though D_{t+2}=0.
        "prior_treated": [1, 0, 0, 0, 0],
        # First entry at t=3, but exits and returns before h=2: endpoint-only
        # checking would incorrectly retain this event.
        "exit_reenter": [0, 0, 1, 0, 1],
    }
    rows = []
    for unit, path in paths.items():
        for time, treatment in enumerate(path, start=1):
            rows.append(
                {
                    "id": unit,
                    "time": time,
                    "w": treatment,
                    "y": float(time) + 0.1 * len(rows),
                }
            )
    data = pd.DataFrame(rows)
    prepared = prepare_panel(
        data, "y", "id", "time", None, "w", nonabsorbing=True,
        left_censoring="keep_as_control_after_L",
        history_policy="diff_diff",
    )
    local = build_local_sample(
        prepared, "y", "id", "time", 2, -1, "first_entry", None, [],
    )
    membership = set(
        local[["id", "time", "D_local"]].itertuples(index=False, name=None)
    )
    assert ("eligible_event", 3, 1) in membership
    assert ("clean_control", 3, 0) in membership
    assert not any(row[0] == "prior_treated" for row in membership)
    assert not any(row[0] == "exit_reenter" for row in membership)


@pytest.mark.parametrize("method", ["ra", "ipw", "ipt", "dr-ipw", "dr-ipt"])
def test_first_entry_runs_for_all_estimation_methods(method):
    result = DRLPDID(
        estimation_method=method,
        pre_window=2,
        post_window=1,
        non_absorbing="first_entry",
        inference="cluster",
    ).fit(
        large_first_entry_panel(), outcome="y", unit="id", time="time",
        treatment="w", covariates=["x"],
    )
    supported = result.event_study.query("horizon != -1")
    assert np.isfinite(supported[["estimate", "se"]].to_numpy()).all()


@pytest.mark.parametrize("method", ["ra", "ipw", "ipt", "dr-ipw", "dr-ipt"])
def test_cluster_bootstrap_is_available_for_all_dr_api_methods(method):
    result = DRLPDID(
        estimation_method=method,
        pre_window=2,
        post_window=1,
        inference="cluster_bootstrap",
        n_bootstrap=3,
        min_bootstrap_success=2 / 3,
        seed=11,
    ).fit(absorbing_panel(), treatment="w", covariates=["x"],
          outcome="y", unit="id", time="time")
    assert result.bootstrap_n_requested == 3
    assert result.bootstrap_n_successful >= 2
    assert result.bootstrap_failure_rate <= 1 / 3


def _nested_cluster_panel():
    data = absorbing_panel().copy()
    ordered = {value: idx for idx, value in enumerate(pd.unique(data["id"]))}
    data["state"] = data["id"].map(lambda value: ordered[value] // 2)
    return data


def _aggregate_unit_influence_to_state(core, data, horizon):
    mapping = data[["id", "state"]].drop_duplicates().set_index("id")["state"]
    unit_psi = core["psi_by_h"][horizon]
    return unit_psi.groupby(mapping.reindex(unit_psi.index).to_numpy()).sum()


def test_drlpdid_coarser_cluster_matches_manual_influence_aggregation():
    data = _nested_cluster_panel()
    kwargs = dict(
        estimation_method="ra", horizons=(-2, 1),
        small_sample_correction="none", inference="cluster",
    )
    by_unit = DRLPDID(**kwargs)
    unit_core = by_unit._fit_core(
        data, "y", "id", "time", "g", None, ["x"]
    )
    by_state = DRLPDID(**kwargs, cluster="state")
    state_core = by_state._fit_core(
        data, "y", "id", "time", "g", None, ["x"]
    )
    for horizon in (-2, 0, 1):
        manual = _aggregate_unit_influence_to_state(unit_core, data, horizon)
        direct = state_core["psi_by_h"][horizon]
        np.testing.assert_allclose(
            direct.sort_index().to_numpy(),
            manual.reindex(direct.index).sort_index().to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        direct_se = state_core["event_study"].set_index("horizon").loc[horizon, "se"]
        assert direct_se == pytest.approx(
            se_from_influence(manual.to_numpy()), rel=1e-10, abs=1e-12
        )


def test_lpdid_coarser_cluster_matches_manual_influence_aggregation():
    data = _nested_cluster_panel()
    kwargs = dict(
        target_estimand="ra", max_pre=2, max_post=1,
        small_sample_correction="none", inference="cluster",
    )
    by_unit = _LPDIDEngine(**kwargs)
    unit_core = by_unit._fit_core(
        data, "y", "id", "time", "g", None, ["x"]
    )
    by_state = _LPDIDEngine(**kwargs, cluster="state")
    state_core = by_state._fit_core(
        data, "y", "id", "time", "g", None, ["x"]
    )
    for horizon in (-2, 0, 1):
        manual = _aggregate_unit_influence_to_state(unit_core, data, horizon)
        direct = state_core["psi_by_h"][horizon]
        np.testing.assert_allclose(
            direct.sort_index().to_numpy(),
            manual.reindex(direct.index).sort_index().to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        direct_se = state_core["event_study"].set_index("horizon").loc[horizon, "se"]
        assert direct_se == pytest.approx(
            se_from_influence(manual.to_numpy()), rel=1e-10, abs=1e-12
        )


@pytest.mark.parametrize("estimator", ["dr", "lp"])
def test_panel_unit_cannot_move_between_inference_clusters(estimator):
    data = _nested_cluster_panel()
    first_id = data["id"].iloc[0]
    idx = data.index[data["id"].eq(first_id)][0]
    data.loc[idx, "state"] = 999
    if estimator == "dr":
        model = DRLPDID(estimation_method="ra", cluster="state")
    else:
        model = _LPDIDEngine(target_estimand="ra", cluster="state")
    with pytest.raises(ValueError, match="exactly one inference cluster"):
        model.fit(data, outcome="y", unit="id", time="time", first_treat="g")

