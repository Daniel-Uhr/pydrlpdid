"""External-oracle tests for the public literature-standard LPDID API."""

import numpy as np
import pytest
import pydrlpdid

diff_diff = pytest.importorskip("diff_diff")
from diff_diff import LPDiD as OracleLPDiD

from pydrlpdid import LPDID
from replication.monte_carlo_v080 import make_panel


def test_public_api_has_one_lpdid_name():
    assert pydrlpdid.LPDID is LPDID
    assert "LPDID" in pydrlpdid.__all__
    assert "LPDiD" not in pydrlpdid.__all__
    assert not hasattr(pydrlpdid, "LPDiD")


@pytest.mark.parametrize("design", ["absorbing", "first_entry", "effect_stabilization"])
@pytest.mark.parametrize("estimand", ["vw", "rw", "ra"])
def test_public_lpdid_matches_diffdiff_312(design, estimand):
    data = make_panel(
        np.random.default_rng(20261110),
        180,
        "absorbing" if design == "absorbing" else "switching",
        "both_correct",
    )
    covariates = ["x", "q"] if estimand == "ra" else None
    reweight = estimand in {"rw", "ra"}
    common = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=reweight,
        cluster="id",
        df_convention="cluster",
    )
    if design == "first_entry":
        common["non_absorbing"] = "first_entry"
    elif design == "effect_stabilization":
        common.update(
            non_absorbing="effect_stabilization", stabilization_window=2
        )

    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=covariates,
    )
    actual = LPDID(**common).fit(**fit)
    expected = OracleLPDiD(**common).fit(**fit)

    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    )
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )
    np.testing.assert_allclose(
        merged["p_value_pyd"], merged["p_value_oracle"], atol=1e-10, rtol=1e-8
    )

    pyd_post = actual.pooled.loc[actual.pooled["window"].eq("post")].iloc[0]
    oracle_post = expected.pooled.loc[expected.pooled["window"].eq("post")].iloc[0]
    assert pyd_post["estimate"] == pytest.approx(
        oracle_post["coefficient"], abs=1e-10
    )
    assert pyd_post["se"] == pytest.approx(oracle_post["se"], rel=1e-8, abs=1e-10)
    assert actual.att == pytest.approx(pyd_post["estimate"], abs=1e-12)
    pyd_pre = actual.pooled.loc[actual.pooled["window"].eq("pre")].iloc[0]
    oracle_pre = expected.pooled.loc[expected.pooled["window"].eq("pre")].iloc[0]
    assert pyd_pre["estimate"] == pytest.approx(
        oracle_pre["coefficient"], abs=1e-10
    )
    assert pyd_pre["se"] == pytest.approx(
        oracle_pre["se"], rel=1e-8, abs=1e-10
    )


def test_public_lpdid_exposes_literature_result_names():
    data = make_panel(
        np.random.default_rng(11), 80, "absorbing", "both_correct"
    )
    result = LPDID(pre_window=2, post_window=1, cluster="id").fit(
        data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
    )
    required = {
        "horizon", "event_time", "estimate", "att", "coefficient",
        "se", "t_stat", "p_value", "ci_lower", "ci_upper",
        "conf_low", "conf_high", "df", "n_clusters",
    }
    assert required.issubset(result.event_study.columns)
    payload = result.to_dict()
    assert payload["att"] == pytest.approx(result.att)
    assert payload["conf_int_lower"] == pytest.approx(result.conf_int_lower)
    assert payload["conf_int_upper"] == pytest.approx(result.conf_int_upper)
    assert result.event_study_df
    assert result.pooled_df


@pytest.mark.parametrize("pmd", [2, "max"])
def test_public_lpdid_pmd_matches_diffdiff_with_missing_outcomes(pmd):
    data = make_panel(
        np.random.default_rng(27), 160, "absorbing", "both_correct"
    )
    rng = np.random.default_rng(91)
    missing = rng.uniform(size=len(data)) < 0.04
    data.loc[missing, "y"] = np.nan
    kwargs = dict(
        pre_window=2, post_window=2, control_group="clean",
        reweight=False, pmd=pmd, cluster="id", df_convention="cluster",
    )
    fit = dict(
        data=data, outcome="y", unit="id", time="time",
        treatment="treatment",
    )
    actual = LPDID(**kwargs).fit(**fit)
    expected = OracleLPDiD(**kwargs).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )


@pytest.mark.parametrize("design", ["absorbing", "first_entry", "effect_stabilization"])
@pytest.mark.parametrize("estimand", ["vw", "rw", "ra"])
def test_public_lpdid_no_composition_matches_diffdiff(design, estimand):
    """Hold the post-event sample fixed exactly as diff-diff does.

    Missing outcomes make this an actual composition test rather than another
    balanced-panel equality check.  Placebo leads remain horizon-specific.
    """
    data = make_panel(
        np.random.default_rng(20261111),
        180,
        "absorbing" if design == "absorbing" else "switching",
        "both_correct",
    )
    rng = np.random.default_rng(31415)
    data.loc[rng.uniform(size=len(data)) < 0.035, "y"] = np.nan
    covariates = ["x", "q"] if estimand == "ra" else None
    common = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=estimand in {"rw", "ra"},
        no_composition=True,
        cluster="id",
        df_convention="cluster",
    )
    if design == "first_entry":
        common["non_absorbing"] = "first_entry"
    elif design == "effect_stabilization":
        common.update(
            non_absorbing="effect_stabilization", stabilization_window=2
        )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=covariates,
    )
    actual = LPDID(**common).fit(**fit)
    expected = OracleLPDiD(**common).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )

    pyd_post = actual.pooled.loc[actual.pooled["window"].eq("post")].iloc[0]
    oracle_post = expected.pooled.loc[expected.pooled["window"].eq("post")].iloc[0]
    assert pyd_post["estimate"] == pytest.approx(
        oracle_post["coefficient"], abs=1e-10
    )
    assert pyd_post["se"] == pytest.approx(
        oracle_post["se"], rel=1e-8, abs=1e-10
    )
    pyd_pre = actual.pooled.loc[actual.pooled["window"].eq("pre")].iloc[0]
    oracle_pre = expected.pooled.loc[expected.pooled["window"].eq("pre")].iloc[0]
    assert pyd_pre["estimate"] == pytest.approx(
        oracle_pre["coefficient"], abs=1e-10
    )
    assert pyd_pre["se"] == pytest.approx(
        oracle_pre["se"], rel=1e-8, abs=1e-10
    )


@pytest.mark.parametrize(
    "pre_pooled,post_pooled", [(3, 2), ((-3, -2), (1, 2))]
)
def test_public_lpdid_custom_pooled_windows_match_diffdiff(
    pre_pooled, post_pooled
):
    data = make_panel(
        np.random.default_rng(909), 160, "absorbing", "both_correct"
    )
    constructor = dict(
        pre_window=3,
        post_window=3,
        control_group="clean",
        reweight=True,
        cluster="id",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=["x", "q"],
        pre_pooled=pre_pooled,
        post_pooled=post_pooled,
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    for window in ("pre", "post"):
        pyd_row = actual.pooled.loc[actual.pooled["window"].eq(window)].iloc[0]
        oracle_row = expected.pooled.loc[expected.pooled["window"].eq(window)].iloc[0]
        assert pyd_row["estimate"] == pytest.approx(
            oracle_row["coefficient"], abs=1e-10
        )
        assert pyd_row["se"] == pytest.approx(
            oracle_row["se"], rel=1e-8, abs=1e-10
        )


@pytest.mark.parametrize("ylags", [1, 2])
def test_public_lpdid_outcome_level_lags_match_diffdiff(ylags):
    data = make_panel(
        np.random.default_rng(1717), 180, "absorbing", "both_correct"
    )
    constructor = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=True,
        cluster="id",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        ylags=ylags,
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )
    for window in ("pre", "post"):
        pyd_row = actual.pooled.loc[actual.pooled["window"].eq(window)].iloc[0]
        oracle_row = expected.pooled.loc[expected.pooled["window"].eq(window)].iloc[0]
        assert pyd_row["estimate"] == pytest.approx(
            oracle_row["coefficient"], abs=1e-10
        )
        assert pyd_row["se"] == pytest.approx(
            oracle_row["se"], rel=1e-8, abs=1e-10
        )


@pytest.mark.parametrize("dylags", [1, 2])
def test_public_lpdid_outcome_change_lags_match_diffdiff(dylags):
    data = make_panel(
        np.random.default_rng(1818), 180, "absorbing", "both_correct"
    )
    constructor = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=True,
        cluster="id",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        dylags=dylags,
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )


@pytest.mark.parametrize("reweight", [False, True])
def test_public_lpdid_absorbed_factors_match_diffdiff(reweight):
    data = make_panel(
        np.random.default_rng(1919), 180, "absorbing", "both_correct"
    )
    data["region"] = data["id"] % 9
    constructor = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=reweight,
        cluster="id",
        rank_deficient_action="silent",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        absorb=["region"],
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )
    for window in ("pre", "post"):
        pyd_row = actual.pooled.loc[actual.pooled["window"].eq(window)].iloc[0]
        oracle_row = expected.pooled.loc[expected.pooled["window"].eq(window)].iloc[0]
        assert pyd_row["estimate"] == pytest.approx(
            oracle_row["coefficient"], abs=1e-10
        )
        assert pyd_row["se"] == pytest.approx(
            oracle_row["se"], rel=1e-8, abs=1e-10
        )


def test_public_lpdid_absorbed_factor_overlap_fails_closed_like_diffdiff():
    data = make_panel(
        np.random.default_rng(2020), 160, "absorbing", "both_correct"
    )
    ever_treated = data.groupby("id")["treatment"].transform("max")
    data["treated_only_region"] = ever_treated.astype(int)
    constructor = dict(
        pre_window=2,
        post_window=1,
        control_group="never_treated",
        reweight=True,
        cluster="id",
        rank_deficient_action="silent",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        absorb=["treated_only_region"],
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    )
    assert merged.loc[merged["horizon"].ne(-1), "estimate"].isna().all()
    assert merged.loc[
        merged["horizon"].ne(-1), "coefficient_oracle"
    ].isna().all()


def test_public_lpdid_coarser_cluster_matches_diffdiff():
    data = make_panel(
        np.random.default_rng(2121), 240, "absorbing", "both_correct"
    )
    data["state"] = data["id"] % 24
    constructor = dict(
        pre_window=2,
        post_window=2,
        control_group="clean",
        reweight=True,
        cluster="state",
    )
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=["x", "q"],
    )
    actual = LPDID(**constructor).fit(**fit)
    expected = OracleLPDiD(**constructor).fit(**fit)
    merged = actual.event_study.merge(
        expected.event_study, on="horizon", suffixes=("_pyd", "_oracle")
    ).dropna(subset=["estimate", "coefficient_oracle"])
    np.testing.assert_allclose(
        merged["estimate"], merged["coefficient_oracle"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        merged["se_pyd"], merged["se_oracle"], atol=1e-10, rtol=1e-8
    )


def test_public_lpdid_rank_deficient_action_is_enforced_and_reported():
    data = make_panel(
        np.random.default_rng(404), 120, "absorbing", "both_correct"
    )
    data["x_duplicate"] = data["x"]
    fit = dict(
        data=data,
        outcome="y",
        unit="id",
        time="time",
        treatment="treatment",
        covariates=["x", "x_duplicate"],
    )

    with pytest.warns(UserWarning, match="Rank-deficient LP-DiD design"):
        warned = LPDID(
            pre_window=2,
            post_window=1,
            reweight=True,
            rank_deficient_action="warn",
            cluster="id",
        ).fit(**fit)
    assert warned.metadata["rank_dropped_columns"]

    with pytest.raises(ValueError, match="Rank-deficient LP-DiD design"):
        LPDID(
            pre_window=2,
            post_window=1,
            reweight=True,
            rank_deficient_action="error",
            cluster="id",
        ).fit(**fit)

    silent = LPDID(
        pre_window=2,
        post_window=1,
        reweight=True,
        rank_deficient_action="silent",
        cluster="id",
    ).fit(**fit)
    assert silent.metadata["rank_dropped_columns"]
