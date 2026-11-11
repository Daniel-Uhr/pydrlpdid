import unittest

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, LPDID, __version__


def absorbing_panel(seed=123):
    rng = np.random.default_rng(seed)
    units = np.arange(80)
    times = np.arange(1, 9)
    first_treat = np.repeat([3, 4, 5, 0], 20)
    alpha = rng.normal(size=len(units))
    x = rng.normal(size=len(units))
    rows = []
    for unit in units:
        for period in times:
            treated = int(
                first_treat[unit] > 0 and period >= first_treat[unit]
            )
            outcome = (
                alpha[unit]
                + 0.4 * period
                + 0.5 * x[unit] * period
                + 2.0 * treated
                + rng.normal(scale=0.5)
            )
            rows.append(
                (
                    unit,
                    period,
                    first_treat[unit],
                    treated,
                    x[unit],
                    outcome,
                )
            )
    return pd.DataFrame(
        rows, columns=["id", "time", "g", "w", "x", "y"]
    )


class EstimatorIdentity072Tests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.8.1")

    def test_dube_ra_and_formal_ra_have_distinct_identities(self):
        data = absorbing_panel()
        common_fit = dict(
            data=data,
            outcome="y",
            unit="id",
            time="time",
            covariates=["x"],
        )
        dube_ra = LPDID(
            pre_window=2,
            post_window=2,
            control_group="clean",
            reweight=True,
            inference="cluster",
        ).fit(treatment="w", **common_fit)
        formal_ra = DRLPDID(
            estimation_method="ra",
            design="absorbing",
            control_group="not_yet_treated",
            horizons=(-2, 2),
            inference="cluster",
        ).fit(first_treat="g", **common_fit)

        self.assertIn("Local Projections DiD", dube_ra.estimator_name)
        self.assertEqual(
            dube_ra.metadata["estimator_identity"], "dube_2025_lpdid_ra"
        )
        self.assertEqual(
            dube_ra.metadata["estimation_sample"],
            "dube_clean_control_sample",
        )
        self.assertEqual(
            formal_ra.metadata["estimator_identity"], "drlpdid_ra"
        )
        self.assertEqual(formal_ra.estimator_name, "RA")
        self.assertEqual(
            formal_ra.metadata["estimator_label"], "RA"
        )
        self.assertEqual(
            formal_ra.metadata["support"],
            "reference_dates_with_events_and_clean_controls",
        )

    def test_drlpdid_ra_uses_the_dube_outcome_training_sample_by_default(self):
        data = absorbing_panel()
        common_fit = dict(
            data=data,
            outcome="y",
            unit="id",
            time="time",
            covariates=["x"],
        )
        dube_ra = LPDID(
            pre_window=2,
            post_window=2,
            control_group="clean",
            reweight=True,
            inference="cluster",
            small_sample_correction="cr1",
        ).fit(treatment="w", **common_fit)
        dr_ra = DRLPDID(
            estimation_method="ra",
            horizons=(-2, 2),
            inference="cluster",
            small_sample_correction="cr1",
        ).fit(first_treat="g", **common_fit)

        merged = dube_ra.event_study.merge(
            dr_ra.event_study,
            on="horizon",
            suffixes=("_lpdid", "_drlpdid"),
        ).dropna(subset=["estimate_lpdid", "estimate_drlpdid"])
        np.testing.assert_allclose(
            merged["estimate_lpdid"], merged["estimate_drlpdid"],
            atol=1e-10, rtol=0,
        )
        np.testing.assert_allclose(
            merged["se_lpdid"], merged["se_drlpdid"],
            atol=1e-10, rtol=1e-8,
        )
        self.assertEqual(
            dr_ra.metadata["or_training_support"], "all_clean_controls"
        )

    def test_drlpdid_ra_matches_dube_placebo_lag_convention(self):
        data = absorbing_panel(seed=2026)
        common_fit = dict(
            data=data,
            outcome="y",
            unit="id",
            time="time",
            covariates=["x"],
            ylags=2,
            dylags=2,
        )
        dube_ra = LPDID(
            pre_window=3,
            post_window=2,
            control_group="clean",
            reweight=True,
            inference="cluster",
            small_sample_correction="cr1",
            rank_deficient_action="silent",
        ).fit(treatment="w", **common_fit)
        dr_ra = DRLPDID(
            estimation_method="ra",
            pre_window=3,
            post_window=2,
            inference="cluster",
            small_sample_correction="cr1",
            rank_deficient_action="silent",
        ).fit(first_treat="g", **common_fit)

        merged = dube_ra.event_study.merge(
            dr_ra.event_study,
            on="horizon",
            suffixes=("_lpdid", "_drlpdid"),
        ).dropna(subset=["estimate_lpdid", "estimate_drlpdid"])
        np.testing.assert_allclose(
            merged["estimate_lpdid"], merged["estimate_drlpdid"],
            atol=1e-9, rtol=0,
        )
        np.testing.assert_allclose(
            merged["se_lpdid"], merged["se_drlpdid"],
            atol=1e-9, rtol=1e-8,
        )
        self.assertNotIn(
            -2, dr_ra.metadata["omitted_overlapping_outcome_controls"]
        )

    def test_public_dube_ra_multiplier_bands(self):
        data = absorbing_panel()
        result = LPDID(
            pre_window=2,
            post_window=2,
            control_group="clean",
            reweight=True,
            inference="multiplier",
            n_bootstrap=49,
            bootstrap_weights="rademacher",
            seed=123,
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="time",
            treatment="w",
            covariates=["x"],
        )
        estimated = result.event_study["horizon"].ne(-1)
        self.assertTrue(
            result.event_study.loc[
                estimated, ["sim_ci_lower", "sim_ci_upper"]
            ].notna().all().all()
        )
        self.assertEqual(
            result.metadata["multiplier"]["weight_type"], "rademacher"
        )


if __name__ == "__main__":
    unittest.main()
