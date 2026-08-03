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
        self.assertEqual(__version__, "0.7.2")

    def test_dube_ra_and_formal_ra_have_distinct_identities(self):
        data = absorbing_panel()
        kwargs = dict(
            data=data,
            outcome="y",
            unit="id",
            time="time",
            first_treat="g",
            covariates=["x"],
        )
        dube_ra = LPDID(
            target_estimand="ra",
            base_period=-1,
            clean_control="not_yet_treated",
            inference="cluster",
            max_pre=2,
            max_post=2,
        ).fit(**kwargs)
        formal_ra = DRLPDID(
            estimation_method="ra",
            design="absorbing",
            control_group="not_yet_treated",
            horizons=(-2, 2),
            inference="cluster",
        ).fit(**kwargs)

        self.assertEqual(dube_ra.estimator_name, "LPDID-RA")
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
        self.assertEqual(formal_ra.estimator_name, "DRLPDID-RA")
        self.assertEqual(
            formal_ra.metadata["estimator_label"], "DRLPDID-RA"
        )
        self.assertEqual(
            formal_ra.metadata["support"],
            "reference_dates_with_events_and_clean_controls",
        )

    def test_native_dube_ra_multiplier_bands(self):
        data = absorbing_panel()
        result = LPDID(
            target_estimand="ra",
            base_period=-1,
            clean_control="not_yet_treated",
            inference="multiplier",
            n_bootstrap=49,
            bootstrap_weights="rademacher",
            seed=123,
            max_pre=2,
            max_post=2,
        ).fit(
            data,
            outcome="y",
            unit="id",
            time="time",
            first_treat="g",
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
