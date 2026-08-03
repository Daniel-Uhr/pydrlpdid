import inspect
import unittest

import numpy as np
import pandas as pd

from pydrlpdid import DRLPDID, __version__
from pydrlpdid._panel_utils import (
    build_local_sample,
    precompute_ccs,
    prepare_panel,
)


def absorbing_panel(seed=123):
    rng = np.random.default_rng(seed)
    units = np.arange(120)
    times = np.arange(1, 9)
    g = np.repeat([3, 4, 5, 0], 30)
    alpha = rng.normal(size=len(units))
    x = rng.normal(size=len(units))
    rows = []
    for i in units:
        for t in times:
            treated = int(g[i] > 0 and t >= g[i])
            y = alpha[i] + 0.4 * t + 0.5 * x[i] * t + 2.0 * treated
            y += rng.normal(scale=0.5)
            rows.append((i, t, g[i], treated, x[i], y))
    return pd.DataFrame(
        rows, columns=["id", "time", "g", "w", "x", "y"]
    )


def switching_panel():
    paths = {
        0: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        3: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        5: [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        # Conservatively removed by the formal public switching API.
        6: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    rows = []
    for unit, path in paths.items():
        for time, treatment in enumerate(path, start=1):
            outcome = 0.25 * unit + 0.4 * time + 1.5 * treatment
            rows.append((unit, time, treatment, outcome))
    return pd.DataFrame(rows, columns=["id", "time", "w", "y"])


class FormalAPITests(unittest.TestCase):
    def test_version_and_constructor_are_narrow(self):
        self.assertEqual(__version__, "0.7.2")
        parameters = set(inspect.signature(DRLPDID).parameters)
        self.assertEqual(
            parameters,
            {
                "estimation_method",
                "design",
                "control_group",
                "stabilization_window",
                "horizons",
                "post_window",
                "inference",
                "n_bootstrap",
                "alpha",
                "seed",
            },
        )

    def test_invalid_design_combinations_fail(self):
        with self.assertRaises(ValueError):
            DRLPDID(design="switching")
        with self.assertRaises(ValueError):
            DRLPDID(design="absorbing", stabilization_window=4)
        with self.assertRaises(ValueError):
            DRLPDID(design="switching", stabilization_window=4,
                     control_group="never_treated")

    def test_all_methods_share_the_supported_stack(self):
        df = absorbing_panel()
        counts = []
        for method in ["ra", "ipw", "ipt", "dr-ipw", "dr-ipt"]:
            result = DRLPDID(
                estimation_method=method,
                horizons=(-2, 2),
                inference="cluster",
            ).fit(
                df, outcome="y", unit="id", time="time",
                first_treat="g", covariates=["x"],
            )
            counts.append(
                result.event_study[
                    ["horizon", "n_event_rows", "n_control_rows",
                     "n_reference_dates"]
                ].reset_index(drop=True)
            )
            supported = result.event_study.query("horizon != -1")
            self.assertTrue(np.isfinite(supported["se"]).all())
        for table in counts[1:]:
            pd.testing.assert_frame_equal(table, counts[0])

    def test_multiplier_keeps_cluster_robust_pointwise_interval(self):
        df = absorbing_panel()
        kwargs = dict(
            data=df, outcome="y", unit="id", time="time",
            first_treat="g", covariates=["x"],
        )
        cluster = DRLPDID(
            estimation_method="dr-ipw", horizons=(-2, 2),
            inference="cluster",
        ).fit(**kwargs)
        multiplier = DRLPDID(
            estimation_method="dr-ipw", horizons=(-2, 2),
            inference="multiplier", n_bootstrap=49, seed=123,
        ).fit(**kwargs)
        np.testing.assert_allclose(
            cluster.event_study[["ci_lower", "ci_upper"]],
            multiplier.event_study[["ci_lower", "ci_upper"]],
            equal_nan=True,
        )
        base = multiplier.event_study.set_index("horizon").loc[-1]
        self.assertEqual(base["estimate"], 0.0)
        self.assertEqual(base["ci_lower"], 0.0)
        self.assertEqual(base["ci_upper"], 0.0)

    def test_complete_prespecified_scalar(self):
        df = absorbing_panel()
        result = DRLPDID(
            estimation_method="dr-ipw",
            horizons=(-2, 2),
            post_window=(0, 2),
            inference="cluster",
        ).fit(
            df, outcome="y", unit="id", time="time",
            first_treat="g", covariates=["x"],
        )
        self.assertEqual(len(result.scalars), 1)
        self.assertEqual(int(result.scalars.iloc[0]["n_horizons"]), 3)

    def test_rank_reduction_preserves_ipt_dript_identity(self):
        df = absorbing_panel()
        df["x_duplicate"] = df["x"]
        fit_kwargs = dict(
            data=df,
            outcome="y",
            unit="id",
            time="time",
            first_treat="g",
            covariates=["x", "x_duplicate"],
        )
        ipt = DRLPDID(
            estimation_method="ipt",
            horizons=(-2, 2),
            post_window=(0, 2),
            inference="cluster",
        ).fit(**fit_kwargs)
        dript = DRLPDID(
            estimation_method="dr-ipt",
            horizons=(-2, 2),
            post_window=(0, 2),
            inference="cluster",
        ).fit(**fit_kwargs)
        supported = ipt.event_study["horizon"] != -1
        differences = np.abs(
            ipt.event_study.loc[supported, "estimate"].to_numpy()
            - dript.event_study.loc[supported, "estimate"].to_numpy()
        )
        for h, diagnostics in dript.metadata["nuisance_diagnostics"].items():
            self.assertLessEqual(
                diagnostics["ipt_dript_nested_difference"],
                diagnostics["ipt_dript_identity_tolerance"],
            )
            self.assertIn("x_duplicate", diagnostics["ipt_dropped_columns"])
        self.assertTrue(np.isfinite(differences).all())
        np.testing.assert_allclose(
            ipt.event_study.loc[supported, "estimate"].to_numpy(),
            dript.event_study.loc[supported, "estimate"].to_numpy(),
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            ipt.event_study.loc[supported, "se"].to_numpy(),
            dript.event_study.loc[supported, "se"].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            ipt.scalars[["estimate", "se"]].to_numpy(),
            dript.scalars[["estimate", "se"]].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_formal_switching_stack_membership(self):
        with self.assertWarns(UserWarning):
            prepared = prepare_panel(
                switching_panel(),
                outcome="y",
                unit="id",
                time="time",
                first_treat=None,
                treatment="w",
                nonabsorbing=True,
                left_censoring="drop",
            )
        self.assertNotIn(6, set(prepared["id"]))
        prepared = precompute_ccs(
            prepared, unit="id", effect_stabilization=2,
            max_pre=1, max_post=2,
        )

        h0 = build_local_sample(
            prepared, "y", "id", "time", 0, -1, "stabilized", 2, [],
            control_pool="stabilized_all", switch_in="sustained",
            control_window="horizon",
        )
        events_h0 = set(
            map(tuple, h0.loc[h0["D_local"].eq(1), ["id", "time"]].to_numpy())
        )
        self.assertIn((5, 8), events_h0)  # eligible re-entry
        self.assertIn((3, 5), events_h0)  # entry before its later reversal

        h2 = build_local_sample(
            prepared, "y", "id", "time", 2, -1, "stabilized", 2, [],
            control_pool="stabilized_all", switch_in="sustained",
            control_window="horizon",
        )
        events_h2 = set(
            map(tuple, h2.loc[h2["D_local"].eq(1), ["id", "time"]].to_numpy())
        )
        self.assertIn((0, 4), events_h2)
        self.assertIn((4, 5), events_h2)
        self.assertNotIn((3, 5), events_h2)  # reverses before t+2

        controls_t5 = h2.loc[
            h2["D_local"].eq(0) & h2["time"].eq(5), ["id", "_treat"]
        ]
        self.assertTrue(
            ((controls_t5["id"] == 1) & (controls_t5["_treat"] == 0)).any()
        )
        self.assertTrue(
            ((controls_t5["id"] == 2) & (controls_t5["_treat"] == 1)).any()
        )


if __name__ == "__main__":
    unittest.main()
