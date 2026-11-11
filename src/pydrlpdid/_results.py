"""Lean result containers for LP-DiD estimators.

Each estimator exposes only the inferential outputs needed for empirical work:

* ``event_study`` — horizon-specific estimates and confidence intervals;
* ``event_study_stable`` — optional fixed-composition path;
* ``scalars`` — pre-specified aggregate summaries.

Research-only horizon diagnostics are intentionally not part of the public result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def _sig_code(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _fmt(x: float, width: int = 12, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return f"{'nan':>{width}}"
    return f"{x:>{width}.{digits}f}"


@dataclass
class EventStudyResults:
    """Shared result object for LP-DiD estimators."""

    estimator_name: str
    n_obs: int
    n_treated_units: int
    n_control_units: int
    n_cohorts: int
    n_periods: int
    base_period: object
    clean_control: str
    effect_stabilization: Optional[int]
    anticipation: int
    inference: str
    alpha: float
    event_study: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_study_stable: pd.DataFrame = field(default_factory=pd.DataFrame)
    scalars: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Expose diff-diff column aliases without removing native names."""
        if not self.event_study.empty:
            aliases = {
                "event_time": "horizon",
                "att": "estimate",
                "coefficient": "estimate",
                "conf_low": "ci_lower",
                "conf_high": "ci_upper",
            }
            for alias, native in aliases.items():
                if alias not in self.event_study and native in self.event_study:
                    self.event_study[alias] = self.event_study[native]
        if not self.scalars.empty:
            aliases = {
                "att": "estimate",
                "coefficient": "estimate",
                "conf_low": "ci_lower",
                "conf_high": "ci_upper",
            }
            for alias, native in aliases.items():
                if alias not in self.scalars and native in self.scalars:
                    self.scalars[alias] = self.scalars[native]

    @property
    def event_study_effects(self) -> pd.DataFrame:
        return self.event_study

    @property
    def scalar_summaries(self) -> pd.DataFrame:
        return self.scalars

    @property
    def aggregates(self) -> pd.DataFrame:
        """All requested pre/post summaries with their explicit weighting rule."""
        return self.scalars.copy()

    @property
    def pooled(self) -> pd.DataFrame:
        """Pre/post aggregate table using the diff-diff results convention."""
        if self.scalars.empty:
            return pd.DataFrame()
        out = self.scalars.copy()
        if "window" not in out:
            term = out.get("term", pd.Series("", index=out.index)).astype(str).str.lower()
            out["window"] = np.where(term.str.contains("pre"), "pre", "post")
        if "aggregation" in out:
            pooled_rows = out["aggregation"].astype(str).eq("pooled")
            out = out.loc[pooled_rows].copy()
        elif "term" in out:
            pooled_rows = out["term"].astype(str).str.contains("pooled", case=False)
            if pooled_rows.any():
                out = out.loc[pooled_rows].copy()
        if "coefficient" not in out and "estimate" in out:
            out["coefficient"] = out["estimate"]
        if "conf_low" not in out and "ci_lower" in out:
            out["conf_low"] = out["ci_lower"]
        if "conf_high" not in out and "ci_upper" in out:
            out["conf_high"] = out["ci_upper"]
        order = pd.Categorical(out["window"], categories=["pre", "post"], ordered=True)
        return out.assign(_window_order=order).sort_values(
            "_window_order"
        ).drop(columns="_window_order").reset_index(drop=True)

    def _post_row(self) -> Optional[pd.Series]:
        if self.scalars.empty:
            return None
        candidates = self.scalars.copy()
        if "window" not in candidates:
            term = candidates.get(
                "term", pd.Series("", index=candidates.index)
            ).astype(str).str.lower()
            candidates["window"] = np.where(
                term.str.contains("pre"), "pre", "post"
            )
        post = candidates.loc[candidates["window"] == "post"]
        if post.empty:
            return None
        primary = str(self.metadata.get("primary_post_aggregation", "pooled"))
        if "aggregation" in post:
            selected = post.loc[post["aggregation"].astype(str).eq(primary)]
            if not selected.empty:
                return selected.iloc[0]
        # Backward compatibility for result objects created before aggregation
        # labels were public.
        if primary == "pooled" and "term" in post:
            selected = post.loc[
                post["term"].astype(str).str.contains("pooled", case=False)
            ]
            if not selected.empty:
                return selected.iloc[0]
        return post.iloc[0]

    def aggregate(self, window: str = "post", aggregation: str = "pooled") -> pd.Series:
        """Return one named aggregate without changing the headline ATT."""
        window = str(window).lower()
        aggregation = str(aggregation).lower().replace("-", "_")
        aliases = {
            "mean_horizons": "equal_horizon",
            "equal_horizons": "equal_horizon",
            "treated_event": "event_weighted",
        }
        aggregation = aliases.get(aggregation, aggregation)
        if window not in {"pre", "post"}:
            raise ValueError("window must be 'pre' or 'post'.")
        if self.scalars.empty or "aggregation" not in self.scalars:
            raise KeyError("No labeled aggregate summaries are available.")
        selected = self.scalars.loc[
            self.scalars["window"].astype(str).eq(window)
            & self.scalars["aggregation"].astype(str).eq(aggregation)
        ]
        if selected.empty:
            raise KeyError(
                f"Aggregate ({window!r}, {aggregation!r}) was not requested."
            )
        return selected.iloc[0].copy()

    @property
    def att(self) -> float:
        row = self._post_row()
        return float(row["estimate"]) if row is not None else float("nan")

    @property
    def se(self) -> float:
        row = self._post_row()
        return float(row["se"]) if row is not None else float("nan")

    @property
    def t_stat(self) -> float:
        row = self._post_row()
        return float(row["t_stat"]) if row is not None else float("nan")

    @property
    def p_value(self) -> float:
        row = self._post_row()
        return float(row["p_value"]) if row is not None else float("nan")

    @property
    def conf_int(self) -> tuple[float, float]:
        row = self._post_row()
        if row is None:
            return (float("nan"), float("nan"))
        return (float(row["ci_lower"]), float(row["ci_upper"]))

    @property
    def conf_int_lower(self) -> float:
        return float(self.conf_int[0])

    @property
    def conf_int_upper(self) -> float:
        return float(self.conf_int[1])

    @property
    def event_study_df(self) -> dict[int, float]:
        """Per-horizon inference degrees of freedom (diff-diff convention)."""
        if self.event_study.empty or "df" not in self.event_study:
            return {}
        return {
            int(row["horizon"]): float(row["df"])
            for _, row in self.event_study.iterrows()
            if np.isfinite(row.get("df", np.nan))
        }

    @property
    def pooled_df(self) -> dict[str, float]:
        """Pre/post scalar degrees-of-freedom provenance."""
        pooled = self.pooled
        if pooled.empty or "df" not in pooled:
            return {}
        out: dict[str, float] = {}
        for _, row in pooled.iterrows():
            value = row.get("df", np.nan)
            if np.isfinite(value):
                out[str(row["window"])] = float(value)
        return out

    def to_dict(self) -> dict:
        """Serialize headline results using diff-diff-compatible names."""
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int_lower,
            "conf_int_upper": self.conf_int_upper,
            "n_obs": int(self.n_obs),
            "n_treated_units": int(self.n_treated_units),
            "n_control_units": int(self.n_control_units),
            "event_study_df": self.event_study_df,
            "pooled_df": self.pooled_df,
            "inference_method": self.inference,
        }
        result.update({
            "control_group": self.metadata.get("control_group", self.clean_control),
            "cluster_name": self.metadata.get("cluster_level"),
            "df_convention": self.metadata.get("df_convention", "cluster"),
        })
        return result

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """Return a defensive copy of event-study or pooled results."""
        level = str(level).lower()
        if level in {"event", "event_study"}:
            return self.event_study.copy()
        if level == "pooled":
            return self.pooled.copy()
        if level in {"aggregate", "aggregates", "scalar", "scalars"}:
            return self.aggregates
        raise ValueError(
            "level must be 'event_study', 'pooled', or 'aggregates'."
        )

    @property
    def bootstrap_n_requested(self) -> Optional[int]:
        value = self.metadata.get("cluster_bootstrap", {}).get("n_requested")
        return None if value is None else int(value)

    @property
    def bootstrap_n_successful(self) -> Optional[int]:
        value = self.metadata.get("cluster_bootstrap", {}).get("n_successful")
        return None if value is None else int(value)

    @property
    def bootstrap_failure_rate(self) -> Optional[float]:
        info = self.metadata.get("cluster_bootstrap", {})
        if "success_rate" not in info:
            return None
        return 1.0 - float(info["success_rate"])

    @property
    def bootstrap_failures(self) -> dict:
        return dict(self.metadata.get("cluster_bootstrap", {}).get("failures", {}))

    def summary(self) -> str:
        width = 104
        lines: list[str] = []
        lines.append("=" * width)
        lines.append(f"{self.estimator_name:^{width}}")
        lines.append("=" * width)
        lines.append("")
        lines.append(f"Validated panel rows:     {self.n_obs:>24}")
        lines.append(f"Ever-treated units:       {self.n_treated_units:>24}")
        lines.append(f"Distinct clean controls:  {self.n_control_units:>24}")
        lines.append(f"First-onset dates:        {self.n_cohorts:>24}")
        lines.append(f"Time periods:             {self.n_periods:>24}")
        lines.append(f"Base period:              {str(self.base_period):>24}")
        design = self.metadata.get("design")
        if design:
            lines.append(f"Design:                   {str(design):>24}")
        lines.append(f"Clean-control rule:       {self.clean_control:>24}")
        if self.effect_stabilization is not None:
            lines.append(
                f"Clean-history length L:   {str(self.effect_stabilization):>24}"
            )
        lines.append(f"Inference:                {self.inference:>24}")
        correction = self.metadata.get("small_sample_correction")
        if correction:
            lines.append(
                f"Small-sample correction:  {str(correction).upper():>24}"
            )
        reference = self.metadata.get("reference_distribution")
        if reference:
            reference_label = (
                "Student t, target G-1"
                if reference == "student_t_G_minus_1" else str(reference)
            )
            lines.append(
                f"Reference distribution:   {reference_label:>24}"
            )
        lines.append(
            f"Cluster level:            {str(self.metadata.get('cluster_level', 'panel unit')):>24}"
        )

        if self.inference == "multiplier":
            multiplier = self.metadata.get("multiplier", {})
            if multiplier:
                lines.append(
                    f"Multiplier weights:        {str(multiplier.get('weight_type', 'not recorded')):>24}"
                )
        if self.inference == "cluster_bootstrap":
            boot = self.metadata.get("cluster_bootstrap", {})
            if boot:
                completed = f"{boot.get('n_successful')}/{boot.get('n_requested')}"
                lines.append(
                    f"Bootstrap successful:      "
                    f"{completed:>24}"
                )

        id_w = 34
        header = (
            f"{'Horizon':<{id_w}}"
            f"{'Estimate':>12}{'Std. Err.':>14}{'t-stat':>14}{'P>|t|':>14}{'Sig.':>8}"
        )

        if not self.scalars.empty:
            lines.append("")
            lines.append("-" * width)
            lines.append(f"{'Scalar summaries':^{width}}")
            lines.append("-" * width)
            lines.append(
                f"{'Term':<{id_w}}"
                f"{'Estimate':>12}{'Std. Err.':>14}{'t-stat':>14}{'P>|t|':>14}{'Sig.':>8}"
            )
            lines.append("-" * width)
            for _, row in self.scalars.iterrows():
                lines.append(
                    f"{str(row['term']):<{id_w}}"
                    f"{_fmt(row.get('estimate', np.nan), 12)}"
                    f"{_fmt(row.get('se', np.nan), 14)}"
                    f"{_fmt(row.get('t_stat', np.nan), 14)}"
                    f"{_fmt(row.get('p_value', np.nan), 14)}"
                    f"{_sig_code(row.get('p_value', np.nan)):>8}"
                )
            lines.append("-" * width)

        if not self.event_study.empty:
            lines.append("")
            lines.append("-" * width)
            lines.append(f"{'Event-study path':^{width}}")
            lines.append("-" * width)
            lines.append(header)
            lines.append("-" * width)
            for _, row in self.event_study.sort_values("horizon").iterrows():
                lines.append(
                    f"{int(row['horizon']):<{id_w}}"
                    f"{_fmt(row.get('estimate', np.nan), 12)}"
                    f"{_fmt(row.get('se', np.nan), 14)}"
                    f"{_fmt(row.get('t_stat', np.nan), 14)}"
                    f"{_fmt(row.get('p_value', np.nan), 14)}"
                    f"{_sig_code(row.get('p_value', np.nan)):>8}"
                )
            lines.append("-" * width)

        if not self.event_study_stable.empty:
            h_star = self.metadata.get("stable_horizon")
            lines.append("")
            lines.append("-" * width)
            lines.append(f"{f'Composition-stable path (H*={h_star})':^{width}}")
            lines.append("-" * width)
            lines.append(header)
            lines.append("-" * width)
            for _, row in self.event_study_stable.sort_values("horizon").iterrows():
                lines.append(
                    f"{int(row['horizon']):<{id_w}}"
                    f"{_fmt(row.get('estimate', np.nan), 12)}"
                    f"{_fmt(row.get('se', np.nan), 14)}"
                    f"{_fmt(row.get('t_stat', np.nan), 14)}"
                    f"{_fmt(row.get('p_value', np.nan), 14)}"
                    f"{_sig_code(row.get('p_value', np.nan)):>8}"
                )
            lines.append("-" * width)

        lines.append("")
        lines.append("Signif. codes:  '***' 0.001  '**' 0.01  '*' 0.05  '.' 0.1")
        lines.append("=" * width)
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())


@dataclass
class LPDIDResults(EventStudyResults):
    """Results for the LP-DiD estimator."""

    target_estimand: str = "VW"
    nonabsorbing: bool = False
    covariates: list = field(default_factory=list)


@dataclass
class DRLPDIDResults(EventStudyResults):
    """Results for the semiparametric DR-LP-DiD estimator."""

    estimation_method: str = "dr-ipt"
    dr_method: Optional[str] = "improved"
    covariates: list = field(default_factory=list)
    target_estimand: str = "ATT"
