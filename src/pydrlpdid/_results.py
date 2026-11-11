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

    @property
    def event_study_effects(self) -> pd.DataFrame:
        return self.event_study

    @property
    def scalar_summaries(self) -> pd.DataFrame:
        return self.scalars

    @property
    def pooled(self) -> pd.DataFrame:
        """Pre/post aggregate table using the diff-diff results convention."""
        if self.scalars.empty:
            return pd.DataFrame()
        out = self.scalars.copy()
        if "window" not in out:
            term = out.get("term", pd.Series("", index=out.index)).astype(str).str.lower()
            out["window"] = np.where(term.str.contains("pre"), "pre", "post")
        if "coefficient" not in out and "estimate" in out:
            out["coefficient"] = out["estimate"]
        if "conf_low" not in out and "ci_lower" in out:
            out["conf_low"] = out["ci_lower"]
        if "conf_high" not in out and "ci_upper" in out:
            out["conf_high"] = out["ci_upper"]
        return out

    def _post_row(self) -> Optional[pd.Series]:
        pooled = self.pooled
        if pooled.empty:
            return None
        post = pooled.loc[pooled["window"] == "post"]
        return None if post.empty else post.iloc[0]

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

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """Return a defensive copy of event-study or pooled results."""
        level = str(level).lower()
        if level in {"event", "event_study"}:
            return self.event_study.copy()
        if level == "pooled":
            return self.pooled.copy()
        raise ValueError("level must be 'event_study' or 'pooled'.")

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
