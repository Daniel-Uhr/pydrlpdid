"""Public LP-DiD estimator using the literature-standard diff-diff API."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from .lpdid import _LPDIDEngine
from ._aggregation import normalize_aggregations


class LPDID:
    """LP-DiD with the constructor and fit vocabulary used by diff-diff.

    Additional pydrlpdid inference choices remain available as keyword-only
    parameters. This is the package's single public LP-DiD benchmark API.
    """

    def __init__(
        self,
        pre_window: int = 2,
        post_window: int = 0,
        control_group: str = "clean",
        reweight: bool = False,
        no_composition: bool = False,
        pmd: Optional[Union[str, int]] = None,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
        non_absorbing: Optional[str] = None,
        stabilization_window: Optional[int] = None,
        df_convention: str = "cluster",
        *,
        inference: str = "cluster",
        small_sample_correction: str = "method_default",
        bootstrap_weights: str = "mammen",
        n_bootstrap: int = 499,
        seed: Optional[int] = None,
        history_policy: str = "diff_diff",
        left_censoring: str = "keep_as_control_after_L",
        pre_aggregation="pooled",
        post_aggregation="pooled",
    ) -> None:
        for name, value in (("pre_window", pre_window), ("post_window", post_window)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if control_group not in {"clean", "never_treated"}:
            raise ValueError("control_group must be 'clean' or 'never_treated'.")
        if not isinstance(reweight, bool):
            raise ValueError("reweight must be boolean.")
        if not isinstance(no_composition, bool):
            raise ValueError("no_composition must be boolean.")
        if rank_deficient_action not in {"warn", "error", "silent"}:
            raise ValueError(
                "rank_deficient_action must be 'warn', 'error', or 'silent'."
            )
        if non_absorbing not in {None, "first_entry", "effect_stabilization"}:
            raise ValueError(
                "non_absorbing must be None, 'first_entry', or "
                "'effect_stabilization'."
            )
        if non_absorbing == "effect_stabilization":
            if (
                isinstance(stabilization_window, bool)
                or not isinstance(stabilization_window, int)
                or stabilization_window < 1
            ):
                raise ValueError(
                    "stabilization_window must be a positive integer for "
                    "effect_stabilization."
                )
        elif stabilization_window is not None:
            raise ValueError(
                "stabilization_window applies only to effect_stabilization."
            )
        if non_absorbing is not None and control_group != "clean":
            raise ValueError(
                "Non-absorbing LP-DiD requires control_group='clean'."
            )
        if df_convention not in {"cluster"}:
            raise NotImplementedError(
                "The public LPDID API currently certifies "
                "df_convention='cluster' only."
            )
        if small_sample_correction not in {"method_default", "cr1", "none"}:
            raise ValueError(
                "small_sample_correction must be 'method_default', 'cr1', or 'none'."
            )
        self.pre_window = pre_window
        self.post_window = post_window
        self.control_group = control_group
        self.reweight = reweight
        self.no_composition = no_composition
        self.pmd = pmd
        self.alpha = float(alpha)
        self.cluster = cluster
        self.rank_deficient_action = rank_deficient_action
        self.non_absorbing = non_absorbing
        self.stabilization_window = stabilization_window
        self.df_convention = df_convention
        self.inference = inference
        self.small_sample_correction = small_sample_correction
        self.bootstrap_weights = bootstrap_weights
        self.n_bootstrap = int(n_bootstrap)
        self.seed = seed
        self.history_policy = history_policy
        self.left_censoring = left_censoring
        self.pre_aggregations = normalize_aggregations(
            pre_aggregation, name="pre_aggregation"
        )
        self.post_aggregations = normalize_aggregations(
            post_aggregation, name="post_aggregation"
        )
        self.pre_aggregation = (
            "all" if self.pre_aggregations == (
                "pooled", "equal_horizon", "event_weighted"
            ) else (
                self.pre_aggregations[0]
                if len(self.pre_aggregations) == 1 else self.pre_aggregations
            )
        )
        self.post_aggregation = (
            "all" if self.post_aggregations == (
                "pooled", "equal_horizon", "event_weighted"
            ) else (
                self.post_aggregations[0]
                if len(self.post_aggregations) == 1 else self.post_aggregations
            )
        )
        self.results_ = None
        self.is_fitted_ = False
        self._engine = None

    def _base_period(self):
        if self.pmd is None:
            return -1
        if self.pmd == "max":
            return "all_pre"
        if isinstance(self.pmd, bool) or not isinstance(self.pmd, int) or self.pmd < 1:
            raise ValueError("pmd must be None, 'max', or a positive integer.")
        return list(range(-int(self.pmd), 0))

    def _clean_control(self) -> str:
        if self.non_absorbing == "first_entry":
            return "first_entry"
        if self.non_absorbing == "effect_stabilization":
            return "stabilized"
        return "never_treated" if self.control_group == "never_treated" else "not_yet_treated"

    def _resolve_pooled_horizons(self, pooled, *, kind: str) -> list[int]:
        if isinstance(pooled, bool):
            raise ValueError(
                f"{kind}_pooled must be None, an int, or a length-2 tuple."
            )
        if kind == "pre":
            supported = set(range(-self.pre_window, -1))
            if pooled is None:
                horizons = sorted(supported)
            elif isinstance(pooled, int):
                horizons = list(range(-pooled, -1))
            elif isinstance(pooled, tuple) and len(pooled) == 2:
                horizons = list(range(int(pooled[0]), int(pooled[1]) + 1))
            else:
                raise ValueError(
                    "pre_pooled must be None, an int, or a length-2 tuple."
                )
        else:
            supported = set(range(0, self.post_window + 1))
            if pooled is None:
                horizons = sorted(supported)
            elif isinstance(pooled, int):
                horizons = list(range(0, pooled + 1))
            elif isinstance(pooled, tuple) and len(pooled) == 2:
                horizons = list(range(int(pooled[0]), int(pooled[1]) + 1))
            else:
                raise ValueError(
                    "post_pooled must be None, an int, or a length-2 tuple."
                )
        invalid = [h for h in horizons if h not in supported]
        if invalid:
            raise ValueError(
                f"Requested pooled {kind} horizons {invalid} fall outside "
                f"the supported window {sorted(supported)}."
            )
        return horizons

    def fit(
        self,
        data,
        outcome,
        unit,
        time,
        treatment,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
        post_pooled=None,
        pre_pooled=None,
        only_event=False,
        only_pooled=False,
        survey_design=None,
    ):
        if survey_design is not None:
            raise NotImplementedError("survey_design is outside the pydrlpdid 0.8 scope.")
        if only_event and only_pooled:
            raise ValueError("only_event and only_pooled cannot both be True.")
        for name, value in (("ylags", ylags), ("dylags", dylags)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        pre_pool_h = self._resolve_pooled_horizons(pre_pooled, kind="pre")
        post_pool_h = self._resolve_pooled_horizons(post_pooled, kind="post")
        if not only_event and not pre_pool_h:
            raise ValueError(
                "pooled pre window is empty; use only_event=True or set "
                "pre_window >= 2."
            )

        covariates = list(covariates or [])
        absorb = list(absorb or [])
        data_for_fit = data
        absorbed_dummy_columns: list[str] = []
        if absorb:
            missing_columns = [column for column in absorb if column not in data]
            if missing_columns:
                raise ValueError(f"Absorbed columns not found: {missing_columns}")
            data_for_fit = data.copy()
            for factor_number, column in enumerate(absorb):
                values = data_for_fit[column]
                dummies = pd.get_dummies(
                    values,
                    prefix=f"__pyd_absorb_{factor_number}",
                    drop_first=True,
                    dtype=float,
                )
                if not dummies.empty:
                    dummies.loc[values.isna(), :] = float("nan")
                    data_for_fit = pd.concat([data_for_fit, dummies], axis=1)
                    absorbed_dummy_columns.extend(dummies.columns.astype(str))
            covariates.extend(absorbed_dummy_columns)
        target = (
            "ra" if self.reweight and (covariates or ylags or dylags or absorb)
            else ("rw" if self.reweight else "vw")
        )
        correction = self.small_sample_correction
        if correction == "method_default":
            correction = "none" if target == "ra" else "cr1"

        self._engine = _LPDIDEngine(
            target_estimand=target,
            base_period=self._base_period(),
            clean_control=self._clean_control(),
            nonabsorbing=(self.non_absorbing is not None),
            effect_stabilization=self.stabilization_window,
            include_lagged_outcome_change=bool(dylags),
            n_lagged_outcome_changes=int(dylags),
            n_lagged_outcome_levels=int(ylags),
            inference=self.inference,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            alpha=self.alpha,
            seed=self.seed,
            max_pre=self.pre_window,
            max_post=self.post_window,
            no_composition=self.no_composition,
            left_censoring=self.left_censoring,
            history_policy=self.history_policy,
            pre_weighting="horizon_specific",
            cluster=self.cluster,
            small_sample_correction=correction,
            rank_deficient_action=self.rank_deficient_action,
            pooled_pre_horizons=(None if only_event else pre_pool_h),
            pooled_post_horizons=post_pool_h,
            pre_aggregations=list(self.pre_aggregations),
            post_aggregations=list(self.post_aggregations),
            # diff-diff keeps lagged outcome changes even when a placebo
            # outcome is their algebraic negative.  This reproduces that
            # published convention (and the resulting mechanically zero lead)
            # in the literature LPDID convention; DRLPDID omits overlaps.
            omit_overlapping_outcome_controls=False,
            absorb_columns=absorb,
        )
        result = self._engine.fit(
            data_for_fit,
            outcome=outcome,
            unit=unit,
            time=time,
            treatment=treatment,
            covariates=covariates,
        )
        result.estimator_name = (
            "Local Projections DiD (Dube, Girardi, Jorda & Taylor 2025)"
        )
        if self.pmd is not None:
            # diff-diff keeps h=-1 as the displayed normalization even when
            # PMD uses a multi-period baseline. Preserve the richer internal
            # baseline representation behind the public normalization.
            base = result.event_study["horizon"].eq(-1)
            if base.any():
                for column in ("estimate", "att", "coefficient"):
                    if column in result.event_study:
                        result.event_study.loc[base, column] = 0.0
                for column in (
                    "se", "t_stat", "p_value", "ci_lower", "ci_upper",
                    "conf_low", "conf_high", "df", "critical_value",
                ):
                    if column in result.event_study:
                        result.event_study.loc[base, column] = float("nan")
                if "n_clusters" in result.event_study:
                    result.event_study.loc[base, "n_clusters"] = float("nan")
        result.metadata.update(
            {
                "public_api": "literature_standard_lpdid",
                "control_group": self.control_group,
                "reweight": self.reweight,
                "no_composition": self.no_composition,
                "pmd": self.pmd,
                "pre_window": self.pre_window,
                "post_window": self.post_window,
                "df_convention": self.df_convention,
                "rank_deficient_action": self.rank_deficient_action,
                "absorb": tuple(absorb),
                "resolved_target_estimand": target,
                "estimand": {
                    "vw": "variance-weighted ATT",
                    "rw": "reweighted ATT",
                    "ra": "regression-adjusted ATT",
                }[target],
                "resolved_small_sample_correction": correction,
                "pre_aggregations": self.pre_aggregations,
                "post_aggregations": self.post_aggregations,
                "primary_post_aggregation": self.post_aggregations[0],
                "aggregation_inference": "conditional_on_realized_design",
            }
        )
        # Give every public scalar an explicit window and weighting rule.  The
        # pooled regression remains the default for diff-diff compatibility.
        if not result.scalars.empty:
            public = result.scalars.copy()
            public["term"] = public.apply(
                lambda row: f"{row['window']} [{row['aggregation']}]", axis=1
            )
            order = pd.Categorical(
                public["window"], categories=["pre", "post"], ordered=True
            )
            method_order = pd.Categorical(
                public["aggregation"],
                categories=["pooled", "equal_horizon", "event_weighted"],
                ordered=True,
            )
            result.scalars = public.assign(
                _order=order, _method_order=method_order
            ).sort_values(
                ["_order", "_method_order"]
            ).drop(columns=["_order", "_method_order"]).reset_index(drop=True)
        if only_event:
            result.scalars = result.scalars.iloc[0:0].copy()
        if only_pooled:
            result.event_study = result.event_study.iloc[0:0].copy()
        self.results_ = result
        self.is_fitted_ = True
        return result

    def get_params(self, deep: bool = True) -> dict:
        del deep
        names = (
            "pre_window", "post_window", "control_group", "reweight",
            "no_composition", "pmd", "alpha", "cluster",
            "rank_deficient_action", "non_absorbing", "stabilization_window",
            "df_convention", "inference", "small_sample_correction",
            "bootstrap_weights", "n_bootstrap", "seed", "history_policy",
            "left_censoring", "pre_aggregation", "post_aggregation",
        )
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        unknown = sorted(set(params) - set(self.get_params()))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {unknown}")
        merged = self.get_params()
        merged.update(params)
        probe = type(self)(**merged)
        self.__dict__.update(probe.__dict__)
        return self
