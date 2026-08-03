"""Panel data utilities for LP-DiD estimators.

Provides panel preparation (``prepare_panel``), local comparison-stack
construction (``build_local_sample``), clean-comparison-support indicators
(``precompute_ccs``), and assorted helpers used by the estimator classes.

Clean-control conditions implemented
-------------------------------------
``'not_yet_treated'``
    Absorbing treatment. Controls: :math:`D_{i,t+h}=0`.
    (Dube et al. 2025, eq. 8.)
``'never_treated'``
    Absorbing treatment. Controls: units with :math:`p_i = \\infty` only.
``'first_entry'``
    Non-absorbing treatment. Treated: first-time entrants at *t* that stay
    treated through :math:`t+h`. Controls: units untreated at :math:`t+h`.
    (Dube et al. 2025, Section 4.2.2, eq. 12.)
``'stabilized'``
    Non-absorbing treatment with effect-stabilization horizon *L*. Treated:
    entered treatment at *t*, no prior change within *L* periods. Controls:
    no treatment-status change within :math:`[-h, L]`.
    (Dube et al. 2025, Section 4.2.3, eq. 13.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import warnings
import math

import numpy as np
import pandas as pd

from ._errors import PanelValidationError, TreatmentValidationError

BasePeriod = Union[int, List[int], Tuple[int, ...], str]

# ---------------------------------------------------------------------------
# Basic panel helpers
# ---------------------------------------------------------------------------

def check_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_panel(df: pd.DataFrame, unit: str, time: str) -> pd.DataFrame:
    out = df.sort_values([unit, time]).reset_index(drop=True).copy()
    out.attrs.update(getattr(df, "attrs", {}))
    return out


def _calendar_shift(df: pd.DataFrame, unit: str, col: str, offset: int) -> pd.Series:
    """Return ``col`` evaluated at calendar time ``t + offset``.

    Prepared panels carry the time-column name in ``DataFrame.attrs``.  The
    lookup therefore respects gaps: if the exact calendar period is absent,
    the shifted value is missing rather than the next available row.
    """
    time = df.attrs.get("_panel_time")
    if time is None or time not in df.columns:
        # Backward-compatible fallback for external helper use.
        return df.groupby(unit, sort=False)[col].shift(-offset)
    keys = pd.MultiIndex.from_arrays([df[unit].to_numpy(), df[time].to_numpy()])
    if keys.has_duplicates:
        raise PanelValidationError(f"Duplicate ({unit}, {time}) rows prevent calendar shifts.")
    values = pd.Series(df[col].to_numpy(), index=keys)
    target = pd.MultiIndex.from_arrays([
        df[unit].to_numpy(),
        pd.to_numeric(df[time], errors="coerce").to_numpy() + offset,
    ])
    return pd.Series(values.reindex(target).to_numpy(), index=df.index)


def lag(df: pd.DataFrame, unit: str, col: str, h: int) -> pd.Series:
    return _calendar_shift(df, unit, col, -int(h))


def lead(df: pd.DataFrame, unit: str, col: str, h: int) -> pd.Series:
    return _calendar_shift(df, unit, col, int(h))


def is_binary(series: pd.Series) -> bool:
    vals = set(pd.Series(series.dropna().unique()).tolist())
    return vals.issubset({0, 1})


def assert_absorbing(df: pd.DataFrame, unit: str, treat: str) -> None:
    """Raise ValueError if treatment ever switches from 1 → 0 within a unit."""
    if (df.groupby(unit, sort=False)[treat].diff() < 0).any():
        raise ValueError(
            f"Treatment column '{treat}' is not absorbing (switches 1→0). "
            "Set nonabsorbing=True to allow non-absorbing treatment."
        )


def _norm_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def z_crit(alpha: float) -> float:
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def p_value_two_sided(t_stat: float) -> float:
    return float(2.0 * (1.0 - _norm_cdf(abs(float(t_stat)))))


# ---------------------------------------------------------------------------
# base_period coercion
# ---------------------------------------------------------------------------

def coerce_base_period(base_period: BasePeriod) -> BasePeriod:
    """Validate and normalize *base_period*.

    Accepted values
    ---------------
    ``int``
        Negative integer, e.g. ``-1``.
    ``list`` / ``tuple``
        Collection of negative integers; returned as sorted list.
    ``'all_pre'``
        Average of all pre-treatment outcomes.
    """
    if isinstance(base_period, int):
        if base_period >= 0:
            raise ValueError("base_period integer must be negative, e.g. -1.")
        return int(base_period)
    if isinstance(base_period, (list, tuple)):
        vals = [int(v) for v in base_period]
        if not vals or any(v >= 0 for v in vals):
            raise ValueError(
                "base_period list/tuple must contain only negative integers."
            )
        return sorted(set(vals))
    if isinstance(base_period, str):
        if base_period != "all_pre":
            raise ValueError("base_period string must be 'all_pre'.")
        return "all_pre"
    raise ValueError(f"Invalid base_period: {base_period!r}.")


# ---------------------------------------------------------------------------
# Panel preparation
# ---------------------------------------------------------------------------

def _derive_first_treat(
    df: pd.DataFrame, unit: str, time: str, treatment: str
) -> pd.Series:
    first = (
        df.loc[df[treatment] == 1, [unit, time]]
        .groupby(unit, sort=False)[time]
        .min()
    )
    mapped = df[unit].map(first).fillna(0)
    return pd.to_numeric(mapped, errors="coerce").fillna(0)


def prepare_panel(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: Optional[str],
    treatment: Optional[str],
    nonabsorbing: bool = False,
    n_lagged_outcome_changes: int = 0,
    *,
    left_censoring: str = "error",
    time_policy: str = "calendar",
) -> pd.DataFrame:
    """Validate and augment a long panel using exact calendar-time shifts.

    ``time_policy='calendar'`` permits gaps but never substitutes a later row
    for a missing calendar period. ``'strict'`` rejects any within-unit gap.
    ``left_censoring`` is ``'error'`` (default) or ``'drop'``.
    """
    if first_treat is None and treatment is None:
        raise TreatmentValidationError("Provide either `first_treat` or `treatment`.")
    if left_censoring not in {"error", "drop"}:
        raise ValueError("left_censoring must be 'error' or 'drop'.")
    if time_policy not in {"calendar", "strict"}:
        raise ValueError("time_policy must be 'calendar' or 'strict'.")
    req = [outcome, unit, time] + ([first_treat] if first_treat else []) + ([treatment] if treatment else [])
    check_columns(data, req)
    if data[[unit, time]].isna().any().any():
        raise PanelValidationError("unit and time must be non-missing.")
    if data.duplicated([unit, time]).any():
        bad = data.loc[data.duplicated([unit, time], keep=False), [unit, time]].head(10)
        raise PanelValidationError(
            f"Duplicate ({unit}, {time}) rows are not allowed. Examples: {bad.to_dict('records')}"
        )

    df = sort_panel(data.copy(), unit, time)
    df[time] = pd.to_numeric(df[time], errors="coerce")
    if df[time].isna().any():
        raise PanelValidationError("time must be numeric and finite.")
    if not np.all(np.isfinite(df[time])):
        raise PanelValidationError("time must be finite.")
    df.attrs["_panel_unit"] = unit
    df.attrs["_panel_time"] = time
    df["_observed_row"] = 1

    if time_policy == "strict":
        diffs = df.groupby(unit, sort=False)[time].diff().dropna()
        if (diffs != 1).any():
            examples = df.loc[df.groupby(unit, sort=False)[time].diff().fillna(1) != 1, [unit, time]].head(10)
            raise PanelValidationError(
                "time_policy='strict' requires consecutive unit-spaced periods; "
                f"gap examples: {examples.to_dict('records')}"
            )

    raw_treat = None
    if treatment is not None:
        raw_treat = pd.to_numeric(df[treatment], errors="coerce")
        if not is_binary(raw_treat):
            raise TreatmentValidationError("treatment must contain only 0/1 or missing values.")
        df["_treat_obs"] = raw_treat.notna().astype(int)
    else:
        df["_treat_obs"] = 1

    # Detect units whose treatment is already on in their first observed row.
    if treatment is not None:
        first_rows = df.groupby(unit, sort=False).head(1)
        already_on = first_rows.loc[pd.to_numeric(first_rows[treatment], errors="coerce") == 1, unit].tolist()
        if already_on and first_treat is None:
            msg = (f"{len(already_on)} unit(s) are treated in their first observed period, "
                   "so first treatment is left-censored. Supply `first_treat` or use left_censoring='drop'.")
            if left_censoring == "error":
                raise TreatmentValidationError(msg)
            warnings.warn(msg + " Dropping them.", stacklevel=2)
            df = df.loc[~df[unit].isin(already_on)].copy()
            raw_treat = pd.to_numeric(df[treatment], errors="coerce")

    if first_treat is None:
        df["_first_treat_internal"] = _derive_first_treat(df, unit, time, treatment)
        first_treat = "_first_treat_internal"

    ft = pd.to_numeric(df[first_treat], errors="coerce").fillna(0)
    if (ft < 0).any():
        raise TreatmentValidationError("first_treat must be 0 (never) or a non-negative period.")
    # first_treat must be constant within unit.
    nunique = ft.groupby(df[unit], sort=False).nunique(dropna=False)
    if (nunique > 1).any():
        raise TreatmentValidationError("first_treat must be constant within unit.")
    df["_first_treat"] = ft

    first_obs = df.groupby(unit, sort=False)[time].transform("min")
    ft_unit = df.groupby(unit, sort=False)["_first_treat"].transform("first")
    left_units = df.loc[(ft_unit > 0) & (ft_unit < first_obs), unit].drop_duplicates().tolist()
    if left_units:
        msg = f"{len(left_units)} unit(s) have first_treat before their first observed period."
        if left_censoring == "error":
            raise TreatmentValidationError(msg)
        warnings.warn(msg + " Dropping them.", stacklevel=2)
        df = df.loc[~df[unit].isin(left_units)].copy()
        if treatment is not None:
            raw_treat = pd.to_numeric(df[treatment], errors="coerce")

    df = sort_panel(df, unit, time)
    df.attrs["_panel_unit"] = unit
    df.attrs["_panel_time"] = time
    df["_first_obs_time"] = df.groupby(unit, sort=False)[time].transform("min")
    df["_left_censored"] = 0

    if nonabsorbing:
        if treatment is None:
            raise TreatmentValidationError("Non-absorbing designs require an observed treatment path.")
        df["_treat"] = pd.to_numeric(df[treatment], errors="coerce")
        if first_treat is not None:
            observed_first = _derive_first_treat(df, unit, time, treatment)
            # only validate positive supplied cohorts against observed first onset
            bad = (df["_first_treat"] > 0) & (observed_first > 0) & (df["_first_treat"] != observed_first)
            if bad.any():
                raise TreatmentValidationError("first_treat is inconsistent with the observed first treatment onset.")
        prev = lag(df, unit, "_treat", 1)
        df["D_treat"] = df["_treat"] - prev
    else:
        expected = ((df["_first_treat"] > 0) & (df[time] >= df["_first_treat"])).astype(int)
        if treatment is not None:
            observed = pd.to_numeric(df[treatment], errors="coerce")
            bad = observed.notna() & (observed.astype(float) != expected.astype(float))
            if bad.any():
                ex = df.loc[bad, [unit, time, first_treat, treatment]].head(10)
                raise TreatmentValidationError(
                    "treatment is inconsistent with absorbing first_treat. "
                    f"Examples: {ex.to_dict('records')}"
                )
        df["_treat"] = expected
        assert_absorbing(df, unit, "_treat")
        df["D_treat"] = ((df["_first_treat"] > 0) & (df[time] == df["_first_treat"])).astype(int)

    df["dy"] = df[outcome] - lag(df, unit, outcome, 1)
    for k in range(1, int(max(n_lagged_outcome_changes, 0)) + 1):
        df[f"ldy{k}"] = lag(df, unit, "dy", k)

    df["_never_treated"] = (df["_first_treat"] == 0).astype(int)
    df["_treated_ever"] = (df["_first_treat"] > 0).astype(int)
    df["_exposure_age"] = np.where(
        df["_first_treat"] > 0, df[time] - df["_first_treat"], np.nan
    )
    df["rel_time"] = df["_exposure_age"]
    return df


# ---------------------------------------------------------------------------
# base_period helpers
# ---------------------------------------------------------------------------

def _all_pre_base(df: pd.DataFrame, outcome: str, unit: str) -> pd.Series:
    obsnum = df.groupby(unit, sort=False).cumcount() + 1
    cumy = df.groupby(unit, sort=False)[outcome].cumsum()
    out = lag(pd.DataFrame({"_cumy": cumy, unit: df[unit]}), unit, "_cumy", 1) / (obsnum - 1)
    out = pd.Series(out, index=df.index)
    out.loc[obsnum <= 1] = np.nan
    return out


def compute_base_series(
    df: pd.DataFrame, outcome: str, unit: str, base_period: BasePeriod
) -> pd.Series:
    if isinstance(base_period, int):
        return lag(df, unit, outcome, abs(base_period))
    if isinstance(base_period, list):
        mats = np.column_stack([lag(df, unit, outcome, abs(v)) for v in base_period])
        complete = np.all(np.isfinite(mats), axis=1)
        out = np.full(len(df), np.nan, dtype=float)
        out[complete] = mats[complete].mean(axis=1)
        return pd.Series(out, index=df.index)
    return _all_pre_base(df, outcome, unit)


def compute_long_difference(
    df: pd.DataFrame, outcome: str, unit: str, h: int, base_period: BasePeriod
) -> pd.Series:
    base = compute_base_series(df, outcome, unit, base_period)
    if h >= 0:
        return lead(df, unit, outcome, h) - base
    return lag(df, unit, outcome, abs(h)) - base


# ---------------------------------------------------------------------------
# Window inference
# ---------------------------------------------------------------------------

def infer_windows(
    df: pd.DataFrame,
    time: str,
    base_period: BasePeriod,
    max_pre: Optional[int] = None,
    max_post: Optional[int] = None,
) -> Tuple[int, int]:
    rel = pd.to_numeric(df["rel_time"], errors="coerce")
    max_post_data = int(rel.max()) if np.isfinite(rel.max()) else 0
    neg_rel = rel.loc[np.isfinite(rel) & (rel < 0)]
    max_pre_data = int(abs(neg_rel.min())) if not neg_rel.empty else 0

    if isinstance(base_period, int):
        max_pre_data = max(max_pre_data, abs(base_period))
    elif isinstance(base_period, list):
        max_pre_data = max(max_pre_data, max(abs(v) for v in base_period))
    else:
        max_pre_data = max(max_pre_data, 1)

    final_pre = max_pre_data if max_pre is None else min(int(max_pre), max_pre_data)
    final_post = max_post_data if max_post is None else min(int(max_post), max_post_data)
    return int(max(final_pre, 0)), int(max(final_post, 0))


# ---------------------------------------------------------------------------
# CCS indicators for the stabilized condition (Section 4.2.3)
# ---------------------------------------------------------------------------

def precompute_ccs(
    df: pd.DataFrame, unit: str, effect_stabilization: int,
    max_pre: int, max_post: int,
) -> pd.DataFrame:
    """Precompute clean-comparison-support indicators horizon by horizon.

    Used by ``clean_control='stabilized'`` to implement eq. (13) of
    Dube et al. (2025, Section 4.2.3).
    """
    out = df.copy()
    out.attrs.update(df.attrs)
    ccs0 = pd.Series(True, index=out.index)
    for k in range(1, effect_stabilization + 1):
        z = lag(out, unit, "D_treat", k)
        ccs0 = ccs0 & z.notna() & (z.abs() != 1)
    out["CCS_0"] = ccs0.astype(int)

    for h in range(1, max_post + 1):
        prev = out[f"CCS_{h-1}"] == 1
        z = lead(out, unit, "D_treat", h)
        no_future = z.notna() & (z.abs() != 1)
        out[f"CCS_{h}"] = (prev & no_future).astype(int)

    out["CCS_m1"] = out["CCS_0"]
    for h in range(2, max_pre + 1):
        prev = out[f"CCS_m{h-1}"] == 1
        lag_prev = lag(out, unit, f"CCS_m{h-1}", 1) == 1
        out[f"CCS_m{h}"] = (prev & lag_prev).astype(int)
    return out


# ---------------------------------------------------------------------------
# Local comparison stack builder
# ---------------------------------------------------------------------------

def build_local_sample(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    h: int,
    base_period: BasePeriod,
    clean_control: str,
    effect_stabilization: Optional[int],
    extra_cols: List[str],
    user_covariates: Optional[List[str]] = None,
    lag_covariates: bool = False,
    fixed_composition_H: Optional[int] = None,
    control_pool: str = "stabilized_all",
    switch_in: str = "sustained",
    control_window: str = "horizon",
) -> pd.DataFrame:
    """Build the horizon-h clean local comparison stack :math:`S_h`.

    Implements the sample restrictions for all four supported clean-control
    conditions (Dube et al. 2025, Sections 3.1, 4.2.2, 4.2.3).

    When ``fixed_composition_H`` is not ``None`` it activates the fixed-
    composition sample of Dube et al. (2025, Section 3.6): the treated and
    control sets are held constant across horizons. Concretely, controls are
    required to be untreated at ``t + H`` (rather than per-horizon ``t + h``),
    and treatment events occurring after ``T - H`` are dropped, so the same
    units populate every horizon's stack (``H = fixed_composition_H``). The
    per-horizon dependent variable is still the horizon-``h`` long difference.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared panel (output of :func:`prepare_panel`).
    outcome, unit, time : str
    h : int
        Event-time horizon (negative = pre-treatment).
    base_period : BasePeriod
    clean_control : {'not_yet_treated', 'never_treated', 'first_entry', \
'stabilized'}
    effect_stabilization : int or None
        Required when ``clean_control='stabilized'``.
    extra_cols : list of str
        Additional columns that must be non-missing.

    Returns
    -------
    pd.DataFrame
        Stack with columns ``D_local`` (1 = treated, 0 = control) and
        ``outcome_local`` (long-differenced outcome).
    """
    s = df.copy()
    s["outcome_local"] = compute_long_difference(s, outcome, unit, h, base_period)

    # --- fixed-composition pre-computation (Section 3.6) ----------------
    # When active, fix the unit set across horizons: controls untreated at
    # t+H and treated events restricted to t <= T - H. ``not_late`` flags
    # treated entries that can be followed the full H periods.
    fixed_comp = fixed_composition_H is not None
    if fixed_comp:
        H = int(fixed_composition_H)
        t_max = s[time].max()
        not_late = s[time] <= (t_max - H)
        # untreated at t+H (absorbing-style fixed control membership)
        ctrl_fixed = lead(s, unit, "_treat", H).fillna(1.0) == 0

    # --- treated mask --------------------------------------------------
    if clean_control in {"not_yet_treated", "never_treated"}:
        # Absorbing: newly treated at t (ΔD = 1)
        treat_mask = s["D_treat"] == 1

    elif clean_control == "first_entry":
        # Non-absorbing, Section 4.2.2 (eq. 12):
        # ΔD_it = 1  AND  D_{i,t-j} = 0 for j ≥ 1 (first-time entry)
        # AND (for h≥0) stays treated through t+h
        newly_treated = s["D_treat"] == 1
        first_time = s["_first_treat"] == s[time]  # never treated before t
        if h >= 0:
            stays_treated = lead(s, unit, "_treat", h).fillna(0) == 1
        else:
            stays_treated = pd.Series(True, index=s.index)
        treat_mask = newly_treated & first_time & stays_treated

    elif clean_control == "stabilized":
        # Non-absorbing, Section 4.2.3 (eq. 13):
        # Treated: ΔD_it = 1 AND no treatment change in [t-L, t-1].
        # The clean pre-window is the CCS_0 indicator, which must be applied
        # to the treated as well (eq. 13 requires D_{i,t-j}=0 for 1<=j<=L on
        # the switching unit), matching the Stata `tdemoc==1 & CCS_0==1`
        # condition in figure_4.do. Earlier versions omitted this restriction.
        if effect_stabilization is None:
            raise ValueError(
                "clean_control='stabilized' requires effect_stabilization."
            )
        # For post-treatment horizons, the switching event itself must have a
        # clean L-period pre-history (CCS_0).  For placebo leads h=-j, Dube et
        # al.'s CCC1 and CCC2 specifications both restrict the event and the
        # controls by CCS_mj; using CCS_0 for leads would admit events whose
        # required pre-history is incomplete or contaminated.
        treated_ccs_col = "CCS_0" if h >= 0 else f"CCS_m{abs(h)}"
        treat_mask = (s["D_treat"] == 1) & (s[treated_ccs_col] == 1)
        # Exclude phantom switches created by missing raw treatment.
        if "_treat_obs" in s.columns:
            treat_mask = treat_mask & (s["_treat_obs"] == 1)
            # A genuine onset is an observed 0->1 transition: require the
            # previous period observed and untreated (matches Stata
            # `dem==1 & l.dem==0`, which excludes first-observation / post-gap
            # rows where the lag is missing and would otherwise be counted as
            # spurious democratization events).
            lag_treat_t = lag(s, unit, "_treat", 1)
            lag_obs_t = lag(s, unit, "_treat_obs", 1)
            treat_mask = treat_mask & (lag_treat_t == 0) & (lag_obs_t == 1)
        # Switch-in estimand (eq. 13 vs application):
        #   'sustained' -> eq. (13) formal target tau_h^{g,n}: the treated must
        #       stay treated at EVERY period from the event through t+h
        #       (D_{i,t+j}=1 for 0<=j<=h). This is the sustained switch-in /
        #       adopt-and-keep effect that the paper's theory defines.
        #   'onset'     -> figure_4.do application: only the clean onset is
        #       required; reverters contribute (composition frozen at entry).
        # The restriction binds only for post-horizons (h>=0); pre-period
        # placebo rows use the clean-event set unchanged.
        if switch_in == "sustained" and h >= 0:
            stay_on = pd.Series(True, index=s.index)
            for j in range(0, h + 1):
                stay_on = stay_on & (lead(s, unit, "_treat", j).fillna(0) == 1)
                if "_treat_obs" in s.columns:
                    stay_on = stay_on & (lead(s, unit, "_treat_obs", j).fillna(0) == 1)
            treat_mask = treat_mask & stay_on

    else:
        raise ValueError(
            f"Unknown clean_control='{clean_control}'. "
            "Must be one of {{'not_yet_treated', 'never_treated', "
            "'first_entry', 'stabilized'}}."
        )

    # Fixed composition: drop treated events that cannot be followed H periods
    if fixed_comp:
        treat_mask = treat_mask & not_late

    # --- control mask --------------------------------------------------
    if clean_control == "never_treated":
        ctrl_mask = s["_never_treated"] == 1

    elif clean_control in {"not_yet_treated", "first_entry"}:
        if fixed_comp:
            # Fixed control membership: untreated at t+H for every horizon
            ctrl_mask = ctrl_fixed
        elif h >= 0:
            # Controls: not yet treated at t+h
            ctrl_mask = lead(s, unit, "_treat", h).fillna(1.0) == 0
        else:
            # Pre-periods: not treated at t
            ctrl_mask = s["_treat"] == 0

    elif clean_control == "stabilized":
        if effect_stabilization is None:
            raise ValueError(
                "clean_control='stabilized' requires effect_stabilization."
            )
        if h < 0:
            # Both empirical CCC variants use CCS_mj for a lead h=-j.  The
            # distinction between entry- and horizon-clean controls concerns
            # post-treatment horizons only (Dube et al., figure_4.do).
            col = "CCS_m1" if h == -1 else f"CCS_m{abs(h)}"
        elif fixed_comp:
            # Fixed-composition analog: hold the clean-control window at the
            # maximum horizon H so the same units serve as controls at every h.
            col = f"CCS_{H}"
        elif control_window == "entry":
            # CCC 1: no treatment change between t-L and t; future treatment
            # status is not conditioned on for post-treatment horizons.
            col = "CCS_0"
        else:
            # CCC 2 / equation (13): controls remain clean through t+h.
            col = f"CCS_{h}"
        stable_mask = s[col] == 1
        # Control = any unit with no treatment ONSET at t and clean CCS history.
        # Matches the Stata condition exactly: D.treat==0 & CCS_j==1
        # (Dube et al. 2025, eq. 13; appendix_sim_LPDiD_estimation.do lines 31-35).
        # The CCS indicator already guarantees the required stability window;
        # an additional exposure-age filter is not needed and not in the paper.
        ctrl_mask = (s["D_treat"] == 0) & stable_mask
        # Exclude phantom control rows from missing raw treatment.
        if "_treat_obs" in s.columns:
            ctrl_mask = ctrl_mask & (s["_treat_obs"] == 1)
        # Control-pool choice (non-absorbing stabilized only):
        #   'stabilized_all'  -> eq. (13): any non-switching unit in a clean
        #       window, INCLUDING stable always-treated units (their long
        #       difference is counterfactual-clean under effect stabilization).
        #   'untreated_only'  -> empirical application (figure_4.do): restrict
        #       controls to currently-untreated stayers (D_{it}=0), matching the
        #       Stata `tdemoc==0` (dem==0 & l.dem==0) clean-control definition.
        if control_pool == "untreated_only":
            # Stayers-at-0 only: currently untreated AND untreated last period
            # with the lag observed (matches Stata `dem==0 & l.dem==0`, which
            # excludes first-observation / post-gap rows where the lag is
            # missing). Using the lagged level rather than just the current
            # level removes those boundary rows from the control pool.
            lag_treat = lag(s, unit, "_treat", 1)
            ctrl_mask = ctrl_mask & (s["_treat"] == 0) & (lag_treat == 0)
            if "_treat_obs" in s.columns:
                lag_obs = lag(s, unit, "_treat_obs", 1)
                ctrl_mask = ctrl_mask & (lag_obs == 1)

    # --- lag user covariates to t-1 (predetermined) --------------------
    # Computed on the full panel BEFORE any subsetting.
    # ldy* columns are already lags and must NOT be lagged again.
    # We overwrite the column in-place so downstream formulas using the
    # original column name automatically receive X_{i,t-1}.
    if lag_covariates and user_covariates:
        for c in user_covariates:
            if c in s.columns:
                s[c] = lag(s, unit, c, 1)

    # --- assemble stack ------------------------------------------------
    s = s.loc[treat_mask | ctrl_mask].copy()
    s["D_local"] = treat_mask.loc[s.index].astype(int)

    needed = [unit, time, "D_local", "outcome_local"] + list(extra_cols)
    s = s.dropna(subset=[c for c in needed if c in s.columns]).copy()
    s = s.loc[np.isfinite(s["outcome_local"])].copy()
    if s.empty:
        return s.reset_index(drop=True)

    # Exact-replication convention (Dube et al. figure_3 / figure_4 reghdfe):
    # do NOT drop time cells that contain only treated units. reghdfe keeps
    # them — their time fixed effect absorbs the treated observations, so they
    # contribute nothing to the treatment coefficient, while still informing the
    # covariate nuisance coefficients. Dropping these cells previously made
    # covariate specifications diverge from reghdfe at long horizons (where the
    # not-yet-treated pool thins and control-less cells appear). We only require
    # that at least one control observation exists overall for identification.
    if (s["D_local"] == 0).sum() == 0:
        return s.iloc[0:0].reset_index(drop=True)
    return s.reset_index(drop=True)
