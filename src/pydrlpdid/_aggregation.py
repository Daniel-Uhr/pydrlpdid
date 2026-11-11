"""Shared aggregation rules for LP-DiD event-study paths.

The package distinguishes a pooled-outcome regression from linear summaries
of separately estimated horizon effects.  The latter are combined with their
cluster influence functions, so their standard errors retain cross-horizon
dependence.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ._inference import se_from_influence
from ._panel_utils import analytic_cluster_inference


AGGREGATION_METHODS = ("pooled", "equal_horizon", "event_weighted")

_ALIASES = {
    "mean_horizons": "equal_horizon",
    "mean_horizon": "equal_horizon",
    "equal_horizons": "equal_horizon",
    "treated_event": "event_weighted",
    "treated_events": "event_weighted",
    "treated_observation": "event_weighted",
    "treated_observations": "event_weighted",
}


def normalize_aggregations(
    value,
    *,
    name: str,
    default: Optional[tuple[str, ...]] = ("pooled",),
) -> tuple[str, ...]:
    """Validate and canonicalize one or more aggregation methods."""
    if value is None:
        return tuple(default or ())
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Iterable):
        raw = list(value)
    else:
        raise ValueError(
            f"{name} must be an aggregation name, a sequence of names, or 'all'."
        )
    if not raw:
        raise ValueError(f"{name} cannot be empty.")

    resolved: list[str] = []
    for item in raw:
        key = str(item).lower().strip().replace("-", "_")
        key = _ALIASES.get(key, key)
        if key == "all":
            candidates = AGGREGATION_METHODS
        elif key in AGGREGATION_METHODS:
            candidates = (key,)
        else:
            allowed = ", ".join((*AGGREGATION_METHODS, "all"))
            raise ValueError(f"{name} must contain only: {allowed}.")
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return tuple(resolved)


def horizon_aggregate(
    event_df: pd.DataFrame,
    psi_by_h: Dict[int, pd.Series],
    clusters_by_h: Dict[int, pd.Index],
    cluster_universe: pd.Index,
    horizons: list[int],
    *,
    window: str,
    method: str,
    alpha: float,
    compute_influence: bool,
) -> tuple[Optional[dict], Optional[pd.Series], list[int]]:
    """Aggregate horizon-specific effects with explicit, auditable weights.

    ``equal_horizon`` gives every supported horizon the same weight.
    ``event_weighted`` gives every treated event-horizon observation the same
    weight by using ``n_event_rows`` as the horizon weight.  Analytic inference
    is conditional on the realized treatment design; a full cluster bootstrap
    re-estimates the weights in each resample.
    """
    method = str(method)
    if method not in {"equal_horizon", "event_weighted"}:
        raise ValueError("Linear horizon aggregation must be equal_horizon or event_weighted.")
    horizons = [int(h) for h in horizons]
    by_h = event_df.set_index("horizon")
    missing = [
        h for h in horizons
        if h not in by_h.index or not np.isfinite(by_h.loc[h, "estimate"])
    ]
    if missing or not horizons:
        return None, None, missing

    if method == "equal_horizon":
        raw_weights = np.ones(len(horizons), dtype=float)
        weight_basis = "equal_supported_horizons"
    else:
        if "n_event_rows" not in by_h.columns:
            return None, None, list(horizons)
        raw_weights = by_h.loc[horizons, "n_event_rows"].to_numpy(dtype=float)
        weight_basis = "realized_treated_event_rows"
    if (
        raw_weights.size == 0
        or not np.isfinite(raw_weights).all()
        or float(raw_weights.sum()) <= 0
    ):
        return None, None, list(horizons)
    weights = raw_weights / raw_weights.sum()
    total_event_rows = (
        int(by_h.loc[horizons, "n_event_rows"].sum())
        if "n_event_rows" in by_h.columns else 0
    )

    estimates = by_h.loc[horizons, "estimate"].to_numpy(dtype=float)
    estimate = float(np.dot(weights, estimates))
    psi = None
    se = np.nan
    if compute_influence:
        matrix = np.column_stack([
            psi_by_h[h].reindex(cluster_universe, fill_value=0.0).to_numpy(dtype=float)
            for h in horizons
        ])
        influence = matrix @ weights
        se = se_from_influence(influence)
        psi = pd.Series(influence, index=cluster_universe, dtype=float)

    scalar_clusters: set = set()
    for h in horizons:
        scalar_clusters.update(clusters_by_h.get(h, pd.Index([])).tolist())
    term = f"{window} [{method}]"
    row = {
        "term": term,
        "window": window,
        "aggregation": method,
        "estimate": estimate,
        "se": se,
        **analytic_cluster_inference(
            estimate, se, alpha, int(len(scalar_clusters))
        ),
        "n_horizons": int(len(horizons)),
        "horizons": tuple(horizons),
        "weights": tuple(float(weight) for weight in weights),
        "weight_basis": weight_basis,
        "n_event_rows": total_event_rows,
    }
    return row, psi, []
