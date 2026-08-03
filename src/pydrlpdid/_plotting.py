from __future__ import annotations

from typing import Optional

import numpy as np


def plot_event_study(results, ax=None, title: str = "Event Study", xlabel: str = "Event time", ylabel: str = "ATT"):
    """Plot an event-study path with pointwise or simultaneous confidence intervals."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc

    df = results.event_study.sort_values("horizon").copy()
    if df.empty:
        raise ValueError("results does not contain event-study estimates.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x = df["horizon"].to_numpy(dtype=float)
    y = df["estimate"].to_numpy(dtype=float)
    if {"sim_ci_lower", "sim_ci_upper"}.issubset(df.columns) and np.isfinite(df["sim_ci_lower"]).any():
        lo = df["sim_ci_lower"].to_numpy(dtype=float)
        hi = df["sim_ci_upper"].to_numpy(dtype=float)
    else:
        lo = df["ci_lower"].to_numpy(dtype=float)
        hi = df["ci_upper"].to_numpy(dtype=float)
    err = np.vstack([y - lo, hi - y])

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(-1.0, linestyle=":", linewidth=1)
    ax.errorbar(x, y, yerr=err, fmt="o", capsize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return ax
