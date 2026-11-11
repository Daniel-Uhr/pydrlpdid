"""Plot the common-sample LPDID and DRLPDID event paths for the received application."""

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results" / "cengiz_qje" / "received_v080"
DESTINATION = Path(
    os.environ.get(
        "PYDRLPDID_MINWAGE_OUTPUT_DIR",
        SOURCE / "figures",
    )
).expanduser().resolve()


def draw_panel(axis, path: Path, title: str) -> None:
    frame = pd.read_csv(path)
    styles = {
        "LPDID-RW+X1": ("#1f4e79", "o", "LPDID-RW + X1"),
        "DRLPDID-DR-IPT|X1": ("#b23a2b", "s", "DRLPDID-DR-IPT + X1"),
    }
    for estimator, (color, marker, label) in styles.items():
        part = frame.loc[frame["estimator"].eq(estimator)].sort_values("horizon")
        axis.fill_between(
            part["horizon"], part["ci_lower"], part["ci_upper"],
            color=color, alpha=0.10, linewidth=0,
        )
        axis.plot(
            part["horizon"], part["estimate"], color=color, marker=marker,
            markersize=3.5, linewidth=1.4, label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(-0.5, color="0.45", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Event horizon")
    axis.grid(axis="y", color="0.9", linewidth=0.6)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    draw_panel(
        axes[0], SOURCE / "state_quarter_event_study.csv",
        "State-quarter panel",
    )
    draw_panel(
        axes[1], SOURCE / "county_year_event_study.csv",
        "County-year panel; inference clustered by state",
    )
    axes[0].set_ylabel("Effect on 100 x log restaurant employment")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Minimum-wage switch-ins: common-sample event paths")
    figure.savefig(DESTINATION / "lpdid_dript_event_paths.png", dpi=220)
    figure.savefig(DESTINATION / "lpdid_dript_event_paths.pdf")


if __name__ == "__main__":
    main()
