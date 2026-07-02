"""M1 demo: one real F1 lap through the canonical pipeline, plotted.

Loads the fastest qualifying lap of a driver, which comes out of the loader
as a canonical Lap on a uniform distance grid, and plots speed vs distance.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file, no display needed

import matplotlib.pyplot as plt

from apextrace.loaders.fastf1_loader import load_fastf1_lap

OUT_FILE = Path(__file__).resolve().parent / "speed_trace.png"


def main() -> None:
    lap = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "NOR")

    print(f"label:    {lap.label}")
    print(f"track:    {lap.track} ({lap.car})")
    print(f"channels: {lap.channels}")
    print(f"lap time: {lap.lap_time:.3f} s (grid) ")
    print(lap.data.head())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lap.data["Distance"], lap.data["Speed"], linewidth=1.2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"{lap.label} ({lap.lap_time:.3f} s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"saved plot to {OUT_FILE}")


if __name__ == "__main__":
    main()
