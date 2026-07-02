"""M2 demo: cumulative time delta plus channel overlays for two laps.

The headline figure of the project: where one lap gains or loses time
against another, with the speed, throttle and brake traces below to
explain why.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file, no display needed

from apextrace.loaders.fastf1_loader import load_fastf1_lap
from apextrace.plots import plot_delta_overlay

OUT_FILE = Path(__file__).resolve().parent / "delta_overlay.png"


def main() -> None:
    lap_a = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "NOR")
    lap_b = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "LEC")

    fig, _ = plot_delta_overlay(lap_a, lap_b)
    fig.savefig(OUT_FILE, dpi=150)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
