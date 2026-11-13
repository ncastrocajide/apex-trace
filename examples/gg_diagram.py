"""M6 demo: the g-g diagram of a lap, from speed and position alone.

Longitudinal acceleration is the rate of change of speed; lateral is
speed squared times the curvature of the driven line. Neither needs an
accelerometer, so this works on any lap with a position channel:

    uv run python examples/gg_diagram.py monza.f1cap     # F1 25 capture
    uv run python examples/gg_diagram.py                 # synthetic session

Plotted against each other, the two accelerations trace the envelope
the car was driven to: braking at the bottom, traction at the top,
cornering at the sides. The corners of that envelope are where both are
used at once, and how full they are is the picture of trail braking and
of power applied before the steering is unwound.

These are estimates with declared smoothing (see apextrace/gforce.py):
the windows are printed with the figure because the peaks depend on
them. The synthetic session is a circle with a made-up speed trace; it
shows the method, not a car.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file, no display needed

import matplotlib.pyplot as plt
import numpy as np

from apextrace.gforce import derived_g, g_report
from apextrace.loaders.f1_25 import load_f1_25_lap, load_f1_25_laps
from apextrace.loaders.f1_25.synthetic import synthetic_session

OUT_FILE = Path(__file__).resolve().parent / "gg_diagram.png"
SPEED_WINDOW = 50.0  # m
POSITION_WINDOW = 100.0  # m


def synthetic_lap():
    def speed(dist):
        dist = np.asarray(dist, dtype=float)
        trace = np.full(dist.shape, 290.0)
        for centre, floor, width in (
            (700.0, 95.0, 90.0),
            (1800.0, 140.0, 120.0),
            (3100.0, 75.0, 85.0),
        ):
            trace -= (290.0 - floor) * np.exp(-(((dist - centre) / width) ** 2))
        return trace

    datagrams, _ = synthetic_session([speed, speed], lap_length=4200.0)
    return load_f1_25_laps(datagrams)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", help="F1 25 capture file")
    args = parser.parse_args()

    if args.path is None:
        print("no capture given: using the synthetic session")
        lap = synthetic_lap()
    else:
        lap = load_f1_25_lap(args.path)

    windows = {"speed_window": SPEED_WINDOW, "position_window": POSITION_WINDOW}
    g = derived_g(lap, **windows)
    report = g_report(lap, **windows)
    print(f"\n{lap.label}  ({lap.lap_time:.3f} s)")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    points = ax.scatter(
        g["GLat"], g["GLong"], c=lap.data["Speed"], s=6, cmap="viridis", linewidths=0
    )
    fig.colorbar(points, ax=ax, label="Speed [km/h]", shrink=0.8)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral acceleration [g]")
    ax.set_ylabel("Longitudinal acceleration [g]  (braking negative)")
    ax.set_title(
        f"{lap.label}\nderived, windows {SPEED_WINDOW:.0f} m (speed) / "
        f"{POSITION_WINDOW:.0f} m (position)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.savefig(OUT_FILE, dpi=150)
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()
