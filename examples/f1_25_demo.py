"""M4 demo: the whole F1 25 path, without F1 25 and without a network.

The milestone claims a second telemetry source costs one loader and
nothing else. This runs that claim end to end on synthetic UDP packets:
real byte layouts, real frame reassembly, real lap cutting, then the
delta engine and the corner report that were written for F1 timing
data and have not been modified.

Two laps of a made-up circuit, identical but for one corner taken
slower. The kinematics are exact by construction, so the numbers below
have known right answers, printed alongside.

    uv run python examples/f1_25_demo.py
    uv run python examples/f1_25_demo.py --capture demo.f1cap

With --capture the session is also written as a capture file, which
the app can replay as if it were live:

    uv run apextrace serve --replay demo.f1cap
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file, no display needed

import numpy as np

from apextrace.delta import delta_time
from apextrace.loaders.f1_25 import load_f1_25_laps, write_capture
from apextrace.loaders.f1_25.synthetic import synthetic_session
from apextrace.plots import plot_delta_overlay
from apextrace.report import corner_report

OUT_FILE = Path(__file__).resolve().parent / "f1_25_demo.png"

LAP_LENGTH = 4200.0
BASE_SPEED = 290.0
RATE_HZ = 30.0

# (apex position [m], apex speed [km/h], corner width [m]). The second
# corner is where the two laps differ.
FAST_CORNERS = ((700.0, 95.0, 90.0), (1800.0, 140.0, 120.0), (3100.0, 75.0, 85.0))
SLOW_CORNERS = ((700.0, 95.0, 90.0), (1800.0, 115.0, 120.0), (3100.0, 75.0, 85.0))

FLOAT_FORMATS = {
    "apex_a_kmh": "{:8.1f}".format,
    "apex_b_kmh": "{:8.1f}".format,
    "brake_m": "{:7.1f}".format,
    "apex_m": "{:7.1f}".format,
    "power_m": "{:7.1f}".format,
    "coast_a_m": "{:8.1f}".format,
    "coast_b_m": "{:8.1f}".format,
    "entry_s": "{:+7.3f}".format,
    "exit_s": "{:+7.3f}".format,
    "total_s": "{:+7.3f}".format,
}


def speed_profile(corners):
    """Distance [m] -> speed [km/h]: Gaussian dips on a constant base."""

    def _speed(dist):
        dist = np.asarray(dist, dtype=float)
        speed = np.full(dist.shape, BASE_SPEED)
        for centre, floor, width in corners:
            speed -= (BASE_SPEED - floor) * np.exp(-(((dist - centre) / width) ** 2))
        return speed

    return _speed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="also write the session as a capture file for the app to replay",
    )
    args = parser.parse_args()

    # One session: a quick lap, the same lap with corner 2 taken slower,
    # and the start of a third so the second one gets finished.
    profiles = [
        speed_profile(FAST_CORNERS),
        speed_profile(SLOW_CORNERS),
        speed_profile(FAST_CORNERS),
        speed_profile(FAST_CORNERS),
    ]
    datagrams, exact = synthetic_session(
        profiles, lap_length=LAP_LENGTH, rate_hz=RATE_HZ
    )
    laps, skipped = load_f1_25_laps(datagrams)
    if len(laps) != 3:
        raise SystemExit(f"expected three laps, got {len(laps)} ({skipped})")

    print(
        f"\nsynthetic F1 25 session, {LAP_LENGTH:.0f} m lap, {RATE_HZ:.0f} Hz, "
        f"{len(datagrams)} UDP packets\n"
    )
    for lap, exact_time in zip(laps, exact, strict=False):
        print(
            f"{lap.label}: {len(lap.data)} grid points, "
            f"{lap.lap_time:.3f} s (exact {exact_time:.3f} s)"
        )

    lap_a, lap_b = laps[0], laps[1]
    delta = delta_time(lap_a, lap_b)
    print(
        f"\nfull-lap delta {delta.iloc[-1]:+.3f} s, "
        f"exact difference {exact[1] - exact[0]:+.3f} s"
    )

    table = corner_report(lap_a, lap_b)
    print(f"\n{lap_b.label} vs {lap_a.label}")
    print(table.to_string(index=False, formatters=FLOAT_FORMATS))
    print(
        "\nThe time is lost in corner #2 and nowhere else, which is where "
        "the two speed traces differ."
    )

    fig, _ = plot_delta_overlay(lap_a, lap_b)
    fig.savefig(OUT_FILE, dpi=150)
    print(f"\nsaved {OUT_FILE}")

    if args.capture is not None:
        # Three datagrams per frame: motion, lap data, telemetry.
        write_capture(args.capture, datagrams, rate_hz=3.0 * RATE_HZ)
        print(f"saved {args.capture}")


if __name__ == "__main__":
    main()
