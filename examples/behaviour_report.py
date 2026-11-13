"""M6 demo: how the inputs differ, corner by corner.

The corner report says where the time went; this says what the hands
and feet did there. Run it on a capture of your own driving:

    uv run python examples/behaviour_report.py monza.f1cap

Without a file it falls back to the synthetic F1 25 session, which
shows the mechanics and nothing else: its pedal and steering traces
are fillers derived from the speed profile, not driving.

Exploratory by design. These are stated readings of the traces, not
scores: compare the same corner across two laps and ask what was done
differently.
"""

import argparse
from pathlib import Path

import numpy as np

from apextrace.behaviour import behaviour_report, lap_behaviour
from apextrace.loaders.f1_25 import load_f1_25_capture, load_f1_25_laps
from apextrace.loaders.f1_25.synthetic import synthetic_session


def synthetic_laps():
    def profile(mid):
        def _speed(dist):
            dist = np.asarray(dist, dtype=float)
            speed = np.full(dist.shape, 290.0)
            for centre, floor, width in (
                (700.0, 95.0, 90.0),
                (1800.0, mid, 120.0),
                (3100.0, 75.0, 85.0),
            ):
                speed -= (290.0 - floor) * np.exp(-(((dist - centre) / width) ** 2))
            return speed

        return _speed

    datagrams, _ = synthetic_session(
        [profile(140.0), profile(115.0), profile(140.0)], lap_length=4200.0
    )
    # A finer grid than the default: input shapes are short.
    return load_f1_25_laps(datagrams, step=2.0)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", help="F1 25 capture file")
    args = parser.parse_args()

    if args.path is None:
        print("no capture given: using the synthetic session (filler inputs)")
        laps = synthetic_laps()
    else:
        laps, _ = load_f1_25_capture(args.path, step=2.0)
    if len(laps) < 2:
        raise SystemExit("need two laps to compare")

    ranked = sorted(laps, key=lambda lap: lap.lap_time)[:2]
    for lap in ranked:
        report = behaviour_report(lap)
        print(f"\n{lap.label}  ({lap.lap_time:.3f} s)")
        print(report.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
        summary = lap_behaviour(report)
        print(
            f"lap: {summary['trail_brake_m']:.0f} m trail braking, "
            f"{summary['throttle_hesitations']:.0f} lifts on exit, "
            f"{summary['steering_reversals']:.0f} steering reversals, "
            f"mean brake build {summary['mean_brake_build_m']:.0f} m"
        )


if __name__ == "__main__":
    main()
