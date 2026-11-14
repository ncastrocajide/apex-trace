"""M7: a MoTeC CSV export through the same engine, untouched again.

Export a few laps from MoTeC i2 (File > Export > CSV) of a log recorded
in Assetto Corsa with ACTI or Telemetrick, or from a real MoTeC logger,
and compare the two fastest:

    uv run python examples/motec_compare.py outing.csv
    uv run python examples/motec_compare.py outing.csv --steering-lock 240

--steering-lock is the wheel angle at full lock, one side, in degrees
(for Assetto Corsa, half the car's total steering rotation). The export
logs steering as an angle and the canonical channel is a fraction of
lock, so without it the steering channel is left out.
"""

import argparse
from pathlib import Path

from apextrace.behaviour import behaviour_report
from apextrace.delta import delta_time
from apextrace.loaders.motec_csv import load_motec_csv
from apextrace.report import corner_report, lap_metrics

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="MoTeC i2 CSV export")
    parser.add_argument("--steering-lock", type=float, default=None, metavar="DEG")
    args = parser.parse_args()

    laps, notes = load_motec_csv(args.path, steering_lock=args.steering_lock)
    print(f"\n{args.path}: {len(laps)} complete laps")
    for lap in laps:
        print(f"  {lap.label}  {lap.lap_time:7.3f} s  {len(lap.channels)} channels")
    for note in notes:
        print(f"  note: {note}")
    if len(laps) < 2:
        raise SystemExit("need two complete laps to compare")

    ranked = sorted(laps, key=lambda lap: lap.lap_time)
    lap_a, lap_b = ranked[0], ranked[1]

    delta = delta_time(lap_a, lap_b)
    print(f"\n{lap_b.label} vs {lap_a.label}: {delta.iloc[-1]:+.3f} s over the lap")
    print(corner_report(lap_a, lap_b).to_string(index=False, formatters=FLOAT_FORMATS))

    for lap in (lap_a, lap_b):
        m = lap_metrics(lap)
        print(
            f"{lap.label}: {m['full_throttle']:.1%} full throttle, "
            f"{m['braking']:.1%} braking, {m['coasting']:.1%} coasting"
        )
        print(
            behaviour_report(lap).to_string(
                index=False, float_format=lambda v: f"{v:.0f}"
            )
        )


if __name__ == "__main__":
    main()
