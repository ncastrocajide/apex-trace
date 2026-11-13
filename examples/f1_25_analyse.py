"""M4: analyse an F1 25 capture with the engine written for F1 data.

Decodes a capture recorded by f1_25_record.py, lists the laps it found,
and compares the two fastest with the same delta and corner report used
for the FastF1 laps in the M2 and M3 examples. Nothing in the engine
knows this data came from a game.

    uv run python examples/f1_25_record.py monza.f1cap
    uv run python examples/f1_25_analyse.py monza.f1cap

Corner names are missing here, unlike the FastF1 examples: official
corner marks are reference data about a real circuit, and the game does
not broadcast them. Corners are still detected and still numbered, so
only the labels are lost, which is what "cosmetic by design" meant.
"""

import argparse
from pathlib import Path

from apextrace.delta import delta_time
from apextrace.loaders.f1_25 import load_f1_25_capture
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
    parser.add_argument("path", type=Path, help="capture file to analyse")
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="include laps the game flagged invalid",
    )
    args = parser.parse_args()

    laps, skipped = load_f1_25_capture(args.path, keep_invalid=args.keep_invalid)

    print(f"\n{args.path}: {len(laps)} usable laps")
    for lap in laps:
        print(f"  {lap.label}  {lap.lap_time:7.3f} s  {len(lap.channels)} channels")
    for drop in skipped:
        print(f"  skipped {drop}")

    if len(laps) < 2:
        raise SystemExit("need two laps to compare; record a longer session")

    # Fastest against second fastest, the pair worth looking at.
    ranked = sorted(laps, key=lambda lap: lap.lap_time)
    lap_a, lap_b = ranked[0], ranked[1]

    delta = delta_time(lap_a, lap_b)
    print(f"\n{lap_b.label} vs {lap_a.label}: {delta.iloc[-1]:+.3f} s over the lap")

    table = corner_report(lap_a, lap_b)
    print(table.to_string(index=False, formatters=FLOAT_FORMATS))

    for lap in (lap_a, lap_b):
        m = lap_metrics(lap)
        print(
            f"{lap.label}: {m['full_throttle']:.1%} full throttle, "
            f"{m['braking']:.1%} braking, {m['coasting']:.1%} coasting"
        )


if __name__ == "__main__":
    main()
