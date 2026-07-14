"""M3 demo: corner-by-corner report of the M2 star pair.

LEC vs NOR, Monza 2024 qualifying. The M2 overlay showed WHERE the lap
swings; this table says it in race engineering terms, corner by corner.
The sum of total_s must reproduce the full-lap delta exactly, because
the corner segments tile the lap.
"""

import fastf1

from apextrace.delta import delta_time
from apextrace.loaders.fastf1_loader import (
    CACHE_DIR,
    load_fastf1_corner_marks,
    load_fastf1_lap,
)
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
    fastf1.Cache.enable_cache(CACHE_DIR)
    lap_a = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "NOR")
    lap_b = load_fastf1_lap(2024, "Italian Grand Prix", "Q", "LEC")
    marks = load_fastf1_corner_marks(2024, "Italian Grand Prix", "Q")

    table = corner_report(lap_a, lap_b, marks)
    print(
        f"\n{lap_b.label}  vs  {lap_a.label}"
        f"  ({table.attrs['landmarks']} landmark anchors)"
    )
    print(table.to_string(index=False, formatters=FLOAT_FORMATS))

    total = table["total_s"].sum()
    final = delta_time(lap_a, lap_b).iloc[-1]
    print(f"\nsum of per-corner deltas {total:+.3f} s, full-lap delta {final:+.3f} s")

    for lap in (lap_a, lap_b):
        m = lap_metrics(lap)
        print(
            f"{lap.label}: {m['full_throttle']:.1%} full throttle, "
            f"{m['braking']:.1%} braking, {m['coasting']:.1%} coasting"
        )


if __name__ == "__main__":
    main()
