"""M2 validation: the delta engine vs official timing, across tracks,
drivers, seasons and session types. The engine never sees the official
gap, so this is an external reference, not circular validation."""

import fastf1
import numpy as np

from apextrace.corners import paired_marks
from apextrace.delta import delta_time
from apextrace.loaders.fastf1_loader import (
    CACHE_DIR,
    load_fastf1_corner_marks,
    load_fastf1_lap,
)

CASES = [
    (2024, "Italian Grand Prix", "Q", "NOR", "LEC"),
    (2024, "Monaco Grand Prix", "Q", "LEC", "PIA"),
    (2024, "British Grand Prix", "Q", "RUS", "HAM"),
    (2023, "Singapore Grand Prix", "Q", "SAI", "RUS"),
    (2024, "Italian Grand Prix", "R", "LEC", "PIA"),
]


def official_gap(year: int, gp: str, session: str, drv_a: str, drv_b: str) -> float:
    """Fastest-lap gap straight from F1 timing (the external reference)."""
    f1s = fastf1.get_session(year, gp, session)
    f1s.load(telemetry=False, weather=False, messages=False)
    times = [
        f1s.laps.pick_drivers(d).pick_fastest()["LapTime"].total_seconds()
        for d in (drv_a, drv_b)
    ]
    return times[1] - times[0]


def main() -> None:
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"{'case':38} {'official':>9} {'d_time':>8} {'err':>7} "
          f"{'d_speed':>8} {'err':>7} {'d_lmk':>8} {'err':>7} {'n_lm':>5} {'shift':>6}")
    for year, gp, session, a, b in CASES:
        lap_a = load_fastf1_lap(year, gp, session, a)
        lap_b = load_fastf1_lap(year, gp, session, b)
        ref = official_gap(year, gp, session, a, b)
        d_t = delta_time(lap_a, lap_b, method="time")
        d_s = delta_time(lap_a, lap_b, method="speed")

        # Landmark-aligned delta: official corner marks projected onto
        # each lap's own axis as extra anchors. The final value must stay
        # pinned by the line crossings; "shift" is how much the alignment
        # moves the delta reading inside the lap.
        marks = load_fastf1_corner_marks(year, gp, session)
        landmarks = paired_marks(lap_a, lap_b, marks)
        d_l = delta_time(lap_a, lap_b, method="time", landmarks=landmarks)
        shift = float(np.max(np.abs(
            d_l.to_numpy() - d_t.reindex(d_l.index).to_numpy())))

        name = f"{gp[:20]} {year} {session} {b}vs{a}"
        print(f"{name:38} {ref:9.3f} {d_t.iloc[-1]:8.3f} "
              f"{d_t.iloc[-1] - ref:7.3f} {d_s.iloc[-1]:8.3f} {d_s.iloc[-1] - ref:7.3f} "
              f"{d_l.iloc[-1]:8.3f} {d_l.iloc[-1] - ref:7.3f} "
              f"{d_l.attrs['landmarks']:5d} {shift:6.3f}")


if __name__ == "__main__":
    main()
