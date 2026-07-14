"""M3 validation: does landmark alignment improve the delta inside the lap?

Any axis alignment gives the same final delta because the line
crossings pin it. Official sector times provide an external reference
at two interior points: the cumulative gap at the end of S1 and S2.
The delta curve is evaluated at each sector boundary and compared to
the official cumulative gap there, once with the two-landmark
line-crossing alignment (M2) and once with official corner marks as
interior anchors (M3). Whichever tracks the official gaps closer is
the honest default.
"""

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


def cumulative_sectors(year: int, gp: str, session: str,
                       driver: str) -> tuple[float, float]:
    """Official elapsed time at the end of S1 and S2 for the fastest lap."""
    f1s = fastf1.get_session(year, gp, session)
    f1s.load(telemetry=False, weather=False, messages=False)
    lap = f1s.laps.pick_drivers(driver).pick_fastest()
    s1 = lap["Sector1Time"].total_seconds()
    s2 = s1 + lap["Sector2Time"].total_seconds()
    return s1, s2


def main() -> None:
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"{'case':34} {'sec':>3} {'official':>9} {'err M2':>7} {'err lmk':>8}")
    err_plain: list[float] = []
    err_marks: list[float] = []
    for year, gp, session, a, b in CASES:
        lap_a = load_fastf1_lap(year, gp, session, a)
        lap_b = load_fastf1_lap(year, gp, session, b)
        landmarks = paired_marks(
            lap_a, lap_b, load_fastf1_corner_marks(year, gp, session))
        d_plain = delta_time(lap_a, lap_b)
        d_marks = delta_time(lap_a, lap_b, landmarks=landmarks)

        sec_a = cumulative_sectors(year, gp, session, a)
        sec_b = cumulative_sectors(year, gp, session, b)
        for n, (t_a, t_b) in enumerate(zip(sec_a, sec_b), start=1):
            gap = t_b - t_a
            # Where lap_a crossed this sector line, on the common axis.
            s_star = float(np.interp(t_a, lap_a.data["Time"],
                                     lap_a.data["Distance"]))
            e_p = float(np.interp(s_star, d_plain.index, d_plain.to_numpy())) - gap
            e_m = float(np.interp(s_star, d_marks.index, d_marks.to_numpy())) - gap
            err_plain.append(e_p)
            err_marks.append(e_m)
            name = f"{gp[:20]} {year} {session} {b}vs{a}"
            print(f"{name:34} {n:3d} {gap:9.3f} {e_p:7.3f} {e_m:8.3f}")

    rms = lambda err: float(np.sqrt(np.mean(np.square(err))))  # noqa: E731
    print(f"\nRMS over {len(err_plain)} sector checks: "
          f"M2 line-crossing {rms(err_plain):.3f} s, "
          f"corner-mark landmarks {rms(err_marks):.3f} s")


if __name__ == "__main__":
    main()
