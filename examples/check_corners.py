"""M3 checkpoint 1: detect corners on a fast track and a slow one.

Sanity check against the real circuit maps: Monza has six braking
zones (Rettifilo, Roggia, Lesmo 1 and 2, Ascari, Parabolica), Monaco
is nothing but corners. The printed apex speeds should match the
valleys already seen in the M1/M2 speed traces.
"""

from apextrace.corners import label_corners, locate_marks, segment_lap
from apextrace.loaders.fastf1_loader import (
    load_fastf1_corner_marks,
    load_fastf1_lap,
)

CASES = (
    (2024, "Italian Grand Prix", "Q", "NOR"),
    (2024, "Monaco Grand Prix", "Q", "LEC"),
)


def fmt(event: float | None) -> str:
    """Event position or a dash when the event has no meaning here."""
    return f"{event:7.1f}" if event is not None else "      -"


def main() -> None:
    for year, gp, session, driver in CASES:
        lap = load_fastf1_lap(year, gp, session, driver)
        corners = segment_lap(lap)
        marks = locate_marks(lap, load_fastf1_corner_marks(year, gp, session))
        label_corners(corners, marks)
        print(f"\n{lap.label}: {len(corners)} corners detected")
        for c in corners:
            print(
                f"  {c.name or '#' + str(c.number):<9} apex {c.apex:7.1f} m"
                f"  {c.apex_speed:5.1f} km/h"
                f"  lift {fmt(c.lift_point)}"
                f"  brake {fmt(c.brake_point)}"
                f"  full {fmt(c.throttle_point)}"
            )


if __name__ == "__main__":
    main()
