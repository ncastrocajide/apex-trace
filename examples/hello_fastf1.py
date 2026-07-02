"""M0 smoke test: prove the whole data chain works end to end.

Downloads one F1 session from a closed season (2024 Italian GP qualifying)
and prints the fastest lap of a few drivers. If this runs, the environment,
FastF1, its cache and the network path are all working.
"""

from datetime import timedelta
from pathlib import Path

import fastf1

# FastF1 downloads tens of MB per session; a local cache makes every run
# after the first one near-instant and avoids hammering the F1 API.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".fastf1_cache"

DRIVERS = ("VER", "LEC", "NOR")


def format_lap_time(lap_time: timedelta) -> str:
    """Render a lap time as m:ss.mmm (e.g. 1:19.327)."""
    total = lap_time.total_seconds()
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    session = fastf1.get_session(2024, "Italian Grand Prix", "Q")
    session.load()  # downloads timing + telemetry on first run, then cached

    print(f"\n{session.event['EventName']} {session.event.year} — {session.name}")
    print("-" * 40)
    for driver in DRIVERS:
        fastest = session.laps.pick_drivers(driver).pick_fastest()
        print(f"  {driver}  {format_lap_time(fastest['LapTime'])}")


if __name__ == "__main__":
    main()
