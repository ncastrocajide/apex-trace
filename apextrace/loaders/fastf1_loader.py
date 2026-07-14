"""Load laps from official F1 timing data into the canonical format.

This is the first source adapter: everything FastF1-specific stays inside
this module. The rest of the package only ever sees canonical Laps.
"""

from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from apextrace.lap import Lap, resample_to_distance

# Repo root: this file lives in apextrace/loaders/, so two parents up.
CACHE_DIR = Path(__file__).resolve().parents[2] / ".fastf1_cache"


def load_fastf1_lap(
    year: int,
    gp: str,
    session: str,
    driver: str,
    lap: int | str = "fastest",
    step: float = 5.0,
) -> Lap:
    """Load one launched lap from an F1 session as a canonical Lap.

    Parameters
    ----------
    year, gp, session:
        Passed straight to fastf1.get_session, e.g. (2024, "Italian Grand
        Prix", "Q"). Prefer closed seasons: telemetry for very recent
        sessions is flaky on the F1 API side.
    driver:
        Three-letter driver code, e.g. "NOR".
    lap:
        "fastest" (default) or an absolute lap number for that driver.
    step:
        Distance grid spacing in metres.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    f1_session = fastf1.get_session(year, gp, session)
    f1_session.load()

    driver_laps = f1_session.laps.pick_drivers(driver)
    if lap == "fastest":
        picked = driver_laps.pick_fastest()
        if picked is None:
            raise ValueError(f"{driver} has no valid quick lap in this session")
    else:
        match = driver_laps[driver_laps["LapNumber"] == lap]
        if match.empty:
            raise ValueError(f"{driver} has no lap number {lap}")
        picked = match.iloc[0]

    # Launched laps only: a lap that starts or ends in the pits has no
    # meaningful distance axis to compare on.
    if pd.notna(picked["PitInTime"]) or pd.notna(picked["PitOutTime"]):
        raise ValueError(f"lap {picked['LapNumber']:.0f} is an in/out lap")

    # interpolate_edges adds synthetic first/last samples exactly at the
    # lap start and end, anchoring Time zero to the real line crossing
    # instead of the first telemetry sample (up to ~0.2 s of phase error).
    telemetry = picked.get_car_data(interpolate_edges=True).add_distance()

    raw = pd.DataFrame(
        {
            "Distance": telemetry["Distance"].to_numpy(dtype=float),
            # Timedelta since lap start, converted to plain seconds.
            "Time": telemetry["Time"].dt.total_seconds().to_numpy(),
            "Speed": telemetry["Speed"].to_numpy(dtype=float),
            "Throttle": telemetry["Throttle"].to_numpy(dtype=float),
            # FastF1 brake is boolean; the canonical schema wants 0-100.
            "Brake": telemetry["Brake"].to_numpy(dtype=float) * 100.0,
            "Gear": telemetry["nGear"].to_numpy(dtype=float),
            "RPM": telemetry["RPM"].to_numpy(dtype=float),
        }
    )
    raw["Time"] -= raw["Time"].iloc[0]

    # Track-map coordinates come from a separate positional stream on its
    # own time base; interpolate them onto the car data timestamps. The
    # source unit is 1/10 m, the canonical schema wants metres.
    pos = picked.get_pos_data()
    if not pos.empty:
        t_car = telemetry["Time"].dt.total_seconds().to_numpy()
        t_pos = pos["Time"].dt.total_seconds().to_numpy()
        for axis in ("X", "Y"):
            raw[axis] = np.interp(
                t_car, t_pos, pos[axis].to_numpy(dtype=float)) / 10.0

    label = f"{driver} {year} {f1_session.event['Location']} {f1_session.name}"
    return Lap(
        data=resample_to_distance(raw, step=step),
        label=label,
        source="fastf1",
        track=str(f1_session.event["Location"]),
        car=str(picked["Team"]),
    )


def load_fastf1_corner_marks(year: int, gp: str, session: str) -> pd.DataFrame:
    """Official corner marks of a session's circuit.

    Returns a DataFrame with Distance [m], Name (official designation,
    e.g. "T10") and X, Y [m] map coordinates. This is reference data
    about the real track, not telemetry: useful to label detected
    corners and to annotate track maps. Cosmetic by design: the engine
    works without it, so sources with no such data lose nothing.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    f1s = fastf1.get_session(year, gp, session)
    f1s.load(telemetry=False, weather=False, messages=False)
    corners = f1s.get_circuit_info().corners
    return pd.DataFrame(
        {
            "Distance": corners["Distance"].to_numpy(dtype=float),
            "Name": [f"T{int(number)}{letter}" for number, letter
                     in zip(corners["Number"], corners["Letter"])],
            # Circuit info coordinates come in 1/10 m, like position data.
            "X": corners["X"].to_numpy(dtype=float) / 10.0,
            "Y": corners["Y"].to_numpy(dtype=float) / 10.0,
        }
    )
