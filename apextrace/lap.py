"""Canonical lap representation.

Every telemetry source is converted into this format by its loader.
Everything downstream (delta, overlays, segmentation, metrics) works on a
Lap and never needs to know where the data came from.

Canonical DataFrame schema: one row per point on a uniform distance grid.

    Column    Unit   Notes
    Distance  m      from the finish line, primary axis, uniform grid
    Time      s      elapsed since lap start
    Speed     km/h
    Throttle  %      0 to 100
    Brake     %      0 to 100 (near on/off in F1 timing data)
    Gear      -      integer gear number
    RPM       rpm    optional
    Steering  deg    optional, absent in F1 timing data
    X, Y      m      optional, track map coordinates
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Channels every source must provide.
CORE_CHANNELS = ("Distance", "Time", "Speed", "Throttle", "Brake", "Gear")

# Channels a source may or may not provide.
OPTIONAL_CHANNELS = ("RPM", "Steering", "X", "Y")

# Discrete-valued channels: resampled by zero-order hold, never interpolated.
DISCRETE_CHANNELS = ("Gear",)


@dataclass
class Lap:
    """One lap of telemetry, normalised onto a uniform distance grid."""

    data: pd.DataFrame   # canonical channels on the distance grid
    label: str           # e.g. "NOR 2024 Monza Q"
    source: str          # "fastf1" | "assetto_corsa" | "f1_25"
    track: str | None = None
    car: str | None = None

    @property
    def channels(self) -> list[str]:
        """Channels actually present in this lap."""
        return list(self.data.columns)

    @property
    def lap_time(self) -> float:
        """Lap time in seconds (elapsed time at the last grid point)."""
        return float(self.data["Time"].iloc[-1])


def resample_to_distance(df: pd.DataFrame, step: float = 5.0) -> pd.DataFrame:
    """Resample telemetry channels onto a uniform distance grid.

    Expects a "Distance" column in metres; every other column is treated
    as a channel. Continuous channels are linearly interpolated, channels
    in DISCRETE_CHANNELS use zero-order hold (last known value).
    """
    if "Distance" not in df.columns:
        raise ValueError("telemetry must contain a 'Distance' column")

    dist = df["Distance"].to_numpy(dtype=float)

    # Interpolation needs a strictly increasing x axis. Duplicate or
    # backwards points (sensor noise) are dropped, keeping the first one.
    keep = np.concatenate(([True], np.diff(dist) > 0.0))
    clean = df.loc[keep]
    dist = dist[keep]

    grid = np.arange(0.0, dist[-1], step)
    out = {"Distance": grid}

    for name in clean.columns:
        if name == "Distance":
            continue
        values = clean[name].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if name in DISCRETE_CHANNELS:
            # Zero-order hold: each grid point takes the last sample at or
            # before it; searchsorted finds that sample by binary search.
            idx = np.searchsorted(dist[valid], grid, side="right") - 1
            out[name] = values[valid][np.clip(idx, 0, None)].astype(int)
        else:
            out[name] = np.interp(grid, dist[valid], values[valid])

    return pd.DataFrame(out)
