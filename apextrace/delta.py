"""Cumulative time delta between two laps on a common distance axis.

Sign convention: delta(s) > 0 means lap_b has lost that much time to
lap_a by distance s; where the curve rises, lap_b is slower right there.

Two methods exist. "time" subtracts the interpolated Time channels and
is the default: validated to millisecond accuracy against official F1
timing across fast, slow and street tracks, in qualifying and race trim
(see examples/validate_delta.py). "speed" integrates the slowness (1/v)
difference instead; it accumulates the speed channel's biases and is
kept as an explicit cross-check, and as a fallback for sources without
a trustworthy time channel.
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from apextrace.lap import Lap


def aligned_axes(lap_a: Lap, lap_b: Lap) -> tuple[np.ndarray, np.ndarray]:
    """Distance axes of both laps reconciled onto lap_a's lap length.

    Each lap measures distance by integrating its own sampled speed, so
    two odometers of the same physical track disagree by a few metres.
    Linearly rescaling lap_b's axis anchors the two landmarks we trust
    most, the timing-line crossings. This is the two-landmark degenerate
    case of landmark alignment; interior landmarks (corners) plug in here
    once corner detection exists (M3/M4).
    """
    dist_a = lap_a.data["Distance"].to_numpy(dtype=float)
    dist_b = lap_b.data["Distance"].to_numpy(dtype=float)
    return dist_a, dist_b * (dist_a[-1] / dist_b[-1])


def common_grid(lap_a: Lap, lap_b: Lap) -> np.ndarray:
    """Shared distance axis: uniform on lap_a's step, plus the exact end."""
    dist_a, dist_b = aligned_axes(lap_a, lap_b)
    step = float(dist_a[1] - dist_a[0])
    end = min(dist_a[-1], dist_b[-1])
    grid = np.arange(0.0, end, step)
    if end - grid[-1] > 1e-9:
        grid = np.append(grid, end)
    return grid


def _on_grid(grid: np.ndarray, dist: np.ndarray, lap: Lap, channel: str) -> np.ndarray:
    """Interpolate one channel of a lap onto a grid, using a given axis."""
    return np.interp(grid, dist, lap.data[channel].to_numpy(dtype=float))


def delta_time(lap_a: Lap, lap_b: Lap, method: str = "time") -> pd.Series:
    """Cumulative time delta of lap_b relative to lap_a, vs distance.

    method:
        "time"  - subtract the interpolated Time channels (default).
        "speed" - integrate the slowness (1/v) difference over distance.
                  Cross-check only: biased where odometer drift is not
                  uniform along the lap.
    """
    grid = common_grid(lap_a, lap_b)
    dist_a, dist_b = aligned_axes(lap_a, lap_b)

    if method == "time":
        delta = _on_grid(grid, dist_b, lap_b, "Time") - _on_grid(grid, dist_a, lap_a, "Time")
        delta -= delta[0]
    elif method == "speed":
        # Speed [km/h] -> slowness [s/m]. Time is the integral of slowness
        # over distance, so integrating the difference gives the delta.
        # Alignment is a change of variables: rescaling lap_b's axis by k
        # requires rescaling its speed by k too, so that integrating
        # slowness over the aligned axis still returns lap_b's own time.
        k_b = dist_a[-1] / float(lap_b.data["Distance"].iloc[-1])
        slowness_a = 3.6 / _on_grid(grid, dist_a, lap_a, "Speed")
        slowness_b = 3.6 / (k_b * _on_grid(grid, dist_b, lap_b, "Speed"))
        delta = cumulative_trapezoid(slowness_b - slowness_a, grid, initial=0.0)
    else:
        raise ValueError(f"unknown method: {method!r}")

    series = pd.Series(delta, index=pd.Index(grid, name="Distance"),
                       name=f"{lap_b.label} vs {lap_a.label}")
    series.attrs["method"] = method
    return series
