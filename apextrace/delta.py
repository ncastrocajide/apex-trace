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


Landmarks = tuple[np.ndarray, np.ndarray]


def _anchors(lap_a: Lap, lap_b: Lap,
             landmarks: Landmarks | None) -> tuple[np.ndarray, np.ndarray]:
    """Matching anchor positions on each lap's own distance axis.

    The line crossings (start and end of the lap) are always anchors;
    paired interior landmarks add more. Both sequences must be strictly
    increasing or the axis map stops being invertible.
    """
    end_a = float(lap_a.data["Distance"].iloc[-1])
    end_b = float(lap_b.data["Distance"].iloc[-1])
    if landmarks is None:
        lm_a = lm_b = np.empty(0)
    else:
        lm_a, lm_b = (np.asarray(lm, dtype=float) for lm in landmarks)
    anchors_a = np.concatenate(([0.0], lm_a, [end_a]))
    anchors_b = np.concatenate(([0.0], lm_b, [end_b]))
    if np.any(np.diff(anchors_a) <= 0.0) or np.any(np.diff(anchors_b) <= 0.0):
        raise ValueError("landmarks must be strictly increasing inside each lap")
    return anchors_a, anchors_b


def aligned_axes(lap_a: Lap, lap_b: Lap,
                 landmarks: Landmarks | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Distance axes of both laps reconciled onto lap_a's axis.

    Each lap measures distance by integrating its own sampled speed, so
    two odometers of the same physical track disagree by a few metres.
    Without landmarks, lap_b's axis is linearly rescaled to anchor the
    two points we trust most, the timing-line crossings (the M2 method,
    validated against official timing). With landmarks (paired interior
    positions, e.g. apexes from paired_apexes), the map becomes piecewise
    linear between anchors, pinning each landmark of lap_b onto its
    partner so the delta stays honest inside the lap, not just at the end.
    """
    dist_a = lap_a.data["Distance"].to_numpy(dtype=float)
    dist_b = lap_b.data["Distance"].to_numpy(dtype=float)
    anchors_a, anchors_b = _anchors(lap_a, lap_b, landmarks)
    return dist_a, np.interp(dist_b, anchors_b, anchors_a)


def common_grid(lap_a: Lap, lap_b: Lap,
                landmarks: Landmarks | None = None) -> np.ndarray:
    """Shared distance axis: uniform on lap_a's step, plus the exact end."""
    dist_a, dist_b = aligned_axes(lap_a, lap_b, landmarks)
    step = float(dist_a[1] - dist_a[0])
    end = min(dist_a[-1], dist_b[-1])
    grid = np.arange(0.0, end, step)
    if end - grid[-1] > 1e-9:
        grid = np.append(grid, end)
    return grid


def _on_grid(grid: np.ndarray, dist: np.ndarray, lap: Lap, channel: str) -> np.ndarray:
    """Interpolate one channel of a lap onto a grid, using a given axis."""
    return np.interp(grid, dist, lap.data[channel].to_numpy(dtype=float))


def delta_time(lap_a: Lap, lap_b: Lap, method: str = "time",
               landmarks: Landmarks | None = None) -> pd.Series:
    """Cumulative time delta of lap_b relative to lap_a, vs distance.

    method:
        "time"  - subtract the interpolated Time channels (default).
        "speed" - integrate the slowness (1/v) difference over distance.
                  Cross-check only: biased where odometer drift is not
                  uniform along the lap.
    landmarks:
        Paired interior positions (positions_in_a, positions_in_b) for
        the piecewise axis alignment, e.g. from paired_apexes. The final
        delta is pinned by the line crossings either way; landmarks make
        the local reading sharper around each corner.
    """
    grid = common_grid(lap_a, lap_b, landmarks)
    dist_a, dist_b = aligned_axes(lap_a, lap_b, landmarks)

    if method == "time":
        delta = _on_grid(grid, dist_b, lap_b, "Time") - _on_grid(grid, dist_a, lap_a, "Time")
        delta -= delta[0]
    elif method == "speed":
        # Speed [km/h] -> slowness [s/m]. Time is the integral of slowness
        # over distance, so integrating the difference gives the delta.
        # Alignment is a change of variables: where lap_b's axis is locally
        # stretched by k (the slope of the axis map, one constant per span
        # between anchors), its speed must be scaled by k too, so that
        # integrating slowness over the aligned axis still returns lap_b's
        # own time. Without landmarks there is one span and one k.
        anchors_a, anchors_b = _anchors(lap_a, lap_b, landmarks)
        slopes = np.diff(anchors_a) / np.diff(anchors_b)
        span = np.clip(np.searchsorted(anchors_a, grid, side="right") - 1,
                       0, slopes.size - 1)
        slowness_a = 3.6 / _on_grid(grid, dist_a, lap_a, "Speed")
        slowness_b = 3.6 / (slopes[span] * _on_grid(grid, dist_b, lap_b, "Speed"))
        delta = cumulative_trapezoid(slowness_b - slowness_a, grid, initial=0.0)
    else:
        raise ValueError(f"unknown method: {method!r}")

    series = pd.Series(delta, index=pd.Index(grid, name="Distance"),
                       name=f"{lap_b.label} vs {lap_a.label}")
    series.attrs["method"] = method
    series.attrs["landmarks"] = 0 if landmarks is None else len(landmarks[0])
    return series
