"""Per-corner comparison of two laps, in race engineering terms.

Everything here is derived: it reads the segmentation, the phase events
and the delta and turns them into the numbers an engineer quotes (brakes
later, carries the apex deeper, on power earlier), plus lap-level
metrics. Nothing is stored back on the Lap.

Sign conventions, all "lap_b relative to lap_a": positive position
deltas [m] mean lap_b does it later along the track (brakes later, apex
deeper, back on power later); positive time deltas [s] mean lap_b loses
time there.
"""

import numpy as np
import pandas as pd

from apextrace.corners import (
    Corner,
    label_corners,
    locate_marks,
    paired_marks,
    segment_lap,
)
from apextrace.delta import aligned_axes, delta_time
from apextrace.lap import Lap


def lap_metrics(lap: Lap, brake_threshold: float = 10.0,
                throttle_threshold: float = 95.0,
                coast_throttle: float = 5.0) -> dict[str, float]:
    """Lap-level driving metrics, as fractions of lap distance.

    full_throttle: at or above throttle_threshold.
    braking: brake applied.
    coasting: neither pedal, rolling on inertia (between the lift and
    the brake, or between brake release and power).
    """
    throttle = lap.data["Throttle"].to_numpy(dtype=float)
    brake = lap.data["Brake"].to_numpy(dtype=float)
    braking = brake > brake_threshold
    return {
        "full_throttle": float(np.mean(throttle >= throttle_threshold)),
        "braking": float(np.mean(braking)),
        "coasting": float(np.mean((throttle < coast_throttle) & ~braking)),
    }


def _coast_metres(lap: Lap, corner: Corner, brake_threshold: float,
                  coast_throttle: float) -> float:
    """Metres spent coasting inside one corner segment."""
    dist = lap.data["Distance"].to_numpy(dtype=float)
    throttle = lap.data["Throttle"].to_numpy(dtype=float)
    brake = lap.data["Brake"].to_numpy(dtype=float)
    seg = (dist >= corner.start) & (dist <= corner.end)
    coast = seg & (throttle < coast_throttle) & (brake <= brake_threshold)
    return float(np.sum(coast) * (dist[1] - dist[0]))


def corner_report(lap_a: Lap, lap_b: Lap,
                  marks: pd.DataFrame | None = None,
                  brake_threshold: float = 10.0,
                  coast_throttle: float = 5.0) -> pd.DataFrame:
    """Corner-by-corner comparison table of lap_b against lap_a.

    One row per corner, labelled from the official marks when given.
    Marks also anchor the axis alignment (validated in
    examples/validate_alignment.py to sharpen the interior delta).
    entry_s and exit_s split each corner's time swing at the apex;
    total_s sums over the table to the full-lap delta exactly, because
    the segments tile the lap with no gaps.
    """
    corners_a = segment_lap(lap_a)
    corners_b = segment_lap(lap_b)
    if len(corners_a) != len(corners_b):
        raise ValueError(
            f"segmentations disagree: {len(corners_a)} corners on "
            f"{lap_a.label!r} vs {len(corners_b)} on {lap_b.label!r}")

    landmarks = None
    if marks is not None:
        label_corners(corners_a, locate_marks(lap_a, marks))
        landmarks = paired_marks(lap_a, lap_b, marks)
    delta = delta_time(lap_a, lap_b, landmarks=landmarks)

    # lap_b's event positions live on its own odometer; read them on the
    # reconciled axis before comparing against lap_a's.
    dist_b = lap_b.data["Distance"].to_numpy(dtype=float)
    _, dist_b_aligned = aligned_axes(lap_a, lap_b, landmarks)

    def onto_a(position: float | None) -> float:
        if position is None:
            return np.nan
        return float(np.interp(position, dist_b, dist_b_aligned))

    def pos(position: float | None) -> float:
        return np.nan if position is None else position

    def delta_at(s: float) -> float:
        return float(np.interp(s, delta.index, delta.to_numpy()))

    rows = []
    for ca, cb in zip(corners_a, corners_b):
        d_start, d_apex, d_end = (delta_at(ca.start), delta_at(ca.apex),
                                  delta_at(ca.end))
        rows.append({
            "corner": ca.name or f"#{ca.number}",
            "apex_a_kmh": ca.apex_speed,
            "apex_b_kmh": cb.apex_speed,
            "brake_m": onto_a(cb.brake_point) - pos(ca.brake_point),
            "apex_m": onto_a(cb.apex) - ca.apex,
            "power_m": onto_a(cb.throttle_point) - pos(ca.throttle_point),
            "coast_a_m": _coast_metres(lap_a, ca, brake_threshold, coast_throttle),
            "coast_b_m": _coast_metres(lap_b, cb, brake_threshold, coast_throttle),
            "entry_s": d_apex - d_start,
            "exit_s": d_end - d_apex,
            "total_s": d_end - d_start,
        })

    table = pd.DataFrame(rows)
    table.attrs["laps"] = (lap_a.label, lap_b.label)
    table.attrs["landmarks"] = 0 if landmarks is None else len(landmarks[0])
    return table
