"""Driving-behaviour heuristics, read from the pedal and steering traces.

The corner report says where time is gained or lost. This module looks
at the inputs behind it: how the brake is applied and released, whether
the car is still on the brakes while turning in, how the throttle comes
back, how busy the steering is. It is exploratory by design, and the
word heuristic is meant literally: each number is a simple, stated
reading of a trace, in physical units, with every threshold exposed as
a parameter. None of them is a score, and none says what a driver
should do; a long brake release is a technique in one corner and a
hesitation in another. Their use is comparative: the same corner, two
laps, what was done differently.

These need analogue inputs, so they are for sim data. F1 timing data
has a near on/off brake and no steering channel: on such a lap the
brake shape metrics are degenerate (peak 100, build 0) and the steering
ones are NaN.

Two limits worth knowing before quoting a figure:

Resolution. Everything is measured on the lap's distance grid, so no
length is finer than one grid step (5 m by default; a loader's ``step``
argument sets it). A brake that reaches its peak in 8 m is at the edge
of what a 5 m grid can say. For input analysis, load laps with a finer
step.

Steering scale. Steering thresholds are relative to each corner's own
peak steering, never absolute. That keeps the steering metrics
independent of how a source normalises its steering channel, at the
price of not being able to say a corner needed "a lot" of lock.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from apextrace.corners import Corner, segment_lap
from apextrace.lap import Lap

NAN = float("nan")


@dataclass
class CornerBehaviour:
    """Input metrics for one corner. Lengths in metres, pedals in %.

    A metric is NaN when its event does not happen in the corner (no
    braking in a flat-out kink) or its channel is missing (steering).
    """

    corner: str
    brake_peak: float  # highest brake pressure in the braking zone [%]
    brake_build_m: float  # brake point to peak pressure
    brake_release_m: float  # peak pressure to brakes off
    trail_brake_m: float  # braking while already turning
    throttle_ramp_m: float  # throttle pickup to full throttle
    throttle_hesitations: float  # lifts on the way back to full throttle
    steering_reversals: float  # direction changes of the steering trace
    pedal_overlap_m: float  # brake and throttle applied together


def _run_from(mask: np.ndarray, start: int) -> int:
    """Index one past the end of the run of True in mask from start."""
    false_after = np.flatnonzero(~mask[start:])
    return mask.size if false_after.size == 0 else start + int(false_after[0])


def count_reversals(values: np.ndarray, gap: float) -> int:
    """Times a trace changes direction by at least gap.

    The reversal count used in steering studies: a change of direction
    only counts once the trace has moved back by more than gap from its
    last extreme, so sensor noise and a hand resting on the wheel do
    not register. A single clean arc, in and back out, is one reversal.
    """
    if values.size < 2 or gap <= 0.0:
        return 0
    count = 0
    direction = 0  # +1 rising, -1 falling, 0 not yet known
    extreme = float(values[0])
    for value in values[1:]:
        move = float(value) - extreme
        if direction == 0:
            if abs(move) >= gap:
                direction = 1 if move > 0 else -1
                extreme = float(value)
        elif move * direction > 0:
            extreme = float(value)  # still going the same way: new extreme
        elif abs(move) >= gap:
            count += 1
            direction = -direction
            extreme = float(value)
    return count


def count_hesitations(throttle: np.ndarray, drop: float) -> int:
    """Times the throttle is eased by at least drop while being applied.

    Counted against the running maximum: a lift only registers once
    the pedal has come back by more than drop from the most it had
    been pressed, and one lift counts once however long it lasts.
    """
    count = 0
    peak = -np.inf
    lifted = False
    for value in throttle:
        if value >= peak:
            peak = float(value)
            lifted = False
        elif not lifted and peak - value >= drop:
            count += 1
            lifted = True
    return count


def corner_behaviour(
    lap: Lap,
    corner: Corner,
    brake_threshold: float = 10.0,
    throttle_threshold: float = 95.0,
    peak_tolerance: float = 0.05,
    trail_steering: float = 0.30,
    hesitation_drop: float = 10.0,
    reversal_gap: float = 0.10,
    overlap_throttle: float = 20.0,
) -> CornerBehaviour:
    """Input metrics for one corner of a lap.

    brake_threshold, throttle_threshold:
        As in segment_lap: Brake % that counts as braking, Throttle %
        that counts as full throttle.
    peak_tolerance:
        Fraction of the peak brake pressure within which the pedal
        counts as "at the peak". A held brake is a plateau, not a
        point; build is measured to where the plateau starts and
        release from where it ends.
    trail_steering:
        Fraction of the corner's peak steering above which the car
        counts as turning, for trail braking.
    hesitation_drop:
        Throttle % the pedal must come back by to count as a lift.
    reversal_gap:
        Fraction of the corner's steering range a change of direction
        must exceed to count as a reversal.
    overlap_throttle:
        Throttle % above which the throttle counts as applied while
        braking.
    """
    data = lap.data
    dist = data["Distance"].to_numpy(dtype=float)
    step = float(dist[1] - dist[0])
    seg = np.flatnonzero((dist >= corner.start) & (dist <= corner.end))
    d = dist[seg]
    throttle = data["Throttle"].to_numpy(dtype=float)[seg]
    brake = data["Brake"].to_numpy(dtype=float)[seg]
    steering = (
        data["Steering"].to_numpy(dtype=float)[seg]
        if "Steering" in lap.channels
        else None
    )
    braking = brake > brake_threshold

    result = CornerBehaviour(
        corner=corner.name or f"#{corner.number}",
        brake_peak=NAN,
        brake_build_m=NAN,
        brake_release_m=NAN,
        trail_brake_m=NAN,
        throttle_ramp_m=NAN,
        throttle_hesitations=NAN,
        steering_reversals=NAN,
        pedal_overlap_m=float(np.sum(braking & (throttle > overlap_throttle)) * step),
    )

    # --- braking zone: the run of braking that starts at the brake point
    if corner.brake_point is not None:
        first = int(np.searchsorted(d, corner.brake_point))
        end = _run_from(braking, first)
        zone = slice(first, end)
        peak = float(np.max(brake[zone]))
        at_peak = first + np.flatnonzero(brake[zone] >= peak * (1.0 - peak_tolerance))
        result.brake_peak = peak
        result.brake_build_m = float(d[at_peak[0]] - d[first])
        # Released where the pedal first drops out of the braking run;
        # if the run reaches the end of the segment, at that end.
        released = d[end] if end < d.size else d[-1]
        result.brake_release_m = float(released - d[at_peak[-1]])

        if steering is not None:
            lock = float(np.max(np.abs(steering)))
            if lock > 0.0:
                turning = np.abs(steering[zone]) >= trail_steering * lock
                result.trail_brake_m = float(np.sum(turning) * step)
            else:
                result.trail_brake_m = 0.0

    # --- throttle application: from the last moment at minimum throttle
    # to sustained full throttle.
    if corner.throttle_point is not None:
        full = int(np.searchsorted(d, corner.throttle_point))
        approach = throttle[: full + 1]
        floor = float(np.min(approach))
        # Within a few percent of the minimum still counts as "not yet
        # picked up": a pedal resting at 2 % is not an application.
        pickup = int(np.flatnonzero(approach <= floor + 5.0)[-1])
        result.throttle_ramp_m = float(d[full] - d[pickup])
        result.throttle_hesitations = float(
            count_hesitations(throttle[pickup : full + 1], hesitation_drop)
        )

    if steering is not None:
        span = float(np.max(steering) - np.min(steering))
        result.steering_reversals = float(
            count_reversals(steering, reversal_gap * span)
        )

    return result


def behaviour_report(
    lap: Lap, corners: list[Corner] | None = None, **thresholds
) -> pd.DataFrame:
    """Input metrics for every corner of a lap, one row per corner.

    corners defaults to the lap's own segmentation, and if given must
    be this lap's own (for instance after label_corners named them):
    the metrics start from each corner's brake and throttle points,
    which are positions on this lap, not on any other. To compare two
    laps, report each on its own corners and pair the rows in order,
    as the corner report does.
    """
    if corners is None:
        corners = segment_lap(lap)
    table = pd.DataFrame(
        [asdict(corner_behaviour(lap, corner, **thresholds)) for corner in corners]
    )
    table.attrs["lap"] = lap.label
    return table


def lap_behaviour(report: pd.DataFrame) -> dict[str, float]:
    """Lap-level summary of a behaviour report.

    Lengths and counts are summed over the corners where they exist;
    the brake shape is averaged, since it describes a habit rather than
    an amount.
    """
    return {
        "trail_brake_m": float(report["trail_brake_m"].sum(min_count=1)),
        "pedal_overlap_m": float(report["pedal_overlap_m"].sum()),
        "throttle_hesitations": float(report["throttle_hesitations"].sum(min_count=1)),
        "steering_reversals": float(report["steering_reversals"].sum(min_count=1)),
        "mean_brake_build_m": float(report["brake_build_m"].mean()),
        "mean_brake_release_m": float(report["brake_release_m"].mean()),
        "mean_throttle_ramp_m": float(report["throttle_ramp_m"].mean()),
    }
