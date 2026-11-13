"""Analysis results shaped for the browser.

Nothing is computed here that the engine does not already compute: this
module calls the delta, the segmentation and the reports, and arranges
what they return as plain JSON-ready structures. Keeping it free of any
web framework means the payloads are tested as ordinary functions, and
the server is left with nothing to do but route.
"""

import math

import numpy as np

from apextrace.behaviour import behaviour_report
from apextrace.corners import segment_lap
from apextrace.delta import aligned_axes, common_grid, delta_time
from apextrace.lap import DISCRETE_CHANNELS, Lap
from apextrace.report import corner_report, lap_metrics

# Channels offered for overlay, in display order, when a lap has them.
OVERLAY_CHANNELS = ("Speed", "Throttle", "Brake", "Gear", "RPM", "Steering")

# Two laps whose lengths differ by more than this are not laps of the
# same circuit. Odometers of the same track disagree by a few metres in
# several thousand, so the margin is generous without letting through
# anything a comparison could mean something for.
MAX_LENGTH_MISMATCH = 0.03


# The behaviour table, column by column: key in the report, heading,
# unit, and decimals to show. Kept here so the browser renders whatever
# it is sent and never needs to know what a metric is.
BEHAVIOUR_COLUMNS = (
    ("brake_peak", "Brake peak", "%", 0),
    ("brake_build_m", "Brake build", "m", 0),
    ("brake_release_m", "Brake release", "m", 0),
    ("trail_brake_m", "Trail braking", "m", 0),
    ("throttle_ramp_m", "Throttle ramp", "m", 0),
    ("throttle_hesitations", "Lifts on exit", "count", 0),
    ("steering_reversals", "Steering reversals", "count", 0),
    ("pedal_overlap_m", "Pedal overlap", "m", 0),
)


class NotComparable(ValueError):
    """Two laps that cannot meaningfully be compared."""


def _values(array: np.ndarray, digits: int = 3) -> list[float | None]:
    """A JSON-ready list: rounded, with non-finite values as null."""
    return [
        round(float(v), digits) if math.isfinite(v) else None
        for v in np.asarray(array, dtype=float)
    ]


def _number(value: float, digits: int = 4) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


def _on_grid(grid: np.ndarray, axis: np.ndarray, lap: Lap, channel: str) -> np.ndarray:
    """One channel of a lap read on a grid, honouring discrete channels."""
    values = lap.data[channel].to_numpy(dtype=float)
    if channel in DISCRETE_CHANNELS:
        # Zero-order hold, as in resampling: a gear is never 4.3.
        idx = np.clip(np.searchsorted(axis, grid, side="right") - 1, 0, axis.size - 1)
        return values[idx]
    return np.interp(grid, axis, values)


def _segments(lap: Lap) -> list[dict]:
    """Corner segments of a lap, or an empty list if none are detected."""
    try:
        corners = segment_lap(lap)
    except ValueError:
        return []
    return [
        {
            "label": corner.name or f"#{corner.number}",
            "start": _number(corner.start, 1),
            "apex": _number(corner.apex, 1),
            "end": _number(corner.end, 1),
            "apex_speed": _number(corner.apex_speed, 1),
            "brake_point": None
            if corner.brake_point is None
            else _number(corner.brake_point, 1),
            "throttle_point": None
            if corner.throttle_point is None
            else _number(corner.throttle_point, 1),
        }
        for corner in corners
    ]


def _records(table) -> list[dict]:
    """A DataFrame as a list of JSON-ready rows."""
    return [
        {
            key: (_number(value) if isinstance(value, float) else value)
            for key, value in row.items()
        }
        for row in table.to_dict(orient="records")
    ]


def _has_driver_inputs(lap: Lap) -> bool:
    """Whether a lap's inputs are rich enough to read behaviour from.

    F1 timing data has a brake that is either off or fully on and no
    steering; the behaviour metrics would be computed, and would mean
    nothing. A steering channel, or a brake that spends time between
    the stops, marks a source that records what the driver did.
    """
    if "Steering" in lap.channels:
        return True
    brake = lap.data["Brake"].to_numpy(dtype=float)
    return float(np.mean((brake > 5.0) & (brake < 95.0))) > 0.01


def _behaviour(laps: list[tuple[str | None, Lap]]) -> dict | None:
    """Behaviour table for one lap, or for two laps row against row.

    Each lap is read on its own corners and the rows are paired in
    order, so two laps only share a table when they segment into the
    same number of corners, as for the corner report.
    """
    if not all(_has_driver_inputs(lap) for _, lap in laps):
        return None
    try:
        reports = [behaviour_report(lap) for _, lap in laps]
    except ValueError:
        return None  # no corners detected
    if len({len(report) for report in reports}) != 1:
        return None

    columns = [{"key": "corner", "title": "Corner"}]
    if len(laps) > 1:
        columns.append({"key": "lap", "title": "Lap"})
    columns += [
        {"key": key, "title": title, "unit": unit, "digits": digits}
        for key, title, unit, digits in BEHAVIOUR_COLUMNS
    ]

    rows = []
    for i in range(len(reports[0])):
        for (name, _), report in zip(laps, reports, strict=True):
            row = _records(report.iloc[[i]])[0]
            # Corner names are the first lap's: the rows are paired.
            row["corner"] = reports[0]["corner"].iloc[i]
            if name is not None:
                row["lap"] = name
            rows.append(row)
    return {"columns": columns, "rows": rows}


def lap_payload(lap: Lap) -> dict:
    """One lap on its own: traces, corners and lap-level metrics."""
    dist = lap.data["Distance"].to_numpy(dtype=float)
    payload = {
        "label": lap.label,
        "lap_time": _number(lap.lap_time),
        "distance": _values(dist, 1),
        "channels": {
            name: _values(lap.data[name].to_numpy(dtype=float))
            for name in OVERLAY_CHANNELS
            if name in lap.channels
        },
        "corners": _segments(lap),
        "metrics": {k: _number(v) for k, v in lap_metrics(lap).items()},
        "behaviour": _behaviour([(None, lap)]),
        "map": None,
    }
    if "X" in lap.channels and "Y" in lap.channels:
        payload["map"] = {
            "x": _values(lap.data["X"].to_numpy(dtype=float), 1),
            "y": _values(lap.data["Y"].to_numpy(dtype=float), 1),
        }
    return payload


def check_comparable(lap_a: Lap, lap_b: Lap) -> None:
    """Raise NotComparable unless both laps are of the same circuit.

    Length is the test, not the track name: names come from each
    source's own vocabulary (a FastF1 lap is at "Monza", a sim export
    may say "monza_2024"), while two laps of one circuit always agree
    on its length to within odometer error.
    """
    end_a = float(lap_a.data["Distance"].iloc[-1])
    end_b = float(lap_b.data["Distance"].iloc[-1])
    if abs(end_a - end_b) > MAX_LENGTH_MISMATCH * max(end_a, end_b):
        raise NotComparable(
            f"these laps are not of the same circuit: {lap_a.label!r} is "
            f"{end_a:.0f} m long and {lap_b.label!r} is {end_b:.0f} m"
        )


def compare_payload(lap_a: Lap, lap_b: Lap) -> dict:
    """lap_b against lap_a: delta, overlays, corner table, metrics, map.

    Everything is read on the engine's common grid, with lap_b on the
    reconciled axis, so a cursor at one distance means the same piece of
    track on every trace. Sign conventions are the engine's: positive
    delta means lap_b is behind.
    """
    check_comparable(lap_a, lap_b)

    grid = common_grid(lap_a, lap_b)
    axis_a, axis_b = aligned_axes(lap_a, lap_b)
    delta = delta_time(lap_a, lap_b).to_numpy()

    shared = [
        name
        for name in OVERLAY_CHANNELS
        if name in lap_a.channels and name in lap_b.channels
    ]
    payload = {
        "a": {"label": lap_a.label, "lap_time": _number(lap_a.lap_time)},
        "b": {"label": lap_b.label, "lap_time": _number(lap_b.lap_time)},
        "cross_source": lap_a.source != lap_b.source,
        "distance": _values(grid, 1),
        "delta": _values(delta, 4),
        "final_delta": _number(delta[-1]),
        "channels": {
            name: {
                "a": _values(_on_grid(grid, axis_a, lap_a, name)),
                "b": _values(_on_grid(grid, axis_b, lap_b, name)),
            }
            for name in shared
        },
        "segments": _segments(lap_a),
        "metrics": {
            "a": {k: _number(v) for k, v in lap_metrics(lap_a).items()},
            "b": {k: _number(v) for k, v in lap_metrics(lap_b).items()},
        },
        "corners": None,
        "corners_error": None,
        "behaviour": _behaviour([("A", lap_a), ("B", lap_b)]),
        "map": None,
    }

    # The corner table needs both laps to segment into the same corners.
    # When they do not (a lift that is a corner on one lap only), the
    # rest of the comparison is still valid, so say why and carry on.
    try:
        payload["corners"] = _records(corner_report(lap_a, lap_b))
    except ValueError as error:
        payload["corners_error"] = str(error)

    if "X" in lap_a.channels and "Y" in lap_a.channels:
        # Time lost per 100 m, the local slope of the delta: colouring
        # the map by it shows where on the circuit the lap swings.
        slope = np.gradient(delta, grid) * 100.0
        payload["map"] = {
            "x": _values(_on_grid(grid, axis_a, lap_a, "X"), 1),
            "y": _values(_on_grid(grid, axis_a, lap_a, "Y"), 1),
            "loss_per_100m": _values(slope, 4),
        }
    return payload
