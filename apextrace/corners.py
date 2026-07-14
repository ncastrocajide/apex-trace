"""Corner segmentation: split a lap into per-corner segments.

An apex is a local minimum of the speed trace, the slowest point of a
corner. Detection runs scipy.signal.find_peaks on the negated Speed
channel with thresholds in physical units (km/h, metres) exposed as
parameters. No smoothing is applied: the resampling grid already
averages out sampling noise, and the house rule is that any smoothing
must be declared, never hidden inside detection.

The lap is partitioned at the midpoints between consecutive apexes, so
every metre of track belongs to exactly one segment and per-corner time
deltas add up exactly to the full-lap delta. Physical events (braking,
lift, back to full throttle) are located inside each segment; a corner
taken flat simply has no such events.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from apextrace.lap import Lap


@dataclass
class Corner:
    """One corner of a segmented lap. Distances in metres from the line.

    The sequential number, segment bounds and apex always exist. Event
    fields are None when the event does not happen in this corner (no
    braking in a flat-out kink, no full throttle between chained
    corners). segment_lap only locates the apexes; events and official
    names are filled by later passes.
    """

    number: int                          # 1-based, in track order
    start: float                         # segment start [m]
    end: float                           # segment end [m]
    apex: float                          # speed minimum position [m]
    apex_speed: float                    # [km/h]
    name: str | None = None              # official designation, e.g. "T4"
    brake_point: float | None = None     # first brake application [m]
    lift_point: float | None = None      # throttle drop before the apex [m]
    throttle_point: float | None = None  # sustained full throttle again [m]


def segment_lap(lap: Lap, min_drop: float = 15.0, min_spacing: float = 50.0,
                brake_threshold: float = 10.0,
                throttle_threshold: float = 95.0) -> list[Corner]:
    """Split a lap into corner segments around its speed minima.

    min_drop: minimum prominence of a speed dip [km/h]. A real corner
        costs at least this much speed; shallower dips are lifts or
        sampling noise.
    min_spacing: minimum distance between apexes [m], so the two kerbs
        of one chicane do not register as two separate corners.
    brake_threshold: Brake % above which the driver is braking. F1 data
        is near on/off so anything low works; sims give real pressure.
    throttle_threshold: Throttle % that counts as full throttle; below
        it on corner entry counts as a lift.
    """
    dist = lap.data["Distance"].to_numpy(dtype=float)
    speed = lap.data["Speed"].to_numpy(dtype=float)
    step = float(dist[1] - dist[0])

    idx, _ = find_peaks(-speed, prominence=min_drop,
                        distance=max(1, round(min_spacing / step)))
    if idx.size == 0:
        raise ValueError(
            "no apexes detected: check the Speed channel or loosen the thresholds")

    apexes = dist[idx]
    inner = (apexes[:-1] + apexes[1:]) / 2.0
    bounds = np.concatenate(([dist[0]], inner, [dist[-1]]))

    corners = [
        Corner(number=i + 1, start=float(bounds[i]), end=float(bounds[i + 1]),
               apex=float(dist[j]), apex_speed=float(speed[j]))
        for i, j in enumerate(idx)
    ]
    _locate_events(lap, corners, brake_threshold, throttle_threshold)
    return corners


def _locate_events(lap: Lap, corners: list[Corner],
                   brake_threshold: float, throttle_threshold: float) -> None:
    """Fill the phase events of each corner: lift, brake, back to full.

    Events are threshold crossings, not levels. A condition already met
    when its search window opens (still braking from a chained corner,
    already flat through a kink) is a carryover from the previous phase,
    not an event of this corner, and leaves the field as None.
    """
    dist = lap.data["Distance"].to_numpy(dtype=float)
    throttle = lap.data["Throttle"].to_numpy(dtype=float)
    brake = lap.data["Brake"].to_numpy(dtype=float)

    for corner in corners:
        entry = (dist >= corner.start) & (dist <= corner.apex)
        exit_ = (dist >= corner.apex) & (dist <= corner.end)
        corner.lift_point = _first_crossing(dist, throttle < throttle_threshold, entry)
        corner.brake_point = _first_crossing(dist, brake > brake_threshold, entry)
        corner.throttle_point = _first_crossing(dist, throttle >= throttle_threshold, exit_)


def _first_crossing(dist: np.ndarray, condition: np.ndarray,
                    region: np.ndarray) -> float | None:
    """Distance of the first False-to-True switch of condition inside region."""
    idx = np.flatnonzero(region)
    if idx.size < 2:
        return None
    cond = condition[idx]
    switch = np.flatnonzero(cond[1:] & ~cond[:-1])
    if switch.size == 0:
        return None
    return float(dist[idx[switch[0] + 1]])


def paired_apexes(corners_a: list[Corner], corners_b: list[Corner],
                  max_offset: float = 50.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Pair the apexes of two segmentations of the same track, by order.

    Fallback for sources without position data; prefer paired_marks
    when X/Y channels exist. Two caveats, both found by validation:
    equal corner counts can still be different corners (Silverstone
    2024 Q detects 7 on both laps, not the same 7), so any pair further
    apart than max_offset returns None and the caller falls back to the
    two-landmark line-crossing alignment validated in M2. And pinning
    apexes together absorbs real apex-placement differences between
    drivers, so expect the local delta to flatten around each corner.
    """
    if len(corners_a) != len(corners_b):
        return None
    pos_a = np.array([c.apex for c in corners_a])
    pos_b = np.array([c.apex for c in corners_b])
    scale = corners_a[-1].end / corners_b[-1].end
    if np.max(np.abs(pos_a - pos_b * scale)) > max_offset:
        return None
    return pos_a, pos_b


def paired_marks(lap_a: Lap, lap_b: Lap,
                 marks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Project the same physical track marks onto both laps' own axes.

    Fixed references (official corner marks) are driver independent, so
    pinning them together reconciles the two odometers without touching
    real driving differences: a later apex stays later. Pinning detected
    apexes instead would absorb exactly that signal. Needs X/Y channels
    on both laps; marks must come in track order. Pairs that land on the
    lap boundaries or break monotonicity are dropped; returns None if
    nothing usable is left.
    """
    pos_a = locate_marks(lap_a, marks)["Distance"].to_numpy(dtype=float)
    pos_b = locate_marks(lap_b, marks)["Distance"].to_numpy(dtype=float)
    end_a = float(lap_a.data["Distance"].iloc[-1])
    end_b = float(lap_b.data["Distance"].iloc[-1])

    kept_a: list[float] = []
    kept_b: list[float] = []
    last_a = last_b = 0.0
    for da, db in zip(pos_a, pos_b):
        if da <= last_a or db <= last_b or da >= end_a or db >= end_b:
            continue
        kept_a.append(da)
        kept_b.append(db)
        last_a, last_b = da, db
    if not kept_a:
        return None
    return np.array(kept_a), np.array(kept_b)


def locate_marks(lap: Lap, marks: pd.DataFrame) -> pd.DataFrame:
    """Locate track marks along a lap by projecting them onto its line.

    marks: table with X and Y columns [m] (e.g. official corner marks).
    Each mark takes the Distance of the nearest sample of the lap's
    driven line, so reference points defined on the map become positions
    on the lap's own distance axis. Needs the optional X/Y channels.
    """
    if "X" not in lap.channels or "Y" not in lap.channels:
        raise ValueError(f"{lap.label!r} has no X/Y channels to project onto")
    x = lap.data["X"].to_numpy(dtype=float)
    y = lap.data["Y"].to_numpy(dtype=float)
    mx = marks["X"].to_numpy(dtype=float)
    my = marks["Y"].to_numpy(dtype=float)

    # Squared distance from every mark to every line sample, marks down
    # the rows via broadcasting; argmin picks the closest sample per row.
    nearest = np.argmin(
        (x[None, :] - mx[:, None]) ** 2 + (y[None, :] - my[:, None]) ** 2,
        axis=1,
    )
    located = marks.copy()
    located["Distance"] = lap.data["Distance"].to_numpy(dtype=float)[nearest]
    return located


def label_corners(corners: list[Corner], marks: pd.DataFrame,
                  max_offset: float = 150.0) -> None:
    """Attach official corner designations to detected corners.

    marks: reference table with Distance [m] and Name columns in track
    order (e.g. from load_fastf1_corner_marks). Each mark goes to the
    nearest detected apex within max_offset metres; a detection that
    collects several marks is a merged chicane and takes the span, e.g.
    "T13-T15". Marks near no apex (flat-out kinks with no speed dip)
    are dropped. Purely cosmetic: nothing downstream depends on names.
    """
    apexes = np.array([c.apex for c in corners])
    collected: dict[int, list[str]] = {}
    for ref_dist, name in zip(marks["Distance"], marks["Name"]):
        i = int(np.argmin(np.abs(apexes - float(ref_dist))))
        if abs(apexes[i] - float(ref_dist)) <= max_offset:
            collected.setdefault(i, []).append(str(name))
    for i, names in collected.items():
        corners[i].name = names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}"
