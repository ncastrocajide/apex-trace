"""Load laps from a MoTeC i2 CSV export into the canonical format.

The third source adapter, and a third kind of source: not an archive
behind an API, not a live binary stream, but a text file a person
exported by hand. MoTeC i2 is where Assetto Corsa telemetry ends up
(the ACTI and Telemetrick apps both log in MoTeC's format), and it is
equally where a real car's MoTeC logger ends up, so this one loader
covers both.

The file is a header of key/value lines, a row of channel names, a row
of units, then one row per sample, sampled in time:

    "Format","MoTeC CSV File",,,"Workbook",""
    "Venue","monza",,,"Worksheet",""
    "Vehicle","ferrari_488_gt3"
    ...
    "Beacon Markers","93.512 186.204 278.731"

    "Time","Distance","Ground Speed","Throttle Pos",...
    "s","m","km/h","%",...

    "0.000","0.0","87.2","100.0",...

MoTeC publish no specification for it; their own advice is to export a
file and copy its structure. So the parser is deliberately tolerant
about everything the format leaves loose (blank lines, which header
keys are present, column order) and strict about the two things it
cannot work without: the row of channel names starting with "Time",
and a channel it can recognise for each core canonical channel.

Three things are specific to this source.

Channel names are whatever the logging setup called them. A table of
known aliases covers the common ones; ``channel_map`` overrides it for
anything else.

Units are stated per channel, so they are converted rather than
assumed: a speed in m/s or mph becomes km/h.

Steering usually arrives as a wheel angle in degrees, and the canonical
channel is a fraction of full lock. The conversion needs the car's
lock, which the file does not contain, so it is a parameter; without
it the steering channel is left out rather than normalised by a guess.

Laps are cut at the beacon markers, the line-crossing times i2 keeps in
the header. They are exact to the millisecond and fall between samples,
so both ends of every lap are interpolated onto the beacon instant:
distance zero at time zero, and the lap time at the lap's full length.
Only laps with a beacon at both ends are returned; what comes before
the first and after the last is an out lap and an unfinished lap.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from apextrace.lap import Lap, resample_to_distance

# Canonical channel -> names it is known to be logged under, compared
# case-insensitively, first match wins.
CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "Distance": ("Distance", "Lap Distance", "Lap Dist"),
    "Speed": ("Ground Speed", "Speed", "Corr Speed", "Vehicle Speed", "GPS Speed"),
    "Throttle": ("Throttle Pos", "Throttle", "Throttle Position"),
    "Brake": ("Brake Pos", "Brake", "Brake Position"),
    "Gear": ("Gear",),
    "RPM": ("Engine RPM", "RPM", "Engine Speed"),
    "Steering": ("Steering Angle", "Steered Angle", "Steering Wheel Angle", "Steering"),
}

# World position, up to three axes. Which two are the ground plane is
# decided from the data, not from the names (see _ground_plane).
POSITION_ALIASES: tuple[tuple[str, ...], ...] = (
    ("Car Coord X", "Car Pos X", "Position X", "GPS X"),
    ("Car Coord Y", "Car Pos Y", "Position Y", "GPS Y"),
    ("Car Coord Z", "Car Pos Z", "Position Z", "GPS Z"),
)

# Multipliers into km/h, by the unit string of the speed channel.
SPEED_UNITS = {"km/h": 1.0, "kph": 1.0, "kmh": 1.0, "m/s": 3.6, "mph": 1.609344}

# A lap of fewer samples than this is a glitch between two beacons.
MIN_SAMPLES = 50

_FORMAT_NAME = "MoTeC CSV File"


class MotecFormatError(ValueError):
    """A file that is not a MoTeC CSV export this loader can read."""


def is_motec_csv(head: bytes) -> bool:
    """Whether the first bytes of a file look like a MoTeC CSV export."""
    return _FORMAT_NAME.encode() in head[:256]


def _read_rows(path: Path) -> list[list[str]]:
    """The file as rows of cells, blank lines dropped."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # i2 writes the Windows code page: a degree sign in a unit is
        # enough to make a file invalid UTF-8.
        text = raw.decode("cp1252")
    rows = csv.reader(text.splitlines())
    return [row for row in rows if any(cell.strip() for cell in row)]


def _number(cell: str) -> float:
    """A data cell as a float; empty or unparseable cells are NaN."""
    cell = cell.strip().replace(",", "")  # "1,234.5": thousands separator
    if not cell:
        return np.nan
    try:
        return float(cell)
    except ValueError:
        return np.nan


def _parse(path: Path) -> tuple[dict[str, str], pd.DataFrame, dict[str, str]]:
    """Header fields, the sample table, and the unit of each channel."""
    rows = _read_rows(path)
    if not rows or len(rows[0]) < 2 or _FORMAT_NAME not in rows[0][1]:
        raise MotecFormatError(f"{path.name} is not a MoTeC CSV export")

    names_at = next(
        (
            i
            for i, row in enumerate(rows)
            if row[0].strip().lower() == "time" and len(row) > 1
        ),
        None,
    )
    if names_at is None or names_at + 2 >= len(rows):
        raise MotecFormatError(
            f"{path.name} has no channel table: expected a row of channel "
            'names starting with "Time", a row of units, then the samples'
        )

    # Header lines are "key","value" with, on most of them, a second
    # key/value pair further along the same line.
    meta: dict[str, str] = {}
    for row in rows[:names_at]:
        cells = [cell.strip() for cell in row]
        meta[cells[0]] = cells[1] if len(cells) > 1 else ""
        if len(cells) > 5 and cells[4]:
            meta[cells[4]] = cells[5]

    names = [cell.strip() for cell in rows[names_at]]
    unit_cells = [cell.strip() for cell in rows[names_at + 1]]
    units = dict(zip(names, unit_cells + [""] * len(names), strict=False))
    width = len(names)
    samples = np.array(
        [
            [_number(cell) for cell in (row + [""] * width)[:width]]
            for row in rows[names_at + 2 :]
        ],
        dtype=float,
    )
    table = pd.DataFrame(samples, columns=names)
    table = table[table["Time"].notna()].reset_index(drop=True)
    if len(table) < 2:
        raise MotecFormatError(f"{path.name} contains no samples")
    return meta, table, units


def _find(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """The first candidate present among the columns, ignoring case."""
    by_lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _percent(values: np.ndarray, unit: str) -> np.ndarray:
    """A pedal channel in percent, whether logged as 0-100 or as 0-1."""
    if unit != "%" and np.nanmax(np.abs(values)) <= 1.0:
        return values * 100.0
    return values


def _steering(
    values: np.ndarray, unit: str, steering_lock: float | None
) -> np.ndarray | None:
    """Steering as percent of full lock, or None if it cannot be had."""
    unit = unit.lower()
    if unit == "%":
        return values
    if unit in ("deg", "°", "degrees", "rad"):
        if steering_lock is None:
            return None
        degrees = np.degrees(values) if unit == "rad" else values
        return degrees / steering_lock * 100.0
    if np.nanmax(np.abs(values)) <= 1.0:
        return values * 100.0  # a plain fraction of lock
    return None


def _ground_plane(table: pd.DataFrame) -> tuple[str, str] | None:
    """The two position channels that span the ground.

    Sims and loggers disagree on which world axis points up (Assetto
    Corsa uses y, others z), and the channel names do not say. The
    data does: a circuit is hundreds of metres across and rises by
    tens, so of three position axes the one with the smallest extent
    is the vertical. With only two axes logged there is nothing to
    decide.
    """
    present = [
        name
        for name in (_find(list(table.columns), names) for names in POSITION_ALIASES)
        if name is not None
    ]
    if len(present) < 2:
        return None
    if len(present) == 3:
        extent = {name: float(np.ptp(table[name].to_numpy())) for name in present}
        present.remove(min(extent, key=extent.get))
    return present[0], present[1]


def _unwrap(distance: np.ndarray) -> np.ndarray:
    """A distance channel made continuous across lap resets.

    Some setups log distance since the line rather than since the start
    of the log, so it drops back to zero every lap. Lap cutting wants
    one running odometer, so each drop is undone by carrying forward
    the reading it dropped from.
    """
    drops = np.diff(distance, prepend=distance[0])
    carried = np.where(drops < -100.0, -drops, 0.0)
    return distance + np.cumsum(carried)


def _beacons(meta: dict[str, str], times: np.ndarray) -> np.ndarray:
    """Line-crossing times that fall within the logged samples [s]."""
    marks = []
    for token in meta.get("Beacon Markers", "").replace(",", " ").split():
        try:
            marks.append(float(token))
        except ValueError:
            continue
    marks = np.unique(np.array(marks, dtype=float))
    # A beacon within a sample of either end still bounds a lap.
    period = float(np.median(np.diff(times)))
    return marks[(marks >= times[0] - period) & (marks <= times[-1] + period)]


def _at(time: float, times: np.ndarray, raw: pd.DataFrame) -> dict[str, float]:
    """Every channel read at one instant between samples.

    Continuous channels are interpolated; the gear is held from the
    sample before, as it is everywhere else.
    """
    row = {}
    before = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, None))
    for name in raw.columns:
        values = raw[name].to_numpy(dtype=float)
        row[name] = (
            values[before] if name == "Gear" else float(np.interp(time, times, values))
        )
    return row


def load_motec_csv(
    path: str | Path,
    step: float = 5.0,
    steering_lock: float | None = None,
    channel_map: dict[str, str] | None = None,
    whole_file: bool = False,
    source: str = "motec_csv",
) -> tuple[list[Lap], list[str]]:
    """Load every complete lap of a MoTeC CSV export as canonical Laps.

    Returns the laps and a list of notes: anything the loader left out
    or had to work around (a steering channel it could not convert, a
    distance it had to integrate), so the caller knows what it got.

    step:
        Distance grid spacing in metres.
    steering_lock:
        Steering wheel angle at full lock, one side, in degrees. Needed
        to convert a steering channel logged in degrees; for Assetto
        Corsa it is half the car's total wheel rotation.
    channel_map:
        Canonical channel -> name of the column to read it from, for
        logs whose names the alias table does not know, e.g.
        ``{"Brake": "Brake Pres Front %"}``.
    whole_file:
        Treat the file as exactly one lap, ignoring beacons. For an
        export of a single lap's range.
    source:
        Recorded on each Lap. Pass "assetto_corsa" for sim exports if
        the distinction from a real logger matters downstream.
    """
    path = Path(path)
    meta, table, units = _parse(path)
    columns = list(table.columns)
    notes: list[str] = []
    overrides = channel_map or {}

    def column(canonical: str) -> str | None:
        if canonical in overrides:
            if overrides[canonical] not in columns:
                raise MotecFormatError(
                    f"channel_map points {canonical} at {overrides[canonical]!r}, "
                    f"which is not in the file; it has: {', '.join(columns)}"
                )
            return overrides[canonical]
        return _find(columns, CHANNEL_ALIASES[canonical])

    found = {name: column(name) for name in CHANNEL_ALIASES}
    missing = [n for n in ("Speed", "Throttle", "Brake", "Gear") if found[n] is None]
    if missing:
        raise MotecFormatError(
            f"{path.name} has no channel recognised as {', '.join(missing)}; "
            f"name them with channel_map. The file has: {', '.join(columns)}"
        )

    times = table["Time"].to_numpy(dtype=float)
    speed_unit = units.get(found["Speed"], "").lower()
    if speed_unit not in SPEED_UNITS:
        raise MotecFormatError(
            f"speed channel {found['Speed']!r} is in {speed_unit or 'no unit'!r}; "
            f"expected one of {', '.join(SPEED_UNITS)}"
        )
    speed = table[found["Speed"]].to_numpy(dtype=float) * SPEED_UNITS[speed_unit]

    raw = pd.DataFrame(
        {
            "Speed": speed,
            "Throttle": _percent(
                table[found["Throttle"]].to_numpy(dtype=float),
                units.get(found["Throttle"], ""),
            ),
            "Brake": _percent(
                table[found["Brake"]].to_numpy(dtype=float),
                units.get(found["Brake"], ""),
            ),
            "Gear": table[found["Gear"]].to_numpy(dtype=float),
        }
    )
    if found["RPM"] is not None:
        raw["RPM"] = table[found["RPM"]].to_numpy(dtype=float)

    if found["Steering"] is not None:
        unit = units.get(found["Steering"], "")
        steering = _steering(
            table[found["Steering"]].to_numpy(dtype=float), unit, steering_lock
        )
        if steering is None:
            notes.append(
                f"steering channel {found['Steering']!r} ({unit or 'no unit'}) left "
                "out: converting it to a fraction of lock needs steering_lock"
            )
        else:
            raw["Steering"] = steering

    plane = _ground_plane(table)
    if plane is not None:
        raw["X"] = table[plane[0]].to_numpy(dtype=float)
        raw["Y"] = table[plane[1]].to_numpy(dtype=float)

    if found["Distance"] is not None:
        odometer = _unwrap(table[found["Distance"]].to_numpy(dtype=float))
    else:
        # No distance logged: integrate the speed, which is what a
        # logger's own distance channel is anyway.
        odometer = cumulative_trapezoid(speed / 3.6, times, initial=0.0)
        notes.append("no distance channel: distance integrated from speed")
    raw["Odometer"] = odometer

    if whole_file:
        bounds = np.array([times[0], times[-1]])
    else:
        bounds = _beacons(meta, times)
        if bounds.size == 0:
            bounds = np.array([times[0], times[-1]])
            notes.append("no beacon markers: the whole file read as one lap")
        elif bounds.size == 1:
            raise MotecFormatError(
                f"{path.name} has a single beacon marker, so no lap in it has "
                "both a start and an end; export a range of whole laps, or "
                "pass whole_file=True if the file is exactly one lap"
            )

    venue = meta.get("Venue", "") or None
    vehicle = meta.get("Vehicle", "") or None
    laps: list[Lap] = []
    for number, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True), 1):
        inside = (times > start) & (times < end)
        if int(np.sum(inside)) < MIN_SAMPLES:
            notes.append(
                f"lap {number}: only {int(np.sum(inside))} samples between its "
                "beacons, not kept"
            )
            continue

        # Both ends are read at the beacon instant, between samples, so
        # the lap starts at distance 0, time 0 and ends on its lap time.
        rows = [_at(start, times, raw), *raw[inside].to_dict("records")]
        rows.append(_at(end, times, raw))
        lap_raw = pd.DataFrame(rows)
        lap_raw.insert(
            0, "Time", np.concatenate(([0.0], times[inside] - start, [end - start]))
        )
        lap_raw.insert(0, "Distance", lap_raw.pop("Odometer") - rows[0]["Odometer"])

        label = " ".join(part for part in (vehicle, venue, f"lap {number}") if part)
        laps.append(
            Lap(
                data=resample_to_distance(lap_raw, step=step),
                label=label,
                source=source,
                track=venue,
                car=vehicle,
            )
        )
    return laps, notes
