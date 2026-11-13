"""Save canonical laps to a table on disk, and load them back.

A capture or an export is the record of a session, and decoding it
again always works. This is for the step after that: laps that are
already canonical, kept as one plain table so they reload in an instant
and can be handed to someone who has neither the game nor the decoder.
Because it stores canonical laps, it is the same file whichever loader
produced them.

The format is CSV, with two comment lines in front:

    # apextrace laps v1
    # [{"lap": 0, "label": "...", "source": "f1_25", "track": "Monza", ...}]
    Lap,Distance,Time,Speed,Throttle,Brake,Gear,RPM,Steering,X,Y
    0,0.0,0.0,290.0,100.0,0.0,7,11200.0,0.0,668.45,0.0
    ...

One row per grid point, the first column saying which lap it belongs
to, the other columns the union of the laps' channels (a cell is empty
where a lap does not have that channel; each lap's own channel list is
kept in the header, so it comes back with exactly the columns it had).
CSV rather than a binary
columnar format because it adds no dependency and opens in anything an
engineer already has: a spreadsheet, MATLAB, a text editor. A lap is a
thousand rows, so there is no size to worry about; a path ending in
``.gz`` is compressed all the same.

Floats are written at full precision and parsed with the round-trip
parser, so a lap read back is the lap that was written, bit for bit.
"""

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from apextrace.lap import CORE_CHANNELS, DISCRETE_CHANNELS, OPTIONAL_CHANNELS, Lap

MAGIC = "# apextrace laps v1"


def is_lap_table(head: bytes) -> bool:
    """Whether the first bytes of a file are a lap table's first line."""
    return head.startswith(MAGIC.encode())


def _open(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def write_laps(laps: Sequence[Lap], out: TextIO) -> None:
    """Write laps as a table to an open text stream."""
    if not laps:
        raise ValueError("no laps to save")

    # Canonical channels first, in schema order; anything else a lap
    # carries follows in the order it first appears.
    seen = [name for lap in laps for name in lap.channels]
    order = [n for n in (*CORE_CHANNELS, *OPTIONAL_CHANNELS) if n in seen]
    order += [n for n in dict.fromkeys(seen) if n not in order]

    meta = [
        {
            "lap": index,
            "label": lap.label,
            "source": lap.source,
            "track": lap.track,
            "car": lap.car,
            "channels": lap.channels,
        }
        for index, lap in enumerate(laps)
    ]
    frames = []
    for index, lap in enumerate(laps):
        frame = lap.data.reindex(columns=order)
        frame.insert(0, "Lap", index)
        frames.append(frame)

    out.write(MAGIC + "\n")
    out.write("# " + json.dumps(meta) + "\n")
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)


def save_laps(laps: Sequence[Lap], path: str | Path) -> None:
    """Save laps to a table file (``.csv``, or ``.csv.gz`` compressed)."""
    with _open(Path(path), "w") as out:
        write_laps(laps, out)


def load_laps(path: str | Path) -> list[Lap]:
    """Load the laps of a table file written by save_laps."""
    path = Path(path)
    with _open(path, "r") as src:
        if src.readline().rstrip("\r\n") != MAGIC:
            raise ValueError(f"{path.name} is not an apextrace lap table")
        try:
            meta = json.loads(src.readline().removeprefix("#"))
            listed = {name for entry in meta for name in entry["channels"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ValueError(f"{path.name} has a damaged lap list") from None
        # The default float parser is fast and can be off by one bit in
        # the last place; the round-trip one reads back what was written.
        table = pd.read_csv(src, float_precision="round_trip")

    missing = [
        name
        for name in ("Lap", *CORE_CHANNELS, *sorted(listed))
        if name not in table.columns
    ]
    if missing:
        raise ValueError(f"{path.name} lacks the columns {', '.join(missing)}")

    laps = []
    for entry in meta:
        data = table[table["Lap"] == entry["lap"]].drop(columns="Lap")
        if data.empty:
            raise ValueError(f"{path.name} lists lap {entry['lap']} but has no rows")
        # The lap's own channels, in its own order. The table holds the
        # union for all laps, so the rest of the columns are not its.
        data = data[entry["channels"]].reset_index(drop=True)
        for name in DISCRETE_CHANNELS:
            if name in data.columns:
                data[name] = data[name].astype(int)
        laps.append(
            Lap(
                data=data,
                label=entry["label"],
                source=entry["source"],
                track=entry.get("track"),
                car=entry.get("car"),
            )
        )
    return laps
