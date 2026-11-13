"""Turn a file the user handed over into canonical Laps.

The app accepts whatever the loaders accept, and tells the formats
apart by content rather than by file name: a capture renamed to
"friday.bin" is still a capture. Each format is one entry in a table of
(sniff, load) pairs, so a new loader becomes available to the app by
adding a line here and nothing anywhere else.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from apextrace.lap import Lap
from apextrace.loaders.f1_25 import load_f1_25_capture
from apextrace.loaders.f1_25.capture import MAGIC as F1_25_CAPTURE_MAGIC
from apextrace.loaders.motec_csv import is_motec_csv, load_motec_csv
from apextrace.storage import is_lap_table, load_laps

# How much of a file a sniffer gets to look at.
_SNIFF_BYTES = 4096

Loaded = tuple[list[Lap], list[str]]


@dataclass(frozen=True)
class LoadOptions:
    """What a loader may need that a file does not contain."""

    # Steering wheel angle at full lock [deg], for sources that log
    # steering in degrees (MoTeC exports).
    steering_lock: float | None = None


class UnsupportedFile(ValueError):
    """A file that none of the loaders recognises."""


def _is_f1_25_capture(head: bytes) -> bool:
    return head.startswith(F1_25_CAPTURE_MAGIC)


def _load_f1_25_capture(path: Path, options: LoadOptions) -> Loaded:
    laps, skipped = load_f1_25_capture(path)
    return laps, [str(item) for item in skipped]


def _load_motec_csv(path: Path, options: LoadOptions) -> Loaded:
    return load_motec_csv(path, steering_lock=options.steering_lock)


def _load_lap_table(path: Path, options: LoadOptions) -> Loaded:
    return load_laps(path), []


Loader = Callable[[Path, LoadOptions], Loaded]

# (name, does this look like one?, load it). First match wins.
FORMATS: list[tuple[str, Callable[[bytes], bool], Loader]] = [
    ("F1 25 capture", _is_f1_25_capture, _load_f1_25_capture),
    ("MoTeC CSV export", is_motec_csv, _load_motec_csv),
    ("apex-trace lap table", is_lap_table, _load_lap_table),
]


def load_file(
    path: str | Path, options: LoadOptions | None = None
) -> tuple[str, list[Lap], list[str]]:
    """Load every lap in a file.

    Returns the name of the format recognised, the laps, and notes on
    anything that was found but left out, so the UI can say why a
    session of six laps produced four.
    """
    path = Path(path)
    options = options or LoadOptions()
    with path.open("rb") as src:
        head = src.read(_SNIFF_BYTES)
    for name, sniff, load in FORMATS:
        if sniff(head):
            laps, notes = load(path, options)
            return name, laps, notes
    known = ", ".join(name for name, _, _ in FORMATS)
    raise UnsupportedFile(f"not a file this app can read (supported: {known})")
