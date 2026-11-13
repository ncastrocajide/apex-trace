"""Turn a file the user handed over into canonical Laps.

The app accepts whatever the loaders accept, and tells the formats
apart by content rather than by file name: a capture renamed to
"friday.bin" is still a capture. Each format is one entry in a table of
(sniff, load) pairs, so a new loader becomes available to the app by
adding a line here and nothing anywhere else.
"""

from collections.abc import Callable
from pathlib import Path

from apextrace.lap import Lap
from apextrace.loaders.f1_25 import load_f1_25_capture
from apextrace.loaders.f1_25.capture import MAGIC as F1_25_CAPTURE_MAGIC

# How much of a file a sniffer gets to look at.
_SNIFF_BYTES = 4096

Loaded = tuple[list[Lap], list[str]]


class UnsupportedFile(ValueError):
    """A file that none of the loaders recognises."""


def _is_f1_25_capture(head: bytes) -> bool:
    return head.startswith(F1_25_CAPTURE_MAGIC)


def _load_f1_25_capture(path: Path) -> Loaded:
    laps, skipped = load_f1_25_capture(path)
    return laps, [str(item) for item in skipped]


# (name, does this look like one?, load it). First match wins.
FORMATS: list[tuple[str, Callable[[bytes], bool], Callable[[Path], Loaded]]] = [
    ("F1 25 capture", _is_f1_25_capture, _load_f1_25_capture),
]


def load_file(path: str | Path) -> tuple[str, list[Lap], list[str]]:
    """Load every lap in a file.

    Returns the name of the format recognised, the laps, and a note for
    each lap that was found but left out, so the UI can say why a
    session of six laps produced four.
    """
    path = Path(path)
    with path.open("rb") as src:
        head = src.read(_SNIFF_BYTES)
    for name, sniff, load in FORMATS:
        if sniff(head):
            laps, notes = load(path)
            return name, laps, notes
    known = ", ".join(name for name, _, _ in FORMATS)
    raise UnsupportedFile(f"not a file this app can read (supported: {known})")
