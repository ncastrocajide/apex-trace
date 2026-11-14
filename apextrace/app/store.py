"""The laps an app session is holding, whatever they came from.

The store is the app's version of the project's central idea: once a
lap is canonical nothing downstream asks where it came from. A lap
recorded live a minute ago, one decoded from a capture file and one
downloaded from F1 timing all sit in the same list and can be compared
with each other.

Access is locked because two kinds of code touch it: request handlers,
which run in worker threads, and the live hub, which runs in the event
loop.
"""

import itertools
import threading
from dataclasses import dataclass

import pandas as pd

from apextrace.lap import Lap


@dataclass(frozen=True)
class StoredLap:
    """A lap in the session, with the handle the UI refers to it by."""

    id: str
    lap: Lap
    origin: str  # "live", "upload:<file name>" or "fastf1"
    # Official corner marks of the lap's circuit, when the source has
    # them (F1 timing does), and a key naming the event they belong to.
    # Marks are positions on one event's map, so two laps can only
    # share them if their keys match.
    marks: pd.DataFrame | None = None
    marks_key: str | None = None

    def summary(self) -> dict:
        """What a lap list needs to show, without the telemetry."""
        data = self.lap.data
        return {
            "id": self.id,
            "label": self.lap.label,
            "source": self.lap.source,
            "track": self.lap.track,
            "car": self.lap.car,
            "origin": self.origin,
            "lap_time": self.lap.lap_time,
            "length": float(data["Distance"].iloc[-1]),
            "channels": self.lap.channels,
        }


class SessionStore:
    """In-memory lap list. Nothing is persisted: captures are the record."""

    def __init__(self) -> None:
        self._laps: dict[str, StoredLap] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(
        self,
        lap: Lap,
        origin: str,
        marks: pd.DataFrame | None = None,
        marks_key: str | None = None,
    ) -> StoredLap:
        with self._lock:
            stored = StoredLap(
                id=f"lap{next(self._ids)}",
                lap=lap,
                origin=origin,
                marks=marks,
                marks_key=marks_key,
            )
            self._laps[stored.id] = stored
            return stored

    def get(self, lap_id: str) -> StoredLap | None:
        with self._lock:
            return self._laps.get(lap_id)

    def remove(self, lap_id: str) -> bool:
        with self._lock:
            return self._laps.pop(lap_id, None) is not None

    def all(self) -> list[StoredLap]:
        """Every lap, in the order it was added."""
        with self._lock:
            return list(self._laps.values())
