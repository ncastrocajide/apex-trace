"""F1 25 loader: the game's live UDP telemetry stream as canonical Laps.

The second source adapter, and the one that tests the claim the project
is built on. Where the FastF1 loader reads a finished session from an
archive, this one reads a live broadcast: the data arrives as binary
packets at up to 60 Hz, sampled in time rather than in distance, with
no lap boundaries and no lap times attached. Everything specific to
that lives behind this package, and what comes out the other side is
the same canonical Lap the analysis engine has always taken. The engine
was not modified to accept it.

    from apextrace.loaders.f1_25 import record, load_f1_25_capture

    record("monza.f1cap", duration=600)        # while driving
    laps, skipped = load_f1_25_capture("monza.f1cap")   # afterwards

Or live, one Lap at a time as the driver crosses the line:

    from apextrace.loaders.f1_25 import listen, stream_f1_25_laps

    for lap in stream_f1_25_laps(listen()):
        print(lap.label, lap.lap_time)
"""

from apextrace.loaders.f1_25.capture import (
    DEFAULT_PORT,
    listen,
    record,
    replay,
    replay_timed,
    write_capture,
)
from apextrace.loaders.f1_25.loader import (
    LapAssembler,
    SkippedLap,
    load_f1_25_capture,
    load_f1_25_laps,
    stream_f1_25_laps,
)
from apextrace.loaders.f1_25.packets import PacketFormatError

__all__ = [
    "DEFAULT_PORT",
    "LapAssembler",
    "PacketFormatError",
    "SkippedLap",
    "listen",
    "load_f1_25_capture",
    "load_f1_25_laps",
    "record",
    "replay",
    "replay_timed",
    "stream_f1_25_laps",
    "write_capture",
]
