"""Turn an F1 25 UDP packet stream into canonical Laps.

The game does not send laps, it sends frames: a burst of packets every
physics tick, each carrying the instantaneous state of every car. Three
things have to happen before the analysis engine can see a Lap.

Frames have to be reassembled. Lap data, car telemetry and motion
arrive as separate datagrams that share a frame identifier; a sample is
only complete once the two mandatory ones have landed. Motion is
optional because the player can switch it off in the game's UDP menu,
and the engine only needs X/Y for track maps and mark projection.

Laps have to be cut. The lap number in the lap data packet is the only
boundary marker, so a lap ends when the next one's first frame arrives.
That frame is worth keeping: it reports the finished lap's official
time and sits physically just past the timing line, which is exactly
what the lap's far end needs.

The distance axis has to be closed. Telemetry is sampled in time, so
the first sample of a lap is already a few metres past the line and the
last one falls a few metres short of it. Both ends are anchored rather
than left ragged: the game's lap timer resets at the line, so distance
0 at time 0 is exact, and the far end is pinned at the lap length with
the official lap time. Without that the lap time would be short by up
to a sample interval, which at 30 Hz and 300 km/h is nearly 3 m of
missing track. This is the same correction the FastF1 loader gets from
interpolate_edges, done with the timing the game already provides.

Lap length comes from the two distance counters, not from the track
table: lap data carries both the distance since the line and the
distance since the session started, so their difference is the reading
at the line and successive laps differ by exactly one lap. The game's
own track length is only a fallback for when that difference is
unusable.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from apextrace.lap import Lap, resample_to_distance
from apextrace.live import LiveSample
from apextrace.loaders.f1_25 import packets
from apextrace.loaders.f1_25.capture import replay
from apextrace.loaders.f1_25.packets import (
    Header,
    LapData,
    Motion,
    PacketFormatError,
    SessionInfo,
    Telemetry,
)

# A lap needs enough samples for the corner detector to have anything to
# work with; a handful of frames is a glitch, not a lap.
MIN_SAMPLES = 50

# How far ahead of the frame being buffered a stray packet may sit
# before it is treated as debris from a flashback rather than as UDP
# reordering. Wide enough to absorb a few swapped datagrams, far
# narrower than the seconds a flashback rewinds.
_MAX_LOOKAHEAD = 10


@dataclass
class _Sample:
    """One reassembled frame: the player's state at a single instant."""

    lap: LapData
    telemetry: Telemetry
    motion: Motion | None = None


@dataclass
class SkippedLap:
    """A lap that was cut but not returned, and the reason why."""

    lap_number: int
    reason: str

    def __str__(self) -> str:
        return f"lap {self.lap_number}: {self.reason}"


class LapAssembler:
    """Stateful decoder: feed it datagrams, it hands back finished Laps.

    Incremental by design. A finished lap is returned the moment the
    next lap's first frame arrives, so the same object serves both
    offline analysis of a capture file and a live session, where the
    lap has to be available as soon as the driver crosses the line.
    """

    def __init__(
        self,
        step: float = 5.0,
        keep_invalid: bool = False,
        keep_pit_laps: bool = False,
        on_sample: Callable[[LiveSample], None] | None = None,
    ) -> None:
        """
        step:
            Distance grid spacing in metres, as for any loader.
        keep_invalid:
            Keep laps the game flagged invalid (track limits, contact).
            Off by default: an invalidated lap is usually not a lap you
            want to compare against, but the telemetry is still real.
        keep_pit_laps:
            Keep in and out laps. Off by default, matching the FastF1
            loader: a lap through the pits has no comparable distance
            axis.
        on_sample:
            Called with every reassembled frame as a LiveSample, for
            consumers that cannot wait for the lap to finish (the live
            delta). Finished Laps are still returned by feed.
        """
        self.step = step
        self.keep_invalid = keep_invalid
        self.keep_pit_laps = keep_pit_laps
        self.on_sample = on_sample
        self.skipped: list[SkippedLap] = []

        self._session: SessionInfo | None = None
        self._pending: dict[int, dict] = {}
        self._samples: list[_Sample] = []
        self._lap_number: int | None = None

    @property
    def track(self) -> str | None:
        """Circuit name, once a session packet has been seen."""
        if self._session is None:
            return None
        return packets.track_name(self._session.track_id)

    def feed(self, datagram: bytes) -> Lap | None:
        """Feed one UDP datagram; return a Lap if one just finished."""
        header = packets.decode_header(datagram)

        if header.packet_id == packets.PACKET_SESSION:
            self._session = packets.decode_session(datagram)
            return None

        if header.packet_id == packets.PACKET_LAP_DATA:
            part = {"lap": packets.decode_lap_data(datagram, header)}
        elif header.packet_id == packets.PACKET_CAR_TELEMETRY:
            part = {"telemetry": packets.decode_telemetry(datagram, header)}
        elif header.packet_id == packets.PACKET_MOTION:
            part = {"motion": packets.decode_motion(datagram, header)}
        else:
            return None

        return self._collect(header, part)

    def _collect(self, header: Header, part: dict) -> Lap | None:
        """Buffer a decoded packet, and assemble the frames it completes.

        A frame is only assembled once a packet from a later frame
        proves its burst is over. Emitting as soon as lap data and
        telemetry have both landed would be a frame quicker, but the
        packets of one burst arrive in no guaranteed order, so a motion
        packet that happened to come last would be thrown away and the
        lap would silently lose its X/Y channels.
        """
        self._pending.setdefault(header.frame, {}).update(part)

        # A flashback rewinds the frame counter, stranding whatever was
        # buffered for frames that will now never arrive.
        for key in [k for k in self._pending if k > header.frame + _MAX_LOOKAHEAD]:
            del self._pending[key]

        finished = None
        for key in sorted(k for k in self._pending if k < header.frame):
            frame = self._pending.pop(key)
            if "lap" not in frame or "telemetry" not in frame:
                continue  # a burst that lost a packet; skip the instant
            # At most one lap can finish here: cutting one takes at
            # least MIN_SAMPLES frames, and this loop spans a handful.
            finished = (
                self._add_sample(
                    _Sample(
                        lap=frame["lap"],
                        telemetry=frame["telemetry"],
                        motion=frame.get("motion"),
                    )
                )
                or finished
            )
        return finished

    def _add_sample(self, sample: _Sample) -> Lap | None:
        """Append a sample to the current lap, cutting at a lap change."""
        if self.on_sample is not None:
            self.on_sample(_live_sample(sample))

        if self._lap_number is None:
            self._lap_number = sample.lap.lap_number

        if sample.lap.lap_number != self._lap_number:
            finished = self._cut(sample)
            self._lap_number = sample.lap.lap_number
            self._samples = [sample]
            return finished

        # A flashback rewinds the car to an earlier point of the same
        # lap. The buffered samples from the abandoned attempt describe
        # track the driver is about to cover again, so drop them and
        # keep the attempt that actually reached the line.
        if self._samples and sample.lap.current_lap_time < (
            self._samples[-1].lap.current_lap_time
        ):
            self._samples = [
                kept
                for kept in self._samples
                if kept.lap.current_lap_time < sample.lap.current_lap_time
            ]

        self._samples.append(sample)
        return None

    def _cut(self, first_of_next: _Sample) -> Lap | None:
        """Finish the buffered lap, or record why it was dropped."""
        samples, self._samples = self._samples, []
        number = self._lap_number if self._lap_number is not None else 0

        reason = self._reject(samples, first_of_next)
        if reason is not None:
            self.skipped.append(SkippedLap(lap_number=number, reason=reason))
            return None
        return self._build(samples, first_of_next, number)

    def _reject(self, samples: list[_Sample], first_of_next: _Sample) -> str | None:
        """Why this lap is not comparable, or None if it is."""
        usable = [s for s in samples if s.lap.lap_distance >= 0.0]
        if len(usable) < MIN_SAMPLES:
            return f"only {len(usable)} usable samples, need {MIN_SAMPLES}"
        if not self.keep_pit_laps and any(s.lap.pit_status != 0 for s in usable):
            return "in or out lap (pit status set)"
        if not self.keep_pit_laps and any(
            s.lap.driver_status not in packets.DRIVER_RACING for s in usable
        ):
            return "not a racing lap (driver status)"
        if not self.keep_invalid and usable[-1].lap.invalid:
            return "invalidated by the game"
        if first_of_next.lap.last_lap_time <= 0.0:
            return "no official lap time reported for it"
        return None

    def _build(
        self, samples: list[_Sample], first_of_next: _Sample, number: int
    ) -> Lap | None:
        """Assemble the canonical Lap, anchoring both ends of the axis."""
        usable = [s for s in samples if s.lap.lap_distance >= 0.0]
        lap_time = first_of_next.lap.last_lap_time

        length = self._lap_length(usable, first_of_next)
        if length is None or length <= usable[-1].lap.lap_distance:
            self.skipped.append(
                SkippedLap(
                    lap_number=number,
                    reason="could not establish a lap length from the odometer",
                )
            )
            return None

        rows = [_row(s) for s in usable]

        # Start anchor: the game's lap timer is zero at the line, so
        # distance 0 at time 0 is exact. Only the channels are held from
        # the first real sample, over the few metres before it.
        if usable[0].lap.lap_distance > 0.0:
            rows.insert(0, {**rows[0], "Distance": 0.0, "Time": 0.0})

        # End anchor: the official lap time at the exact lap length. The
        # channels come from the frame just past the line, which is the
        # same instant physically, so nothing is invented here either.
        rows.append({**_row(first_of_next), "Distance": length, "Time": lap_time})

        raw = pd.DataFrame(rows)
        if raw[["Distance", "Time"]].isna().to_numpy().any():
            self.skipped.append(
                SkippedLap(lap_number=number, reason="missing distance or time samples")
            )
            return None

        track = self.track
        return Lap(
            data=resample_to_distance(raw, step=self.step),
            label=f"F1 25 {track or 'unknown track'} lap {number}",
            source="f1_25",
            track=track,
            # The team is only in the participants packet, which this
            # loader does not decode; a sim lap is the player's car.
            car=None,
        )

    def _lap_length(
        self, usable: list[_Sample], first_of_next: _Sample
    ) -> float | None:
        """Distance the game's counters measured for this lap [m].

        total_distance runs continuously across the timing line while
        lap_distance resets at it, so their difference is the reading
        where the lap started. The gap between one lap's reading and
        the next is the length of the lap. The median over the lap
        absorbs the odd corrupt frame; the game's track length is the
        fallback when the difference is unusable.
        """
        starts = np.array(
            [s.lap.total_distance - s.lap.lap_distance for s in usable], dtype=float
        )
        next_start = first_of_next.lap.total_distance - first_of_next.lap.lap_distance
        length = float(next_start - np.median(starts))
        if length > 0.0:
            return length
        if self._session is not None and self._session.track_length > 0.0:
            return self._session.track_length
        return None


def _live_sample(sample: _Sample) -> LiveSample:
    """The instant a live consumer sees, in canonical units."""
    motion = sample.motion
    return LiveSample(
        lap_number=sample.lap.lap_number,
        distance=sample.lap.lap_distance,
        time=sample.lap.current_lap_time,
        speed=sample.telemetry.speed,
        throttle=sample.telemetry.throttle * 100.0,
        brake=sample.telemetry.brake * 100.0,
        gear=sample.telemetry.gear,
        steering=sample.telemetry.steer * 100.0,
        x=None if motion is None else motion.world_x,
        y=None if motion is None else motion.world_z,
        valid=(
            not sample.lap.invalid
            and sample.lap.pit_status == 0
            and sample.lap.driver_status in packets.DRIVER_RACING
        ),
    )


def _row(sample: _Sample) -> dict[str, float]:
    """One canonical row from a reassembled frame, in canonical units."""
    row = {
        "Distance": sample.lap.lap_distance,
        "Time": sample.lap.current_lap_time,
        "Speed": sample.telemetry.speed,
        # The game reports pedals and steering as fractions of full
        # travel; the canonical schema wants percent.
        "Throttle": sample.telemetry.throttle * 100.0,
        "Brake": sample.telemetry.brake * 100.0,
        "Steering": sample.telemetry.steer * 100.0,
        "Gear": float(sample.telemetry.gear),
        "RPM": float(sample.telemetry.rpm),
    }
    if sample.motion is not None:
        # World y is vertical in the game's axes; the canonical X/Y are
        # the ground plane, so canonical Y comes from world z.
        row["X"] = sample.motion.world_x
        row["Y"] = sample.motion.world_z
    return row


def load_f1_25_laps(
    datagrams: Iterable[bytes],
    step: float = 5.0,
    keep_invalid: bool = False,
    keep_pit_laps: bool = False,
    strict: bool = False,
) -> tuple[list[Lap], list[SkippedLap]]:
    """Decode a whole stream of UDP datagrams into canonical Laps.

    Returns the laps and the ones that were skipped with their reasons,
    because "the capture contained four laps and you got one" is a
    question worth being able to answer without re-running anything.

    strict:
        Raise on a datagram that does not decode. Off by default: a
        capture can begin mid-packet or contain a stray datagram from
        another program on the same port, and one bad datagram should
        not cost the whole session.
    """
    assembler = LapAssembler(
        step=step, keep_invalid=keep_invalid, keep_pit_laps=keep_pit_laps
    )
    laps: list[Lap] = []
    for datagram in datagrams:
        try:
            lap = assembler.feed(datagram)
        except PacketFormatError:
            if strict:
                raise
            continue
        if lap is not None:
            laps.append(lap)
    return laps, assembler.skipped


def load_f1_25_capture(
    path: str | Path,
    step: float = 5.0,
    keep_invalid: bool = False,
    keep_pit_laps: bool = False,
    strict: bool = False,
) -> tuple[list[Lap], list[SkippedLap]]:
    """Decode a capture file recorded by apextrace.loaders.f1_25.capture."""
    return load_f1_25_laps(
        replay(path),
        step=step,
        keep_invalid=keep_invalid,
        keep_pit_laps=keep_pit_laps,
        strict=strict,
    )


def stream_f1_25_laps(
    datagrams: Iterator[bytes],
    step: float = 5.0,
    keep_invalid: bool = False,
    keep_pit_laps: bool = False,
) -> Iterator[Lap]:
    """Yield Laps as they finish, for a live session.

    Wraps a live socket (or a replay) so the caller sees each lap the
    moment the driver crosses the line, rather than at the end of the
    session.
    """
    assembler = LapAssembler(
        step=step, keep_invalid=keep_invalid, keep_pit_laps=keep_pit_laps
    )
    for datagram in datagrams:
        try:
            lap = assembler.feed(datagram)
        except PacketFormatError:
            continue
        if lap is not None:
            yield lap
