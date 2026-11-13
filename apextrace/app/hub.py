"""The live side of the app: datagrams in, delta readings out.

One object owns the path from the wire to the browser. Datagrams arrive
from a UDP socket or from a capture being replayed; the F1 25 assembler
turns them into instants and, at each line crossing, into a finished
Lap; every instant is read against the reference lap by the live delta;
and the result is fanned out to whoever is connected.

A replayed capture goes through exactly the same code as a live socket,
paced by the arrival times stored in the file. That is what makes the
live path testable, and usable for reviewing a session as it happened,
without anyone sitting in the car.

Everything here runs in the event loop. The only blocking work is the
decoding of one datagram, which is microseconds.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import BinaryIO

from apextrace.app.store import SessionStore, StoredLap
from apextrace.live import LiveDelta, LiveSample
from apextrace.loaders.f1_25 import LapAssembler, PacketFormatError, replay_timed
from apextrace.loaders.f1_25.capture import write_header, write_record

log = logging.getLogger(__name__)

# Messages a slow client may have queued before samples start being
# dropped for it. A couple of seconds of a 60 Hz stream.
_CLIENT_QUEUE = 256

# A stream is "receiving" if a datagram arrived this recently [s].
_RECEIVING_WINDOW = 2.0


class LiveHub:
    """Decode a live stream, keep the running delta, broadcast both."""

    def __init__(self, store: SessionStore, step: float = 5.0) -> None:
        self.store = store
        self.source = "idle"
        self.packets = 0
        self.bad_packets = 0

        self._assembler = LapAssembler(step=step, on_sample=self._on_sample)
        self._skipped_reported = 0
        self._reference: StoredLap | None = None
        self._pinned = False
        self._delta: LiveDelta | None = None
        self._lap_number: int | None = None
        self._trace: list[tuple[float, float]] = []
        self._last_packet: float | None = None
        self._clients: set[asyncio.Queue] = set()
        self._recorder: BinaryIO | None = None
        self._record_started = 0.0

    # --- reference lap ------------------------------------------------

    @property
    def reference(self) -> StoredLap | None:
        return self._reference

    def set_reference(self, stored: StoredLap | None, pinned: bool) -> None:
        """Choose the lap the live delta is measured against.

        pinned=True is the user's explicit choice and stays until they
        change it. pinned=False is automatic mode, where the reference
        follows the best lap of the live session. Passing None with
        pinned=False returns to automatic mode.
        """
        self._pinned = pinned
        if stored is None and not pinned:
            stored = self._best_live_lap()
        self._apply_reference(stored)

    def _apply_reference(self, stored: StoredLap | None) -> None:
        if stored is self._reference:
            return
        self._reference = stored
        self._delta = None if stored is None else LiveDelta(stored.lap)
        # Readings against the old reference mean nothing against the
        # new one; the trace restarts from wherever the car is now.
        self._trace = []
        self._broadcast(
            {
                "type": "reference",
                "reference": None if stored is None else stored.summary(),
                "pinned": self._pinned,
            }
        )

    def _best_live_lap(self) -> StoredLap | None:
        """Fastest live lap of the circuit currently being driven."""
        track = self._assembler.track
        candidates = [
            stored
            for stored in self.store.all()
            if stored.origin == "live" and stored.lap.track == track
        ]
        return min(candidates, key=lambda s: s.lap.lap_time, default=None)

    # --- the stream ---------------------------------------------------

    def handle_datagram(self, datagram: bytes) -> None:
        """Take one datagram off the wire."""
        self.packets += 1
        self._last_packet = time.monotonic()
        if self._recorder is not None:
            # Recorded before decoding, like any capture: a datagram
            # this version cannot read is still part of the session.
            write_record(
                self._recorder, time.monotonic() - self._record_started, datagram
            )

        try:
            lap = self._assembler.feed(datagram)
        except PacketFormatError:
            self.bad_packets += 1
            return

        if lap is not None:
            self._on_lap(lap)
        for skipped in self._assembler.skipped[self._skipped_reported :]:
            self._broadcast(
                {
                    "type": "skipped",
                    "lap_number": skipped.lap_number,
                    "reason": skipped.reason,
                }
            )
        self._skipped_reported = len(self._assembler.skipped)

    def _on_sample(self, sample: LiveSample) -> None:
        """One instant of the lap in progress: read it and send it on."""
        if sample.lap_number != self._lap_number:
            self._lap_number = sample.lap_number
            self._trace = []

        message = {
            "type": "sample",
            "lap": sample.lap_number,
            "d": round(sample.distance, 1),
            "t": round(sample.time, 3),
            "speed": round(sample.speed, 1),
            "gear": sample.gear,
            "throttle": round(sample.throttle, 1),
            "brake": round(sample.brake, 1),
            "valid": sample.valid,
            "delta": None,
            "predicted": None,
            "trend": None,
        }
        # Before the line (formation, out lap) there is no lap to place
        # the car on, so no delta either.
        if self._delta is not None and sample.distance >= 0.0:
            # A flashback moves the car back; the trace follows it.
            while self._trace and self._trace[-1][0] >= sample.distance:
                self._trace.pop()
            reading = self._delta.update(sample.distance, sample.time)
            message["delta"] = round(reading.delta, 3)
            message["predicted"] = round(reading.predicted_lap_time, 3)
            message["trend"] = round(reading.trend, 3)
            self._trace.append((message["d"], message["delta"]))
        self._broadcast(message, droppable=True)

    def _on_lap(self, lap) -> None:
        """A lap just finished: keep it, and maybe promote it."""
        stored = self.store.add(lap, origin="live")
        previous = self._reference
        if not self._pinned:
            self._apply_reference(self._best_live_lap())
        self._broadcast(
            {
                "type": "lap",
                "lap": stored.summary(),
                "is_best": self._reference is stored and previous is not stored,
            }
        )

    # --- clients ------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_CLIENT_QUEUE)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._clients.discard(queue)

    def status(self) -> dict:
        receiving = (
            self._last_packet is not None
            and time.monotonic() - self._last_packet < _RECEIVING_WINDOW
        )
        return {
            "type": "status",
            "source": self.source,
            "receiving": receiving,
            "packets": self.packets,
            "bad_packets": self.bad_packets,
            "track": self._assembler.track,
            "recording": self._recorder is not None,
        }

    def snapshot(self) -> dict:
        """Everything a client needs to draw the live view from cold."""
        reference = self._reference
        return {
            "type": "hello",
            "status": self.status(),
            "reference": None if reference is None else reference.summary(),
            "pinned": self._pinned,
            "lap": self._lap_number,
            "trace": [list(point) for point in self._trace],
            "laps": [s.summary() for s in self.store.all() if s.origin == "live"],
        }

    def _broadcast(self, message: dict, droppable: bool = False) -> None:
        """Queue a message for every client, without ever blocking.

        A client that cannot keep up must not slow the stream down for
        the others, or for the decoder. Samples are expendable, the
        next one supersedes them, so a full queue simply skips them.
        Everything else (laps, reference changes) is state the client
        would be wrong without, so room is made by discarding the
        oldest queued message.
        """
        for queue in self._clients:
            if queue.full():
                if droppable:
                    continue
                queue.get_nowait()
            queue.put_nowait(message)

    # --- sources ------------------------------------------------------

    def start_recording(self, path: str | Path) -> None:
        """Write every datagram received from now on to a capture file."""
        self._recorder = Path(path).open("wb")
        write_header(self._recorder)
        self._record_started = time.monotonic()

    def stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    async def run_udp(self, port: int, host: str = "0.0.0.0") -> None:
        """Receive the game's UDP stream until cancelled."""
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpReceiver(self), local_addr=(host, port)
        )
        self.source = f"udp:{port}"
        try:
            await asyncio.Event().wait()
        finally:
            transport.close()

    async def run_replay(
        self, path: str | Path, speed: float = 1.0, loop: bool = False
    ) -> None:
        """Replay a capture file as if it were arriving live.

        speed scales the pacing (2.0 is twice as fast as recorded).
        With loop the capture restarts when it ends, which is handy
        for leaving a demo running.
        """
        path = Path(path)
        self.source = f"replay:{path.name}"
        while True:
            started = time.monotonic()
            for count, (arrival, datagram) in enumerate(replay_timed(path)):
                wait = arrival / speed - (time.monotonic() - started)
                if wait > 0.002:
                    await asyncio.sleep(wait)
                elif count % 64 == 0:
                    # Behind schedule or flat out: still let the
                    # clients be served.
                    await asyncio.sleep(0)
                self._safe_handle(datagram)
            if not loop:
                break

    async def run_status(self, interval: float = 1.0) -> None:
        """Tell clients periodically whether the stream is alive."""
        while True:
            await asyncio.sleep(interval)
            self._broadcast(self.status(), droppable=True)

    def _safe_handle(self, datagram: bytes) -> None:
        """handle_datagram, for callers that must survive anything."""
        try:
            self.handle_datagram(datagram)
        except Exception:  # noqa: BLE001 - one bad frame must not end the session
            self.bad_packets += 1
            log.exception("failed to process a datagram")


class _UdpReceiver(asyncio.DatagramProtocol):
    """asyncio's datagram callback, forwarding to the hub."""

    def __init__(self, hub: LiveHub) -> None:
        self._hub = hub

    def datagram_received(self, data: bytes, addr) -> None:
        self._hub._safe_handle(data)
