"""Receive the F1 25 UDP stream, and record it for later.

The game broadcasts telemetry to a UDP port and never repeats itself:
miss a packet and that instant is gone. So the first thing to do with
the stream is put it on disk verbatim, before any parsing, and analyse
the file afterwards. Recording is then the only part that has to work
while the driver is on track, and a decoding bug costs a re-run of the
loader rather than a re-run of the session.

The capture format is deliberately dumb: a short magic header, then one
length-prefixed record per datagram with the arrival time. Storing the
raw bytes means a capture stays readable by any tool that knows the F1
spec, and keeping the timestamps means a capture can later be replayed
at the speed it arrived, which is what makes the live delta testable
without sitting in the car.
"""

import socket
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

# Default port of the game's UDP output, and its own default.
DEFAULT_PORT = 20777

# Largest F1 25 packet is well under 2 KiB; 4 KiB leaves headroom and
# still shows up as a truncation if a bigger datagram ever arrives.
MAX_DATAGRAM = 4096

MAGIC = b"APXCAP01"
_RECORD = struct.Struct("<dI")  # arrival time [s from start], length


def write_header(out: BinaryIO) -> None:
    """Start a capture file."""
    out.write(MAGIC)


def write_record(out: BinaryIO, arrival: float, datagram: bytes) -> None:
    """Append one datagram and its arrival time [s from capture start]."""
    out.write(_RECORD.pack(arrival, len(datagram)))
    out.write(datagram)


def listen(
    port: int = DEFAULT_PORT, host: str = "0.0.0.0", timeout: float | None = None
) -> Iterator[bytes]:
    """Yield datagrams from the game's UDP stream as they arrive.

    host defaults to every interface, which is what a console or a
    second PC broadcasting to this machine needs. timeout is the
    seconds to wait for a single datagram before giving up, so a
    caller can fail with a clear message instead of blocking forever
    when the game is not actually sending.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        if timeout is not None:
            sock.settimeout(timeout)
        while True:
            yield sock.recv(MAX_DATAGRAM)


def record(
    path: str | Path,
    port: int = DEFAULT_PORT,
    host: str = "0.0.0.0",
    duration: float | None = None,
    timeout: float | None = 30.0,
) -> int:
    """Record the UDP stream to a capture file; return the packet count.

    Runs until duration seconds have passed, until the stream goes
    quiet for timeout seconds, or until interrupted with Ctrl-C, which
    is the normal way to end a session. All three are ordinary
    endings, not errors: a gap in the stream means the player quit to
    the menu, and half an hour of driving should not end in a
    traceback. Whatever was received is already on disk, because each
    record is written as it arrives, so a capture that stopped early
    is a shorter capture rather than a corrupt one. A return of zero
    means nothing ever arrived, which is the only case a caller needs
    to treat as a failure.
    """
    path = Path(path)
    started = time.monotonic()
    count = 0

    with path.open("wb") as out:
        write_header(out)
        try:
            for datagram in listen(port=port, host=host, timeout=timeout):
                write_record(out, time.monotonic() - started, datagram)
                out.flush()
                count += 1
                if duration is not None and time.monotonic() - started >= duration:
                    break
        except (KeyboardInterrupt, TimeoutError):
            pass
    return count


def write_capture(
    path: str | Path, datagrams: list[bytes], rate_hz: float = 90.0
) -> None:
    """Write datagrams to a capture file with evenly spaced arrivals.

    For captures that did not come off a socket: the synthetic streams
    used by the demo and the tests. rate_hz is datagrams per second,
    so a 30 Hz stream of three-packet bursts is 90.
    """
    with Path(path).open("wb") as out:
        write_header(out)
        for i, datagram in enumerate(datagrams):
            write_record(out, i / rate_hz, datagram)


def replay(path: str | Path) -> Iterator[bytes]:
    """Yield the datagrams of a capture file, in the order recorded."""
    for _arrival, datagram in replay_timed(path):
        yield datagram


def replay_timed(path: str | Path) -> Iterator[tuple[float, bytes]]:
    """Yield (arrival time [s from capture start], datagram) pairs."""
    path = Path(path)
    with path.open("rb") as src:
        magic = src.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not an apextrace capture file")
        while True:
            head = src.read(_RECORD.size)
            if len(head) < _RECORD.size:
                # A capture cut short by Ctrl-C or a full disk ends on a
                # partial record; everything before it is still good.
                return
            arrival, length = _RECORD.unpack(head)
            payload = src.read(length)
            if len(payload) < length:
                return
            yield arrival, payload
