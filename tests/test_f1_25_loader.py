"""Lap assembly from the F1 25 UDP stream, and the engine on top of it.

The point of M4 is that a second source costs one loader and nothing
else, so the last test here is the one that matters: two laps that came
off the wire as UDP packets, run through the same delta engine as the
FastF1 laps, recovering a lap time difference known in closed form.
"""

import socket
import threading
import time

import numpy as np
import pytest

from apextrace.corners import segment_lap
from apextrace.delta import delta_time
from apextrace.lap import CORE_CHANNELS
from apextrace.loaders.f1_25 import (
    LapAssembler,
    PacketFormatError,
    load_f1_25_capture,
    load_f1_25_laps,
    record,
    replay,
    replay_timed,
    write_capture,
)
from apextrace.loaders.f1_25.synthetic import (
    session_packet,
    synthetic_session,
    synthetic_stream,
)

LAP_LENGTH = 3000.0

# Three corners, well apart, deep enough to detect. Speed returns to
# base at both ends so the trace is continuous across the timing line.
DIPS = ((600.0, 90.0, 90.0), (1500.0, 130.0, 110.0), (2400.0, 70.0, 80.0))


def speed_profile(base=280.0, dips=DIPS):
    """Distance [m] -> speed [km/h], Gaussian dips on a constant base."""

    def _speed(dist):
        dist = np.asarray(dist, dtype=float)
        speed = np.full(dist.shape, base)
        for centre, floor, width in dips:
            speed = speed - (base - floor) * np.exp(-(((dist - centre) / width) ** 2))
        return speed

    return _speed


@pytest.fixture
def stream():
    """Factory: a UDP stream of laps, and the exact lap time it implies."""

    def _stream(profile=None, laps=3, **kwargs):
        return synthetic_stream(
            profile or speed_profile(),
            lap_length=LAP_LENGTH,
            laps=laps,
            rate_hz=20.0,
            **kwargs,
        )

    return _stream


def test_a_lap_is_cut_for_every_completed_lap_of_the_stream(stream):
    """A lap only finishes when the next one starts, so N laps give N-1."""
    datagrams, _ = stream(laps=3)
    laps, skipped = load_f1_25_laps(datagrams)
    assert len(laps) == 2
    assert skipped == []


def test_lap_time_matches_the_kinematics_it_was_built_from(stream):
    """The end anchor pins the axis to the game's own official time."""
    datagrams, lap_time = stream()
    laps, _ = load_f1_25_laps(datagrams)
    # The game reports lap times in whole milliseconds; that rounding is
    # the only difference allowed between the two.
    assert laps[0].lap_time == pytest.approx(lap_time, abs=1e-3)


def test_each_lap_of_a_session_gets_its_own_official_time():
    """The time reported after the line belongs to the lap before it."""
    slower = list(DIPS)
    slower[0] = (600.0, 70.0, 90.0)
    profiles = [speed_profile(), speed_profile(dips=tuple(slower)), speed_profile()]
    datagrams, lap_times = synthetic_session(
        profiles, lap_length=LAP_LENGTH, rate_hz=20.0
    )
    laps, _ = load_f1_25_laps(datagrams)

    assert [lap.lap_time for lap in laps] == pytest.approx(lap_times[:2], abs=1e-3)
    assert laps[1].lap_time > laps[0].lap_time


def test_the_distance_axis_is_closed_at_both_ends(stream):
    """Time sampling leaves both ends short; the loader anchors them."""
    datagrams, _ = stream()
    laps, _ = load_f1_25_laps(datagrams, step=5.0)
    dist = laps[0].data["Distance"].to_numpy()

    assert dist[0] == 0.0
    assert dist[-1] == pytest.approx(LAP_LENGTH)
    assert laps[0].data["Time"].iloc[0] == 0.0
    # Uniform grid apart from the exact final point.
    assert np.allclose(np.diff(dist[:-1]), 5.0)


def test_canonical_schema_and_units(stream):
    datagrams, _ = stream()
    laps, _ = load_f1_25_laps(datagrams)
    lap = laps[0]

    assert set(CORE_CHANNELS) <= set(lap.channels)
    assert {"RPM", "Steering", "X", "Y"} <= set(lap.channels)
    assert lap.source == "f1_25"
    assert lap.track == "Monza"

    # Pedals and steering arrive as fractions and must land in percent.
    assert 0.0 <= lap.data["Throttle"].min() and lap.data["Throttle"].max() <= 100.0
    assert lap.data["Throttle"].max() == pytest.approx(100.0)
    assert 0.0 <= lap.data["Brake"].min() and lap.data["Brake"].max() <= 100.0
    assert lap.data["Brake"].max() > 1.0
    assert lap.data["Steering"].abs().max() <= 100.0
    assert lap.data["Steering"].min() < 0.0 < lap.data["Steering"].max()

    # Gear is discrete: zero-order hold, never interpolated to 4.3.
    gear = lap.data["Gear"].to_numpy()
    assert np.array_equal(gear, gear.astype(int))


def test_position_channels_describe_the_ground_plane(stream):
    """Canonical Y comes from world z; world y is height and is dropped."""
    datagrams, _ = stream()
    laps, _ = load_f1_25_laps(datagrams)
    lap = laps[0]

    radius = np.hypot(lap.data["X"].to_numpy(), lap.data["Y"].to_numpy())
    assert radius == pytest.approx(LAP_LENGTH / (2.0 * np.pi), rel=1e-2)


def test_invalidated_laps_are_dropped_with_a_reason(stream):
    datagrams, _ = stream(laps=3, invalid_laps=(1,))
    laps, skipped = load_f1_25_laps(datagrams)

    assert len(laps) == 1
    assert [s.lap_number for s in skipped] == [1]
    assert "invalidated" in skipped[0].reason


def test_invalidated_laps_can_be_kept_on_request(stream):
    datagrams, _ = stream(laps=3, invalid_laps=(1,))
    laps, skipped = load_f1_25_laps(datagrams, keep_invalid=True)
    assert len(laps) == 2
    assert skipped == []


def test_pit_laps_are_dropped(stream):
    """An in or out lap has no comparable distance axis, as in FastF1."""
    datagrams, _ = stream(laps=3, pit_laps=(1,))
    laps, skipped = load_f1_25_laps(datagrams)

    assert len(laps) == 1
    assert [s.lap_number for s in skipped] == [1]
    assert "pit" in skipped[0].reason


def test_motion_is_optional(stream):
    """The player can switch the motion packet off; laps still decode."""
    datagrams, _ = stream(include_motion=False)
    laps, _ = load_f1_25_laps(datagrams)

    assert set(CORE_CHANNELS) <= set(laps[0].channels)
    assert "X" not in laps[0].channels


def test_a_stray_datagram_does_not_cost_the_session(stream):
    datagrams, _ = stream()
    polluted = [b"not an f1 packet at all"] + datagrams
    laps, _ = load_f1_25_laps(polluted)
    assert len(laps) == 2


def test_strict_mode_surfaces_a_bad_datagram(stream):
    datagrams, _ = stream()
    with pytest.raises(PacketFormatError):
        load_f1_25_laps([b"not an f1 packet at all"] + datagrams, strict=True)


def test_frames_reassemble_when_packets_arrive_out_of_order(stream):
    """The packets of one frame are separate datagrams; order is not given.

    Reversing each burst puts motion last, after the two packets that
    make a frame usable. Assembling a frame the moment those two land
    would drop it and cost the lap its position channels.
    """
    datagrams, lap_time = stream()
    swapped = list(datagrams)
    for i in range(1, len(swapped) - 2, 3):
        swapped[i : i + 3] = reversed(swapped[i : i + 3])

    laps, skipped = load_f1_25_laps(swapped)
    assert len(laps) == 2
    assert skipped == []
    assert {"X", "Y"} <= set(laps[0].channels)
    assert laps[0].lap_time == pytest.approx(lap_time, abs=1e-3)


def test_a_flashback_keeps_only_the_attempt_that_reached_the_line(stream):
    """Rewinding mid-lap must not leave the abandoned stretch in the lap."""
    datagrams, lap_time = stream(laps=3)
    per_lap = (len(datagrams) - 1) // 3
    # Re-send a stretch from the middle of lap 1: frame counter and lap
    # timer both jump backwards, as they do when the player rewinds.
    cut = 1 + (per_lap // 2 // 3) * 3
    rewound = datagrams[:cut] + datagrams[cut - 150 :]

    laps, skipped = load_f1_25_laps(rewound)
    assert len(laps) == 2
    assert skipped == []
    assert laps[0].lap_time == pytest.approx(lap_time, abs=1e-3)
    assert np.all(np.diff(laps[0].data["Time"].to_numpy()) > 0.0)


def test_the_assembler_returns_each_lap_as_it_finishes(stream):
    """A live session needs the lap at the line, not at the end."""
    datagrams, _ = stream()
    assembler = LapAssembler()
    finished = [lap for lap in map(assembler.feed, datagrams) if lap is not None]
    assert len(finished) == 2


def test_capture_round_trip(tmp_path, stream):
    """A capture is raw bytes on disk: what goes in is what comes out."""
    datagrams, _ = stream()
    path = tmp_path / "session.f1cap"
    write_capture(path, datagrams, rate_hz=20.0)

    assert list(replay(path)) == datagrams
    assert [t for t, _ in replay_timed(path)][:3] == [0.0, 0.05, 0.10]

    laps, _ = load_f1_25_capture(path)
    assert len(laps) == 2


def test_a_truncated_capture_still_reads_what_it_has(tmp_path, stream):
    """Ctrl-C during recording ends on a partial record, not a corrupt file."""
    datagrams, _ = stream()
    path = tmp_path / "cut.f1cap"
    write_capture(path, datagrams)
    with path.open("ab") as out:
        out.write(b"\x00" * 6)  # a record header cut in half

    assert list(replay(path)) == datagrams


def test_a_foreign_file_is_not_read_as_a_capture(tmp_path):
    path = tmp_path / "nope.f1cap"
    path.write_bytes(b"some other file entirely")
    with pytest.raises(ValueError, match="not an apextrace capture"):
        list(replay(path))


def test_record_writes_a_readable_capture(tmp_path):
    """The recorder and the reader agree, over a real UDP socket.

    Loopback only, and the sole test here that touches a socket: the
    recorder is the one piece that cannot be exercised by handing bytes
    to a function, and it is also the piece that runs unattended while
    someone is driving.
    """
    packet = session_packet()
    path = tmp_path / "live.f1cap"

    def send_once():
        # UDP does not queue for a socket that does not exist yet, so
        # give the recorder a moment to bind before sending.
        time.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(packet, ("127.0.0.1", port))

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))  # let the OS pick a free port
        port = probe.getsockname()[1]

    sender = threading.Thread(target=send_once)
    sender.start()
    try:
        count = record(path, host="127.0.0.1", port=port, duration=0.0, timeout=5.0)
    finally:
        sender.join()

    assert count == 1
    assert list(replay(path)) == [packet]


def test_recording_a_silent_port_ends_quietly_with_nothing(tmp_path):
    """A stream that never starts is a zero-packet capture, not a crash."""
    path = tmp_path / "silent.f1cap"
    assert record(path, host="127.0.0.1", port=0, duration=None, timeout=0.2) == 0
    assert list(replay(path)) == []


def test_the_engine_reads_f1_25_laps_unmodified(stream):
    """The M4 claim: a new source costs one loader and nothing else.

    Two laps identical but for one slower corner, both assembled from
    UDP packets, compared by the delta engine that was written for
    FastF1 data and has not been touched since.
    """
    slower = list(DIPS)
    slower[1] = (1500.0, 110.0, 110.0)  # 20 km/h slower through corner 2

    fast_datagrams, fast_time = stream(speed_profile())
    slow_datagrams, slow_time = stream(speed_profile(dips=tuple(slower)))
    lap_a = load_f1_25_laps(fast_datagrams)[0][0]
    lap_b = load_f1_25_laps(slow_datagrams)[0][0]

    # Corner detection finds the three dips, on both laps.
    corners_a = segment_lap(lap_a)
    assert len(corners_a) == 3
    assert [round(c.apex / 100) * 100 for c in corners_a] == [600, 1500, 2400]
    assert len(segment_lap(lap_b)) == 3

    # The full-lap delta recovers the difference in lap time.
    delta = delta_time(lap_a, lap_b)
    assert float(delta.iloc[-1]) == pytest.approx(slow_time - fast_time, abs=5e-3)

    # And it is lost where the slow corner is, not spread over the lap.
    def at(s):
        return float(np.interp(s, delta.index, delta.to_numpy()))

    assert at(1200.0) == pytest.approx(0.0, abs=5e-3)
    assert at(1800.0) == pytest.approx(slow_time - fast_time, abs=5e-2)
