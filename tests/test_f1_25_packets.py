"""Wire-format tests for the F1 25 decoder.

Byte layouts are the one thing in this project that cannot be derived
from first principles: they are a published contract, and getting an
offset wrong decodes neighbouring fields into plausible numbers rather
than failing. So the sizes are pinned against the published figures,
and every field the loader reads is checked by round-tripping a packet
built from known values.
"""

import struct

import pytest

from apextrace.loaders.f1_25 import PacketFormatError
from apextrace.loaders.f1_25 import packets as pk
from apextrace.loaders.f1_25.synthetic import session_packet, synthetic_stream


def _flat(dist):
    """Constant 200 km/h; the argument only fixes the array shape."""
    return 200.0 + 0.0 * dist


def _flat_stream(**kwargs):
    """One lap of packets at a constant speed, round a 1 km circle."""
    datagrams, _ = synthetic_stream(_flat, lap_length=1000.0, laps=1, **kwargs)
    return datagrams


def _first(datagrams, packet_id):
    """First datagram of a given packet type, with its header."""
    for datagram in datagrams:
        header = pk.decode_header(datagram)
        if header.packet_id == packet_id:
            return datagram, header
    raise AssertionError(f"no packet with id {packet_id} in the stream")


def test_struct_sizes_match_the_published_spec():
    """The F1 25 spec fixes these; a typo in a format string moves them."""
    assert pk.HEADER_SIZE == 29
    assert pk.MOTION_STRIDE == 60
    assert pk.LAP_DATA_STRIDE == 57
    assert pk.CAR_TELEMETRY_STRIDE == 60
    assert pk.MAX_CARS == 22


def test_header_round_trip():
    datagram = session_packet()
    header = pk.decode_header(datagram)
    assert header.packet_format == pk.PACKET_FORMAT
    assert header.game_year == 25
    assert header.packet_id == pk.PACKET_SESSION
    assert header.player_index == 0


def test_wrong_packet_format_is_rejected():
    """An older game format would otherwise decode as quiet nonsense."""
    datagram = bytearray(session_packet())
    struct.pack_into("<H", datagram, 0, 2024)
    with pytest.raises(PacketFormatError, match="2024"):
        pk.decode_header(bytes(datagram))


def test_short_datagram_is_rejected():
    with pytest.raises(PacketFormatError, match="shorter than"):
        pk.decode_header(b"\xe9\x07" + b"\x00" * 5)


def test_session_packet_carries_the_circuit():
    info = pk.decode_session(session_packet(track_id=11, track_length=5793.0))
    assert info.track_id == 11
    assert info.track_length == pytest.approx(5793.0)
    assert pk.track_name(11) == "Monza"


def test_unknown_track_id_gets_a_placeholder_not_a_guess():
    assert pk.track_name(200) == "track_200"


def test_lap_data_fields_decode_at_the_right_offsets():
    datagrams = _flat_stream()
    datagram, header = _first(datagrams, pk.PACKET_LAP_DATA)
    lap = pk.decode_lap_data(datagram, header)

    assert lap.lap_number == 1
    assert lap.last_lap_time == 0.0  # no previous lap yet
    assert lap.current_lap_time == 0.0  # first frame is the line crossing
    assert lap.lap_distance == pytest.approx(0.0)
    assert lap.total_distance == pytest.approx(0.0)
    assert lap.pit_status == 0
    assert lap.invalid is False
    assert lap.driver_status == pk.DRIVER_FLYING_LAP


def test_car_telemetry_fields_decode_at_the_right_offsets():
    datagrams = _flat_stream()
    datagram, header = _first(datagrams, pk.PACKET_CAR_TELEMETRY)
    telemetry = pk.decode_telemetry(datagram, header)

    assert telemetry.speed == pytest.approx(200.0)
    assert 0.0 <= telemetry.throttle <= 1.0
    assert 0.0 <= telemetry.brake <= 1.0
    assert -1.0 <= telemetry.steer <= 1.0
    assert 1 <= telemetry.gear <= 8
    assert telemetry.rpm > 0


def test_motion_maps_the_ground_plane_to_x_and_z():
    """World y is vertical in the game's axes, so it must not be used."""
    datagrams = _flat_stream()
    datagram, header = _first(datagrams, pk.PACKET_MOTION)
    motion = pk.decode_motion(datagram, header)

    radius = 1000.0 / (2.0 * 3.141592653589793)
    assert motion.world_y == 0.0
    assert (motion.world_x**2 + motion.world_z**2) ** 0.5 == pytest.approx(
        radius, rel=1e-3
    )


def test_the_player_is_found_at_a_non_zero_car_index():
    """Every array is indexed by player_car_index, not assumed to be 0."""
    datagrams = _flat_stream(player_index=7)
    datagram, header = _first(datagrams, pk.PACKET_CAR_TELEMETRY)
    assert header.player_index == 7
    assert pk.decode_telemetry(datagram, header).speed == pytest.approx(200.0)


def test_truncated_car_array_is_rejected():
    datagrams = _flat_stream()
    datagram, header = _first(datagrams, pk.PACKET_CAR_TELEMETRY)
    with pytest.raises(PacketFormatError, match="too short"):
        pk.decode_telemetry(datagram[: pk.HEADER_SIZE + 4], header)
