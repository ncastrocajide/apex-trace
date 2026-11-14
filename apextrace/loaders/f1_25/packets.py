"""Wire format of the F1 25 UDP telemetry stream.

The game broadcasts a burst of packets per physics frame, each one a
fixed-layout little-endian C struct with no padding: a 29 byte header
followed by an array of per-car structs, indexed by car. All we ever
want is the player's car, so decoding is a header parse plus one slice
at ``header_size + player_index * stride``.

Two deliberate restrictions keep this module small and hard to break:

Only four packet types are decoded (motion, session, lap data, car
telemetry). The other twelve carry tyre temperatures, damage, setups
and classification: real data, but nothing the canonical Lap has a
column for, so parsing them would be dead weight.

Only the *prefix* of each per-car struct is decoded, up to the last
field we need. The full layouts are still declared below because they
define the stride, but a field we do not read is never unpacked. This
is not premature optimisation; it is the part that survives contact
with next year's game. Codemasters have grown these structs at the tail
almost every season while leaving the leading fields alone, so a
decoder that only reads prefixes keeps working after a bump that would
break a full-struct parse, and the stride constants are the single
place that needs revisiting.

Layouts are from the official F1 25 UDP specification (packet format
2025). Sizes are asserted at import: the published per-car struct sizes
are 29 (header), 60 (motion), 57 (lap data) and 60 (car telemetry), so
a typo in a format string fails loudly here rather than silently
decoding garbage from the wrong offset.
"""

import struct
from typing import NamedTuple

# The value of m_packetFormat this module knows how to read.
PACKET_FORMAT = 2025

# Packet ids. Stable across F1 2020-25; only the ones we decode.
PACKET_MOTION = 0
PACKET_SESSION = 1
PACKET_LAP_DATA = 2
PACKET_CAR_TELEMETRY = 6

# Cars per packet array. 22 rather than 20, for the game modes that
# field more entries than a real grid.
MAX_CARS = 22

# m_driverStatus. A lap is only comparable if the driver was actually
# racing it, so the loader keeps FLYING_LAP and ON_TRACK and drops the
# rest (in garage, in lap, out lap).
DRIVER_IN_GARAGE = 0
DRIVER_FLYING_LAP = 1
DRIVER_IN_LAP = 2
DRIVER_OUT_LAP = 3
DRIVER_ON_TRACK = 4
DRIVER_RACING = (DRIVER_FLYING_LAP, DRIVER_ON_TRACK)

# m_trackId. Track ids are cosmetic here: they name the Lap, nothing
# reads them. Only the ids that have been stable since F1 2020 are
# listed; anything else falls back to "track_<id>" rather than risk
# labelling a lap with the wrong circuit.
TRACK_NAMES = {
    0: "Melbourne",
    1: "Paul Ricard",
    2: "Shanghai",
    3: "Sakhir",
    4: "Catalunya",
    5: "Monaco",
    6: "Montreal",
    7: "Silverstone",
    8: "Hockenheim",
    9: "Hungaroring",
    10: "Spa",
    11: "Monza",
    12: "Singapore",
    13: "Suzuka",
    14: "Abu Dhabi",
    15: "Texas",
    16: "Brazil",
    17: "Austria",
    18: "Sochi",
    19: "Mexico",
    20: "Baku",
    21: "Sakhir Short",
    22: "Silverstone Short",
    23: "Texas Short",
    24: "Suzuka Short",
    25: "Hanoi",
    26: "Zandvoort",
    27: "Imola",
    28: "Portimao",
    29: "Jeddah",
    30: "Miami",
    31: "Las Vegas",
    32: "Losail",
}


def track_name(track_id: int) -> str:
    """Circuit name for a track id, or a stable placeholder if unknown."""
    return TRACK_NAMES.get(track_id, f"track_{track_id}")


# --- header ---------------------------------------------------------
#
# uint16 m_packetFormat            uint8  m_packetId
# uint8  m_gameYear                uint64 m_sessionUID
# uint8  m_gameMajorVersion        float  m_sessionTime
# uint8  m_gameMinorVersion        uint32 m_frameIdentifier
# uint8  m_packetVersion           uint32 m_overallFrameIdentifier
#                                  uint8  m_playerCarIndex
#                                  uint8  m_secondaryPlayerCarIndex

_HEADER = struct.Struct("<HBBBBBQfIIBB")
HEADER_SIZE = _HEADER.size


class Header(NamedTuple):
    """The fields of the packet header that the loader acts on."""

    packet_format: int
    game_year: int
    packet_id: int
    session_uid: int
    session_time: float  # s since session start
    frame: int  # physics frame; ties a burst of packets together
    player_index: int  # index of the player's car in every array


class Telemetry(NamedTuple):
    """Player car inputs and engine state for one frame."""

    speed: float  # km/h
    throttle: float  # 0 to 1
    steer: float  # -1 to 1, fraction of full lock
    brake: float  # 0 to 1
    gear: int  # 1-8, 0 neutral, -1 reverse
    rpm: int


class LapData(NamedTuple):
    """Player car timing and position along the lap for one frame."""

    last_lap_time: float  # s, the lap before this one; 0 if none yet
    current_lap_time: float  # s since the timing line
    lap_distance: float  # m since the timing line; negative before it
    total_distance: float  # m since the session started, never resets
    lap_number: int
    pit_status: int  # 0 none, 1 pitting, 2 in pit area
    invalid: bool  # lap already invalidated (track limits, contact)
    driver_status: int


class Motion(NamedTuple):
    """Player car position in world space [m].

    The game's world axes are x/z on the ground plane and y vertical;
    the canonical schema's X/Y are the ground plane, so the loader maps
    canonical Y from world z.
    """

    world_x: float
    world_y: float  # vertical
    world_z: float


class SessionInfo(NamedTuple):
    """The session facts worth carrying onto a Lap."""

    track_id: int
    track_length: float  # m, the game's own figure for the circuit
    session_type: int


# --- per-car struct prefixes ----------------------------------------
#
# Each pair below is (prefix we decode, stride of the full struct). The
# full field list is given so the stride can be re-derived by hand when
# the format changes.

# CarMotionData: 6 floats (world position xyz, world velocity xyz),
# 6 int16 (forward and right direction vectors), 6 floats (g force
# lateral/longitudinal/vertical, yaw, pitch, roll).
_MOTION = struct.Struct("<3f")
MOTION_STRIDE = 60

# LapData, decoded as far as m_driverStatus. Tail after the prefix:
# uint8 m_resultStatus, uint8 m_pitLaneTimerActive, uint16
# m_pitLaneTimeInLaneInMS, uint16 m_pitStopTimerInMS, uint8
# m_pitStopShouldServePen, float m_speedTrapFastestSpeed, uint8
# m_speedTrapFastestLap.
_LAP_DATA = struct.Struct("<IIHBHBHBHBfffBBBBBBBBBBBBB")
LAP_DATA_STRIDE = 57

# CarTelemetryData, decoded as far as m_engineRPM. Tail after the
# prefix: uint8 m_drs, uint8 m_revLightsPercent, uint16
# m_revLightsBitValue, uint16 m_brakesTemperature[4], uint8
# m_tyresSurfaceTemperature[4], uint8 m_tyresInnerTemperature[4],
# uint16 m_engineTemperature, float m_tyresPressure[4], uint8
# m_surfaceType[4].
_CAR_TELEMETRY = struct.Struct("<HfffBbH")
CAR_TELEMETRY_STRIDE = 60

# PacketSessionData has no per-car array and has churned every season,
# but its first eight bytes have not moved since F1 2020: uint8
# m_weather, int8 m_trackTemperature, int8 m_airTemperature, uint8
# m_totalLaps, uint16 m_trackLength, uint8 m_sessionType, int8
# m_trackId. Everything after that is left unread on purpose.
_SESSION = struct.Struct("<BbbBHBb")

assert HEADER_SIZE == 29, HEADER_SIZE
assert _MOTION.size <= MOTION_STRIDE
assert _LAP_DATA.size == 45, _LAP_DATA.size
assert _LAP_DATA.size <= LAP_DATA_STRIDE
assert _CAR_TELEMETRY.size == 18, _CAR_TELEMETRY.size
assert _CAR_TELEMETRY.size <= CAR_TELEMETRY_STRIDE
assert _SESSION.size == 8, _SESSION.size


class PacketFormatError(ValueError):
    """A datagram is not an F1 25 UDP packet this module can read."""


def decode_header(datagram: bytes) -> Header:
    """Parse the packet header, or raise PacketFormatError.

    Anything arriving on a telemetry port that is not a well-formed F1
    25 packet is an error worth surfacing, not a packet to skip: the
    usual cause is the game being set to broadcast an older format,
    which would otherwise decode as plausible-looking nonsense.
    """
    if len(datagram) < HEADER_SIZE:
        raise PacketFormatError(
            f"datagram is {len(datagram)} bytes, shorter than the "
            f"{HEADER_SIZE} byte header"
        )
    (
        packet_format,
        game_year,
        _major,
        _minor,
        _version,
        packet_id,
        session_uid,
        session_time,
        frame,
        _overall_frame,
        player_index,
        _secondary_index,
    ) = _HEADER.unpack_from(datagram)

    if packet_format != PACKET_FORMAT:
        raise PacketFormatError(
            f"packet format {packet_format} is not {PACKET_FORMAT}: set the "
            "game's UDP format to F1 25, or use a loader for that year"
        )
    return Header(
        packet_format=packet_format,
        game_year=game_year,
        packet_id=packet_id,
        session_uid=session_uid,
        session_time=session_time,
        frame=frame,
        player_index=player_index,
    )


def _player_slice(datagram: bytes, header: Header, stride: int, size: int) -> bytes:
    """Bytes of the player's entry in a per-car array."""
    if header.player_index >= MAX_CARS:
        raise PacketFormatError(
            f"player car index {header.player_index} is outside the "
            f"{MAX_CARS} car array"
        )
    start = HEADER_SIZE + header.player_index * stride
    if len(datagram) < start + size:
        raise PacketFormatError(
            f"packet {header.packet_id} is {len(datagram)} bytes, too short to "
            f"hold car {header.player_index} at offset {start}"
        )
    return datagram[start : start + size]


def decode_telemetry(datagram: bytes, header: Header) -> Telemetry:
    """Player car telemetry from a PacketCarTelemetryData datagram."""
    speed, throttle, steer, brake, _clutch, gear, rpm = _CAR_TELEMETRY.unpack(
        _player_slice(datagram, header, CAR_TELEMETRY_STRIDE, _CAR_TELEMETRY.size)
    )
    return Telemetry(
        speed=float(speed),
        throttle=throttle,
        steer=steer,
        brake=brake,
        gear=gear,
        rpm=rpm,
    )


def decode_lap_data(datagram: bytes, header: Header) -> LapData:
    """Player car lap data from a PacketLapData datagram."""
    fields = _LAP_DATA.unpack(
        _player_slice(datagram, header, LAP_DATA_STRIDE, _LAP_DATA.size)
    )
    return LapData(
        last_lap_time=fields[0] / 1000.0,
        current_lap_time=fields[1] / 1000.0,
        lap_distance=fields[10],
        total_distance=fields[11],
        lap_number=fields[14],
        pit_status=fields[15],
        invalid=bool(fields[18]),
        driver_status=fields[25],
    )


def decode_motion(datagram: bytes, header: Header) -> Motion:
    """Player car world position from a PacketMotionData datagram."""
    world_x, world_y, world_z = _MOTION.unpack(
        _player_slice(datagram, header, MOTION_STRIDE, _MOTION.size)
    )
    return Motion(world_x=world_x, world_y=world_y, world_z=world_z)


def decode_session(datagram: bytes) -> SessionInfo:
    """Circuit and session type from a PacketSessionData datagram."""
    if len(datagram) < HEADER_SIZE + _SESSION.size:
        raise PacketFormatError("session packet is too short to hold the track fields")
    _weather, _track_temp, _air_temp, _laps, length, session_type, track_id = (
        _SESSION.unpack_from(datagram, HEADER_SIZE)
    )
    return SessionInfo(
        track_id=track_id,
        track_length=float(length),
        session_type=session_type,
    )
