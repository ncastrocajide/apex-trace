"""Generate a well-formed F1 25 UDP stream from an analytic lap.

The loader has to be testable by someone who does not have the game
running, which in practice means everyone: continuous integration, a
laptop on a train, anyone reading the repository. So the packets are
synthesised here, at the wire level, and fed to the real decoder. What
gets exercised is the actual byte layout, the actual frame reassembly
and the actual lap cutting; nothing is stubbed out.

Only the kinematics are exact. Distance, time and speed are consistent
by construction, in the same way as the synthetic laps in the test
suite: time is the integral of slowness over distance, so the lap time
the loader recovers can be checked against a number known in closed
form. Everything else, the pedals, the gear, the steering, the shape of
the track, is a plausible filler derived from the speed trace. It is
there so the channels exist and get resampled, not because a synthetic
driver brakes like a real one.

This is a development and test utility that happens to live in the
package rather than in the test suite, because the demo example and the
app's replay mode use it too: it is the only way to show the engine
running on F1 25 data without owning F1 25.
"""

import struct
from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid

from apextrace.loaders.f1_25.packets import (
    CAR_TELEMETRY_STRIDE,
    LAP_DATA_STRIDE,
    MAX_CARS,
    MOTION_STRIDE,
    PACKET_CAR_TELEMETRY,
    PACKET_FORMAT,
    PACKET_LAP_DATA,
    PACKET_MOTION,
    PACKET_SESSION,
)

SpeedProfile = Callable[[np.ndarray], np.ndarray]

_HEADER = struct.Struct("<HBBBBBQfIIBB")
_MOTION = struct.Struct("<3f")
_LAP_DATA = struct.Struct("<IIHBHBHBHBfffBBBBBBBBBBBBB")
_CAR_TELEMETRY = struct.Struct("<HfffBbH")
_SESSION = struct.Struct("<BbbBHBb")

# Bytes each packet carries after its per-car array: the time trial car
# indices for lap data, the MFD panel and suggested gear for telemetry.
# Zero-filled here. The loader never reads past the player's slice, so
# these only matter for producing packets of a believable length.
_LAP_DATA_TAIL = 2
_CAR_TELEMETRY_TAIL = 3
_MOTION_TAIL = 0

# The session packet is a few hundred bytes of session state; only its
# first eight are read, so the rest is zero padding of a plausible size.
_SESSION_LENGTH = 753

# Metres from brake release back to full throttle in the filler pedals.
_THROTTLE_RAMP = 40.0

# Gear change points [km/h]. A gearbox, roughly.
_GEAR_SHIFTS = (70.0, 110.0, 150.0, 190.0, 230.0, 270.0, 310.0)


def _header(packet_id: int, frame: int, session_time: float, player: int) -> bytes:
    return _HEADER.pack(
        PACKET_FORMAT,  # m_packetFormat
        25,  # m_gameYear
        1,  # m_gameMajorVersion
        0,  # m_gameMinorVersion
        1,  # m_packetVersion
        packet_id,  # m_packetId
        0x1234_5678_9ABC_DEF0,  # m_sessionUID
        session_time,  # m_sessionTime
        frame,  # m_frameIdentifier
        frame,  # m_overallFrameIdentifier
        player,  # m_playerCarIndex
        255,  # m_secondaryPlayerCarIndex
    )


def _car_packet(
    packet_id: int,
    frame: int,
    session_time: float,
    player: int,
    payload: bytes,
    stride: int,
    tail: int,
) -> bytes:
    """A packet whose per-car array holds payload at the player's index."""
    array = bytearray(MAX_CARS * stride + tail)
    array[player * stride : player * stride + len(payload)] = payload
    return _header(packet_id, frame, session_time, player) + bytes(array)


def session_packet(
    track_id: int = 11, track_length: float = 5793.0, player: int = 0
) -> bytes:
    """A PacketSessionData naming the circuit."""
    body = bytearray(_SESSION_LENGTH)
    body[: _SESSION.size] = _SESSION.pack(
        0,  # m_weather, clear
        30,  # m_trackTemperature [C]
        22,  # m_airTemperature [C]
        1,  # m_totalLaps
        int(round(track_length)),  # m_trackLength [m]
        9,  # m_sessionType, time trial
        track_id,  # m_trackId
    )
    return _header(PACKET_SESSION, 0, 0.0, player) + bytes(body)


def _gear_and_rpm(speed_kmh: float) -> tuple[int, int]:
    """A gear and an engine speed consistent with a road speed."""
    gear = 1 + int(np.searchsorted(_GEAR_SHIFTS, speed_kmh))
    low = 0.0 if gear == 1 else _GEAR_SHIFTS[gear - 2]
    high = _GEAR_SHIFTS[min(gear - 1, len(_GEAR_SHIFTS) - 1)]
    span = max(high - low, 1.0)
    fraction = float(np.clip((speed_kmh - low) / span, 0.0, 1.0))
    return gear, int(6000 + 6000 * fraction)


class _LapModel:
    """One lap's exact kinematics and filler channels on a fine grid."""

    def __init__(self, speed: SpeedProfile, lap_length: float) -> None:
        # Fine reference grid: the exact kinematics live here, and the
        # emitted frames are sampled from it in time, the way the game
        # samples the car rather than the track.
        self.dist = np.linspace(0.0, lap_length, 4001)
        self.speed = np.asarray(speed(self.dist), dtype=float)
        if np.any(self.speed <= 0.0):
            raise ValueError("speed must be positive everywhere along the lap")
        self.time = cumulative_trapezoid(3.6 / self.speed, self.dist, initial=0.0)
        self.lap_time = float(self.time[-1])

        # Pedals from the speed gradient: on the brakes in proportion to
        # the deceleration, off the throttle while braking, and back to
        # full throttle over a fixed ramp once the brakes come off.
        gradient = np.gradient(self.speed, self.dist)
        scale = float(np.max(np.abs(gradient))) or 1.0
        self.brake = np.clip(-gradient / scale, 0.0, 1.0)
        braking = self.brake >= 0.02
        last_braking = np.maximum.accumulate(
            np.where(braking, np.arange(self.dist.size), -1)
        )
        since_release = np.where(
            last_braking >= 0, self.dist - self.dist[last_braking], np.inf
        )
        self.throttle = np.clip(since_release / _THROTTLE_RAMP, 0.0, 1.0)
        # Steering: more lock where the car is slower, sign alternating
        # around the lap so the channel is not trivially constant.
        lock = 1.0 - self.speed / float(np.max(self.speed))
        self.steer = lock * np.sin(2.0 * np.pi * self.dist / lap_length)
        # Track shape: a circle of the right circumference. A
        # placeholder, but a closed one, so track maps and mark
        # projection have something geometrically consistent to use.
        self.radius = lap_length / (2.0 * np.pi)
        self.angle = 2.0 * np.pi * self.dist / lap_length

    def at(self, channel: np.ndarray, dist: float) -> float:
        return float(np.interp(dist, self.dist, channel))


def synthetic_session(
    profiles: Sequence[SpeedProfile],
    lap_length: float = 5793.0,
    rate_hz: float = 30.0,
    track_id: int = 11,
    player_index: int = 0,
    include_motion: bool = True,
    invalid_laps: tuple[int, ...] = (),
    pit_laps: tuple[int, ...] = (),
) -> tuple[list[bytes], list[float]]:
    """Build a UDP stream of consecutive laps, one per speed profile.

    profiles:
        One callable per lap, mapping distance along the lap [m] to
        speed [km/h]. Each must stay positive; time is integrated from
        it. A lap is only finished once the next lap's first frame
        arrives, so N profiles yield N - 1 complete laps, exactly as a
        real session does.
    rate_hz:
        Frame rate of the stream. The game offers 10 to 60 Hz; the
        default matches its own default of 30.
    invalid_laps, pit_laps:
        1-based lap numbers to mark as invalidated, or as running
        through the pits, for exercising the loader's filters.

    Returns the datagrams and the exact lap time [s] implied by each
    profile, which is what the decoded Laps should reproduce.
    """
    models = [_LapModel(profile, lap_length) for profile in profiles]
    datagrams = [
        session_packet(track_id=track_id, track_length=lap_length, player=player_index)
    ]
    frame = 1
    session_time = 0.0

    for lap_number, model in enumerate(models, start=1):
        previous = models[lap_number - 2].lap_time if lap_number > 1 else 0.0
        last_lap_ms = int(round(previous * 1000.0))
        invalid = 1 if lap_number in invalid_laps else 0
        pit = 1 if lap_number in pit_laps else 0
        driver_status = 2 if pit else 1
        lap_start_total = (lap_number - 1) * lap_length

        sample_times = np.arange(0.0, model.lap_time, 1.0 / rate_hz)
        sample_dist = np.interp(sample_times, model.time, model.dist)

        for lap_t, dist in zip(sample_times, sample_dist, strict=True):
            speed_kmh = model.at(model.speed, dist)
            gear, rpm = _gear_and_rpm(speed_kmh)

            if include_motion:
                angle = model.at(model.angle, dist)
                datagrams.append(
                    _car_packet(
                        PACKET_MOTION,
                        frame,
                        session_time,
                        player_index,
                        _MOTION.pack(
                            model.radius * np.cos(angle),  # m_worldPositionX
                            0.0,  # m_worldPositionY, vertical
                            model.radius * np.sin(angle),  # m_worldPositionZ
                        ),
                        MOTION_STRIDE,
                        _MOTION_TAIL,
                    )
                )

            datagrams.append(
                _car_packet(
                    PACKET_LAP_DATA,
                    frame,
                    session_time,
                    player_index,
                    _LAP_DATA.pack(
                        last_lap_ms,  # m_lastLapTimeInMS
                        int(round(lap_t * 1000.0)),  # m_currentLapTimeInMS
                        0,  # m_sector1TimeMSPart
                        0,  # m_sector1TimeMinutesPart
                        0,  # m_sector2TimeMSPart
                        0,  # m_sector2TimeMinutesPart
                        0,  # m_deltaToCarInFrontMSPart
                        0,  # m_deltaToCarInFrontMinutesPart
                        0,  # m_deltaToRaceLeaderMSPart
                        0,  # m_deltaToRaceLeaderMinutesPart
                        float(dist),  # m_lapDistance
                        float(lap_start_total + dist),  # m_totalDistance
                        0.0,  # m_safetyCarDelta
                        1,  # m_carPosition
                        lap_number,  # m_currentLapNum
                        pit,  # m_pitStatus
                        0,  # m_numPitStops
                        0,  # m_sector
                        invalid,  # m_currentLapInvalid
                        0,  # m_penalties
                        0,  # m_totalWarnings
                        0,  # m_cornerCuttingWarnings
                        0,  # m_numUnservedDriveThroughPens
                        0,  # m_numUnservedStopGoPens
                        1,  # m_gridPosition
                        driver_status,  # m_driverStatus
                    ),
                    LAP_DATA_STRIDE,
                    _LAP_DATA_TAIL,
                )
            )

            datagrams.append(
                _car_packet(
                    PACKET_CAR_TELEMETRY,
                    frame,
                    session_time,
                    player_index,
                    _CAR_TELEMETRY.pack(
                        int(round(speed_kmh)),  # m_speed [km/h]
                        model.at(model.throttle, dist),  # m_throttle
                        model.at(model.steer, dist),  # m_steer
                        model.at(model.brake, dist),  # m_brake
                        0,  # m_clutch
                        gear,  # m_gear
                        rpm,  # m_engineRPM
                    ),
                    CAR_TELEMETRY_STRIDE,
                    _CAR_TELEMETRY_TAIL,
                )
            )

            frame += 1
            session_time += 1.0 / rate_hz

    return datagrams, [model.lap_time for model in models]


def synthetic_stream(
    speed: SpeedProfile,
    lap_length: float = 5793.0,
    laps: int = 3,
    **kwargs,
) -> tuple[list[bytes], float]:
    """Build a UDP stream of repeated identical laps, and their lap time.

    The single-profile case of synthetic_session: ``laps`` laps of
    packets, so ``laps - 1`` complete laps, all with the same exact lap
    time, which is returned alongside the datagrams.
    """
    datagrams, lap_times = synthetic_session(
        [speed] * laps, lap_length=lap_length, **kwargs
    )
    return datagrams, lap_times[0]
