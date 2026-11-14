"""A synthetic MoTeC outing and a writer for i2-style CSV exports.

Shared by the loader's tests and the app's: an outing is synthesised
the way a logger would record it. The car joins the track part-way
round, completes two laps and starts a third, sampled at a fixed rate
in time, with beacon markers at the exact line-crossing instants. The
kinematics are exact, so lap times and lengths have closed-form answers.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

LAP_LENGTH = 3000.0
JOIN_AT = 2000.0  # where on the lap the outing starts [m]
RATE_HZ = 20.0
STEERING_LOCK = 180.0  # deg, one side


def profile(mid_floor):
    dips = ((600.0, 90.0, 90.0), (1500.0, mid_floor, 110.0), (2400.0, 70.0, 80.0))

    def _speed(position):
        speed = np.full(position.shape, 280.0)
        for centre, floor, width in dips:
            speed = speed - (280.0 - floor) * np.exp(
                -(((position - centre) / width) ** 2)
            )
        return speed

    return _speed


class Outing:
    """A logged outing: samples at a fixed rate, and where the line was."""

    def __init__(self, mid_floors=(130.0, 130.0, 105.0, 130.0), tail=800.0):
        # Lap index 0 is the out lap (joined at JOIN_AT); 1 and 2 are
        # complete; 3 is the one the outing ends part-way through.
        total = (LAP_LENGTH - JOIN_AT) + 2 * LAP_LENGTH + tail
        run = np.linspace(0.0, total, 20001)  # odometer since the log started
        on_track = run + JOIN_AT
        lap_index = np.minimum((on_track // LAP_LENGTH).astype(int), 3)
        position = on_track - lap_index * LAP_LENGTH
        speed = np.empty_like(run)
        for index, floor in enumerate(mid_floors):
            mask = lap_index == index
            speed[mask] = profile(floor)(position[mask])
        time = cumulative_trapezoid(3.6 / speed, run, initial=0.0)

        crossings = (LAP_LENGTH - JOIN_AT) + LAP_LENGTH * np.arange(3)
        self.beacons = np.interp(crossings, run, time)
        self.lap_times = np.diff(self.beacons)

        self.time = np.arange(0.0, time[-1], 1.0 / RATE_HZ)
        self.run = np.interp(self.time, time, run)
        self.position = (self.run + JOIN_AT) % LAP_LENGTH
        self.speed = np.interp(self.time, time, speed)

        gradient = np.gradient(self.speed, self.run)
        scale = np.max(np.abs(gradient))
        self.brake = np.clip(-gradient / scale, 0.0, 1.0) * 100.0
        self.throttle = np.where(self.brake < 2.0, 100.0, 0.0)
        self.gear = np.clip(self.speed // 45.0, 1, 7)
        self.rpm = 6000.0 + 20.0 * self.speed
        angle = 2.0 * np.pi * self.position / LAP_LENGTH
        self.steer_deg = 90.0 * (1.0 - self.speed / 280.0) * np.sign(np.sin(angle))
        radius = LAP_LENGTH / (2.0 * np.pi)
        self.x = radius * np.cos(angle)
        self.height = 12.0 * np.sin(2.0 * angle)  # world y: up, in Assetto Corsa
        self.z = radius * np.sin(angle)

    def columns(self):
        """(name, unit, values) for every channel, as ACTI would log them."""
        return [
            ("Time", "s", self.time),
            ("Distance", "m", self.run),
            ("Ground Speed", "km/h", self.speed),
            ("Throttle Pos", "%", self.throttle),
            ("Brake Pos", "%", self.brake),
            ("Gear", "", self.gear),
            ("Engine RPM", "rpm", self.rpm),
            ("Steering Angle", "deg", self.steer_deg),
            ("Car Coord X", "m", self.x),
            ("Car Coord Y", "m", self.height),
            ("Car Coord Z", "m", self.z),
        ]


def write_motec_csv(path, columns, beacons=None, encoding="utf-8", fmt="{:.4f}"):
    """Write an i2-style CSV: quoted header lines, names, units, samples."""
    lines = [
        '"Format","MoTeC CSV File",,,"Workbook",""',
        '"Venue","monza",,,"Worksheet",""',
        '"Vehicle","ferrari_488_gt3",,,"",""',
        '"Driver","",,,"Engine ID",""',
        '"Device","ADL"',
        '"Comment","",,,"Session",""',
        '"Log Date","11/11/2026",,,"Origin Time","0.000","s"',
        '"Log Time","10:00:00",,,"Start Time","0.000","s"',
        '"Sample Rate","20.000","Hz",,"End Time","120.000","s"',
        '"Duration","120.000","s",,"Start Distance","0","m"',
        '"Range","Outing",,,"End Distance","7800","m"',
    ]
    if beacons is not None:
        marks = " ".join(f"{beacon:.3f}" for beacon in beacons)
        lines.append(f'"Beacon Markers","{marks}"')
    lines += ["", ",".join(f'"{name}"' for name, _, _ in columns)]
    lines += [",".join(f'"{unit}"' for _, unit, _ in columns), ""]
    table = np.column_stack([values for _, _, values in columns])
    lines += [",".join(f'"{fmt.format(v)}"' for v in row) for row in table]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode(encoding))
    return path
