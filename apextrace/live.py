"""Live delta: the lap in progress against a finished reference lap.

The post-session engine compares two complete laps. While driving there
is only one complete lap, the reference, and a stream of instants from
the other. The question at each instant is the same one the delta
engine answers, asked at a single point: the reference was at this
distance after some time; the car is here now after its own; the
difference is the delta. So the whole method is one interpolation of
the reference's Time channel, and it is source-agnostic for the same
reason the rest of the engine is: the reference is a canonical Lap, and
the instants are plain distances and times, whichever loader made them.

Sign convention, as everywhere else: delta > 0 means the lap in
progress is behind the reference.

One honest difference from the post-session delta. Comparing two
finished laps, the engine reconciles their distance axes by pinning
both line crossings, because each lap measured the track with its own
odometer. A lap in progress has not reached the far line yet, so there
is nothing to pin: the live delta carries whatever the two odometers
disagree by at the current point. That is metres, so hundredths of a
second, and it is corrected the moment the lap finishes and goes
through the full engine. The live number is for the driver; the
post-session number is the one to quote.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from apextrace.lap import Lap


@dataclass(frozen=True)
class LiveSample:
    """One instant of a lap in progress, in canonical units.

    What a live source hands over per frame: enough to place the car on
    the reference lap (distance, time) and to show the driver's inputs.
    """

    lap_number: int
    distance: float  # m since the timing line
    time: float  # s since the timing line
    speed: float  # km/h
    throttle: float  # %
    brake: float  # %
    gear: int
    steering: float | None = None  # % of full lock
    x: float | None = None  # m
    y: float | None = None  # m
    valid: bool = True  # still a comparable lap (not invalidated, not pitting)


@dataclass(frozen=True)
class LiveReading:
    """The live delta at one instant."""

    distance: float  # m, clipped onto the reference lap
    time: float  # s since the line, lap in progress
    reference_time: float  # s, reference lap at the same distance
    delta: float  # s, positive means behind the reference
    predicted_lap_time: float  # s, reference lap time plus current delta
    trend: float  # s gained (-) or lost (+) over the last trend window


class LiveDelta:
    """Running delta of a lap in progress against a reference Lap."""

    def __init__(self, reference: Lap, trend_window: float = 150.0) -> None:
        """
        reference:
            The finished lap to compare against, from any source.
        trend_window:
            Metres of track over which the trend is measured. The delta
            says where the lap stands; the trend says whether it is
            being won or lost right now, which is what a driver can act
            on. Long enough to span a braking zone, short enough to
            belong to one corner.
        """
        self.reference = reference
        self.trend_window = trend_window
        self._dist = reference.data["Distance"].to_numpy(dtype=float)
        self._time = reference.data["Time"].to_numpy(dtype=float)
        self._length = float(self._dist[-1])
        self._history: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        """Forget the lap in progress."""
        self._history.clear()

    def update(self, distance: float, time: float) -> LiveReading:
        """Delta at one instant of the lap in progress.

        Instants are expected in driving order. One that lies behind
        the last one seen means the car went back: a new lap started,
        or the driver rewound. Either way the readings beyond that
        point no longer describe this attempt, so they are dropped
        from the trend rather than compared across the jump.
        """
        while self._history and self._history[-1][0] >= distance:
            self._history.pop()

        # Past either end of the reference there is no reference to
        # read; hold its end values rather than extrapolate.
        position = min(max(distance, 0.0), self._length)
        reference_time = float(np.interp(position, self._dist, self._time))
        delta = time - reference_time

        self._history.append((distance, delta))
        while (
            len(self._history) > 2
            and distance - self._history[1][0] >= self.trend_window
        ):
            self._history.popleft()

        return LiveReading(
            distance=position,
            time=time,
            reference_time=reference_time,
            delta=delta,
            predicted_lap_time=self.reference.lap_time + delta,
            trend=delta - self._history[0][1],
        )
