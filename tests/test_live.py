"""Live delta against a reference lap, with answers known in closed form."""

import numpy as np
import pytest

from apextrace.delta import delta_time
from apextrace.live import LiveDelta, LiveSample
from apextrace.loaders.f1_25 import LapAssembler, load_f1_25_laps
from apextrace.loaders.f1_25.synthetic import synthetic_session

DIST = np.arange(0.0, 2001.0, 5.0)


def test_identical_lap_reads_zero_everywhere(make_lap):
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    live = LiveDelta(reference)
    for dist, time in zip(DIST, reference.data["Time"], strict=True):
        assert live.update(dist, time).delta == pytest.approx(0.0, abs=1e-12)


def test_constant_speed_deficit_grows_linearly(make_lap):
    """180 against 200 km/h: the delta is the slowness gap times distance."""
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    live = LiveDelta(reference)

    for dist in (500.0, 1000.0, 2000.0):
        reading = live.update(dist, dist * 3.6 / 180.0)
        expected = dist * (3.6 / 180.0 - 3.6 / 200.0)
        assert reading.delta == pytest.approx(expected)
        assert reading.predicted_lap_time == pytest.approx(
            reference.lap_time + expected
        )


def test_sign_convention_matches_the_post_session_engine(make_lap):
    """Positive means the lap in progress is behind, as in delta_time."""
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    slower = make_lap(DIST, np.full(DIST.size, 190.0))
    live = LiveDelta(reference)

    post = delta_time(reference, slower)
    for dist, time in zip(DIST[::40], slower.data["Time"][::40], strict=True):
        assert live.update(dist, time).delta == pytest.approx(
            float(np.interp(dist, post.index, post.to_numpy())), abs=1e-9
        )


def test_trend_reports_what_is_happening_now_not_the_running_total(make_lap):
    """Lose time early, then match the reference: delta stays, trend clears."""
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    live = LiveDelta(reference, trend_window=100.0)
    ref_pace = 3.6 / 200.0

    time = 0.0
    readings = {}
    for prev, dist in zip(DIST[:-1], DIST[1:], strict=True):
        pace = 3.6 / 150.0 if dist <= 500.0 else ref_pace
        time += (dist - prev) * pace
        readings[dist] = live.update(dist, time)

    lost = 500.0 * (3.6 / 150.0 - ref_pace)
    assert readings[400.0].trend == pytest.approx(100.0 * (3.6 / 150.0 - ref_pace))
    assert readings[1500.0].delta == pytest.approx(lost)
    assert readings[1500.0].trend == pytest.approx(0.0, abs=1e-9)


def test_going_backwards_starts_the_trend_afresh(make_lap):
    """A new lap or a flashback must not be compared across the jump."""
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    live = LiveDelta(reference)

    live.update(1900.0, 40.0)  # far behind, end of a slow lap
    reading = live.update(10.0, 10.0 * 3.6 / 200.0)  # next lap, on pace
    assert reading.delta == pytest.approx(0.0, abs=1e-9)
    assert reading.trend == pytest.approx(0.0, abs=1e-9)


def test_positions_past_the_reference_hold_its_end(make_lap):
    """The live odometer can overrun the reference's by a few metres."""
    reference = make_lap(DIST, np.full(DIST.size, 200.0))
    live = LiveDelta(reference)
    reading = live.update(2012.0, reference.lap_time + 0.3)
    assert reading.distance == pytest.approx(2000.0)
    assert reading.delta == pytest.approx(0.3)


def _profile(floor):
    def _speed(dist):
        dist = np.asarray(dist, dtype=float)
        return 280.0 - (280.0 - floor) * np.exp(-(((dist - 1500.0) / 110.0) ** 2))

    return _speed


def test_live_samples_from_the_f1_25_stream_reproduce_the_final_delta():
    """End to end: UDP packets in, live delta out, right at the line.

    Lap 1 is the reference; lap 2 takes the corner slower. Fed frame by
    frame, the live delta just before the line must equal the lap time
    difference the post-session engine reports once the lap is done.
    """
    datagrams, lap_times = synthetic_session(
        [_profile(130.0), _profile(105.0), _profile(130.0)],
        lap_length=3000.0,
        rate_hz=20.0,
    )
    reference = load_f1_25_laps(datagrams)[0][0]

    samples: list[LiveSample] = []
    assembler = LapAssembler(on_sample=samples.append)
    for datagram in datagrams:
        assembler.feed(datagram)

    live = LiveDelta(reference)
    lap_two = [s for s in samples if s.lap_number == 2]
    readings = [live.update(s.distance, s.time) for s in lap_two]

    assert all(s.valid for s in lap_two)
    assert lap_two[0].steering is not None and lap_two[0].x is not None
    # Before the corner the two laps are the same lap.
    before = [r for r in readings if r.distance < 1100.0]
    assert max(abs(r.delta) for r in before) < 2e-3
    # At the end, the whole difference has been booked.
    assert readings[-1].delta == pytest.approx(lap_times[1] - lap_times[0], abs=5e-3)
    assert readings[-1].predicted_lap_time == pytest.approx(lap_times[1], abs=5e-3)
