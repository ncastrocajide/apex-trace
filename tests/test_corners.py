"""Segmentation against synthetic speed profiles with known corners."""

import numpy as np
import pytest

from apextrace.corners import Corner, paired_apexes, segment_lap

TWO_DIPS = ((1000.0, 100.0, 120.0), (2600.0, 150.0, 150.0))


def test_two_dips_give_two_corners(make_lap, dipped_speed):
    dist = np.arange(0.0, 4001.0, 5.0)
    lap = make_lap(dist, dipped_speed(dist, dips=TWO_DIPS))
    corners = segment_lap(lap)
    assert [c.number for c in corners] == [1, 2]
    assert corners[0].apex == pytest.approx(1000.0, abs=5.0)
    assert corners[1].apex == pytest.approx(2600.0, abs=5.0)
    assert corners[0].apex_speed == pytest.approx(100.0, abs=1.0)


def test_segments_tile_the_lap_with_no_gaps(make_lap, dipped_speed):
    dist = np.arange(0.0, 4001.0, 5.0)
    corners = segment_lap(make_lap(dist, dipped_speed(dist, dips=TWO_DIPS)))
    assert corners[0].start == 0.0
    assert corners[-1].end == dist[-1]
    assert corners[0].end == corners[1].start


def test_constant_speed_has_no_corners_and_raises(make_lap):
    dist = np.arange(0.0, 2001.0, 5.0)
    with pytest.raises(ValueError, match="apex"):
        segment_lap(make_lap(dist, np.full(dist.size, 250.0)))


def test_phase_events_are_threshold_crossings(make_lap, dipped_speed):
    dist = np.arange(0.0, 2001.0, 5.0)
    speed = dipped_speed(dist, dips=((1000.0, 80.0, 120.0),))
    throttle = np.where((dist >= 800.0) & (dist < 1200.0), 0.0, 100.0)
    brake = np.where((dist >= 850.0) & (dist < 1000.0), 100.0, 0.0)
    (corner,) = segment_lap(make_lap(dist, speed, throttle=throttle, brake=brake))
    assert corner.lift_point == 800.0
    assert corner.brake_point == 850.0
    assert corner.throttle_point == 1200.0


def test_flat_out_corner_has_no_events(make_lap, dipped_speed):
    # A speed dip with the throttle pinned and no braking: the corner
    # exists, its events do not. None is information, not a failure.
    dist = np.arange(0.0, 2001.0, 5.0)
    (corner,) = segment_lap(make_lap(dist, dipped_speed(dist)))
    assert corner.lift_point is None
    assert corner.brake_point is None
    assert corner.throttle_point is None


def _corners(*apexes: float) -> list[Corner]:
    bounds = [
        0.0,
        *[(a + b) / 2 for a, b in zip(apexes, apexes[1:], strict=False)],
        4000.0,
    ]
    return [
        Corner(
            number=i + 1,
            start=bounds[i],
            end=bounds[i + 1],
            apex=apex,
            apex_speed=100.0,
        )
        for i, apex in enumerate(apexes)
    ]


def test_paired_apexes_pairs_matching_detections():
    pos_a, pos_b = paired_apexes(_corners(1000.0, 3000.0), _corners(1010.0, 2990.0))
    np.testing.assert_allclose(pos_a, [1000.0, 3000.0])
    np.testing.assert_allclose(pos_b, [1010.0, 2990.0])


def test_paired_apexes_rejects_different_counts():
    assert paired_apexes(_corners(1000.0, 3000.0), _corners(1000.0)) is None


def test_paired_apexes_rejects_offset_pairs():
    # The Silverstone lesson, synthesised: equal counts but one pair is
    # hundreds of metres apart, so these are not the same corners.
    assert paired_apexes(_corners(1000.0, 3000.0), _corners(1000.0, 3400.0)) is None
