"""Behaviour heuristics on inputs drawn by hand.

One corner, apex at 1000 m, on a 5 m grid. Every pedal and steering
trace below is piecewise linear with its corners at stated distances,
so each metric has an answer that can be read off the construction.
"""

import numpy as np
import pytest

from apextrace.behaviour import (
    behaviour_report,
    count_hesitations,
    count_reversals,
    lap_behaviour,
)

DIST = np.arange(0.0, 2001.0, 5.0)


def piecewise(points):
    """Linear interpolation through (distance, value) points, on DIST."""
    xs, ys = zip(*points, strict=True)
    return np.interp(DIST, xs, ys)


# Brake: on at 800, 80 % from 830 to 900, off by 960.
BRAKE = piecewise([(0, 0), (800, 0), (830, 80), (900, 80), (960, 0), (2000, 0)])
# Throttle: lifted at 790, picked up at 1010, full by 1090.
THROTTLE = piecewise(
    [(0, 100), (790, 100), (795, 0), (1010, 0), (1090, 100), (2000, 100)]
)
# Steering: turn-in from 920, 60 % of lock at the apex, straight by 1100.
STEERING = piecewise([(0, 0), (920, 0), (1000, 60), (1100, 0), (2000, 0)])


@pytest.fixture
def corner_lap(make_lap, dipped_speed):
    """Factory: the one-corner lap with the given inputs."""

    def _lap(throttle=THROTTLE, brake=BRAKE, steering=STEERING):
        lap = make_lap(DIST, dipped_speed(DIST), throttle=throttle, brake=brake)
        if steering is not None:
            lap.data["Steering"] = steering
        return lap

    return _lap


def test_brake_shape(corner_lap):
    row = behaviour_report(corner_lap()).iloc[0]
    # Braking (> 10 %) starts at the 805 m sample: the ramp reads 13.3 %
    # there. Within 5 % of the 80 % peak from 830 to 900. Back under
    # 10 % at the 955 m sample.
    assert row["brake_peak"] == pytest.approx(80.0)
    assert row["brake_build_m"] == pytest.approx(830.0 - 805.0)
    assert row["brake_release_m"] == pytest.approx(955.0 - 900.0)


def test_trail_braking_is_braking_while_turning(corner_lap):
    row = behaviour_report(corner_lap()).iloc[0]
    # Turning means steering above 30 % of the corner's 60 % peak, 18 %,
    # which the turn-in ramp passes at 944 m: grid samples 945 and 950
    # are still inside the braking zone, 955 is not.
    assert row["trail_brake_m"] == pytest.approx(10.0)


def test_releasing_the_brake_before_turn_in_is_no_trail(corner_lap):
    early = piecewise([(0, 0), (800, 0), (830, 80), (880, 80), (910, 0), (2000, 0)])
    row = behaviour_report(corner_lap(brake=early)).iloc[0]
    assert row["trail_brake_m"] == 0.0


def test_throttle_ramp_runs_from_pickup_to_full(corner_lap):
    row = behaviour_report(corner_lap()).iloc[0]
    # Last sample at minimum throttle is 1010; 95 % is passed at 1086,
    # so the first full-throttle sample is 1090.
    assert row["throttle_ramp_m"] == pytest.approx(1090.0 - 1010.0)
    assert row["throttle_hesitations"] == 0.0


def test_a_lift_on_exit_counts_as_one_hesitation(corner_lap):
    hesitant = piecewise(
        [
            (0, 100),
            (790, 100),
            (795, 0),
            (1010, 0),
            (1050, 60),
            (1065, 30),
            (1120, 100),
            (2000, 100),
        ]
    )
    row = behaviour_report(corner_lap(throttle=hesitant)).iloc[0]
    assert row["throttle_hesitations"] == 1.0
    assert row["throttle_ramp_m"] > 80.0  # and it took longer to get there


def test_a_clean_arc_is_one_steering_reversal(corner_lap):
    row = behaviour_report(corner_lap()).iloc[0]
    assert row["steering_reversals"] == 1.0


def test_a_mid_corner_correction_adds_two_reversals(corner_lap):
    corrected = piecewise(
        [(0, 0), (920, 0), (980, 60), (1000, 40), (1020, 60), (1100, 0), (2000, 0)]
    )
    row = behaviour_report(corner_lap(steering=corrected)).iloc[0]
    assert row["steering_reversals"] == 3.0


def test_pedal_overlap_is_measured_in_metres(corner_lap):
    # 30 % throttle held from 900 to 920 while the brake is still on.
    overlapped = piecewise(
        [
            (0, 100),
            (790, 100),
            (795, 0),
            (895, 0),
            (900, 30),
            (920, 30),
            (925, 0),
            (1010, 0),
            (1090, 100),
            (2000, 100),
        ]
    )
    assert behaviour_report(corner_lap()).iloc[0]["pedal_overlap_m"] == 0.0
    row = behaviour_report(corner_lap(throttle=overlapped)).iloc[0]
    assert row["pedal_overlap_m"] == pytest.approx(25.0)  # samples 900..920


def test_without_a_steering_channel_only_steering_metrics_are_missing(corner_lap):
    """F1 timing data has no steering; the pedal metrics still stand."""
    row = behaviour_report(corner_lap(steering=None)).iloc[0]
    assert np.isnan(row["trail_brake_m"])
    assert np.isnan(row["steering_reversals"])
    assert row["brake_build_m"] == pytest.approx(25.0)
    assert row["throttle_ramp_m"] == pytest.approx(80.0)


def test_a_corner_taken_flat_has_no_brake_or_throttle_events(make_lap, dipped_speed):
    """A lift-free kink: nothing to measure, and nothing invented."""
    lap = make_lap(DIST, dipped_speed(DIST))  # full throttle, no brake
    row = behaviour_report(lap).iloc[0]
    assert np.isnan(row["brake_peak"])
    assert np.isnan(row["brake_build_m"])
    assert np.isnan(row["throttle_ramp_m"])
    assert row["pedal_overlap_m"] == 0.0


def test_thresholds_are_parameters_not_constants(corner_lap):
    """A stricter idea of 'turning' finds less trail braking."""
    default = behaviour_report(corner_lap()).iloc[0]["trail_brake_m"]
    strict = behaviour_report(corner_lap(), trail_steering=0.45).iloc[0]
    assert strict["trail_brake_m"] < default


def test_lap_summary_adds_lengths_and_averages_shapes(corner_lap):
    summary = lap_behaviour(behaviour_report(corner_lap()))
    assert summary["trail_brake_m"] == pytest.approx(10.0)
    assert summary["mean_brake_build_m"] == pytest.approx(25.0)
    assert summary["steering_reversals"] == 1.0


def test_reversal_counting_ignores_movement_inside_the_gap():
    wobble = np.array([0.0, 10.0, 9.0, 10.5, 9.5, 20.0, 0.0])
    assert count_reversals(wobble, gap=2.0) == 1  # only the final unwind
    assert count_reversals(wobble, gap=0.5) == 5
    assert count_reversals(np.zeros(10), gap=1.0) == 0


def test_one_long_lift_is_one_hesitation():
    throttle = np.array([0, 40, 80, 60, 50, 55, 70, 100], dtype=float)
    assert count_hesitations(throttle, drop=10.0) == 1
    two = np.array([0, 50, 30, 60, 40, 100], dtype=float)
    assert count_hesitations(two, drop=10.0) == 2
