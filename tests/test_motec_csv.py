"""MoTeC CSV loader, on exports written to the format by hand.

The outing and the writer live in support_motec.py; see there for how
the data is built and why its answers are known.
"""

import numpy as np
import pytest
from support_motec import (
    LAP_LENGTH,
    RATE_HZ,
    STEERING_LOCK,
    Outing,
    write_motec_csv,
)

from apextrace.corners import segment_lap
from apextrace.delta import delta_time
from apextrace.lap import CORE_CHANNELS
from apextrace.loaders.motec_csv import MotecFormatError, is_motec_csv, load_motec_csv


@pytest.fixture(scope="module")
def outing():
    return Outing()


@pytest.fixture
def export(tmp_path, outing):
    return write_motec_csv(tmp_path / "outing.csv", outing.columns(), outing.beacons)


def test_complete_laps_are_cut_at_the_beacons(export, outing):
    """Out lap and unfinished lap dropped; the two between beacons kept."""
    laps, notes = load_motec_csv(export, steering_lock=STEERING_LOCK)
    assert len(laps) == 2
    assert notes == []
    # Beacons are written to the millisecond, as i2 writes them.
    assert [lap.lap_time for lap in laps] == pytest.approx(outing.lap_times, abs=1e-3)
    assert laps[1].lap_time > laps[0].lap_time + 0.3  # the slow corner


def test_both_ends_of_a_lap_sit_on_the_line(export):
    """Beacons fall between samples; the ends are interpolated onto them."""
    laps, _ = load_motec_csv(export, steering_lock=STEERING_LOCK)
    data = laps[0].data
    assert data["Distance"].iloc[0] == 0.0
    assert data["Time"].iloc[0] == 0.0
    assert data["Distance"].iloc[-1] == pytest.approx(LAP_LENGTH, abs=0.1)
    assert np.allclose(np.diff(data["Distance"].to_numpy()[:-1]), 5.0)


def test_canonical_schema_and_metadata(export):
    laps, _ = load_motec_csv(export, steering_lock=STEERING_LOCK)
    lap = laps[0]
    assert set(CORE_CHANNELS) <= set(lap.channels)
    assert {"RPM", "Steering", "X", "Y"} <= set(lap.channels)
    assert "Odometer" not in lap.channels
    assert (lap.source, lap.track, lap.car) == ("motec_csv", "monza", "ferrari_488_gt3")
    assert lap.label == "ferrari_488_gt3 monza lap 1"
    gear = lap.data["Gear"].to_numpy()
    assert np.array_equal(gear, gear.astype(int))


def test_steering_in_degrees_needs_the_lock(export):
    """No lock, no steering channel, and a note saying so. Never a guess."""
    laps, notes = load_motec_csv(export)
    assert "Steering" not in laps[0].channels
    assert len(notes) == 1 and "steering_lock" in notes[0]

    laps, _ = load_motec_csv(export, steering_lock=STEERING_LOCK)
    steering = laps[0].data["Steering"]
    # Up to 90 deg of wheel at the slowest corner, of 180 deg of lock.
    assert steering.abs().max() == pytest.approx(
        90.0 * (1 - 70.0 / 280.0) / 180.0 * 100.0, rel=0.02
    )


def test_the_ground_plane_is_found_from_the_data(export):
    """Height is the axis with the least extent, whatever it is called."""
    laps, _ = load_motec_csv(export)
    lap = laps[0]
    radius = np.hypot(lap.data["X"].to_numpy(), lap.data["Y"].to_numpy())
    assert radius == pytest.approx(LAP_LENGTH / (2.0 * np.pi), rel=1e-3)


def test_units_are_converted_not_assumed(tmp_path, outing, export):
    columns = outing.columns()
    converted = []
    for name, unit, values in columns:
        if name == "Ground Speed":
            converted.append((name, "m/s", values / 3.6))
        elif name in ("Throttle Pos", "Brake Pos"):
            converted.append((name, "", values / 100.0))  # logged as a fraction
        else:
            converted.append((name, unit, values))
    other = write_motec_csv(tmp_path / "si.csv", converted, outing.beacons)

    reference = load_motec_csv(export)[0][0].data
    data = load_motec_csv(other)[0][0].data
    for channel in ("Speed", "Throttle", "Brake"):
        assert np.allclose(data[channel], reference[channel], atol=0.02)

    furlongs = [
        (n, "furlong/fortnight" if n == "Ground Speed" else u, v) for n, u, v in columns
    ]
    odd = write_motec_csv(tmp_path / "odd.csv", furlongs, outing.beacons)
    with pytest.raises(MotecFormatError, match="furlong"):
        load_motec_csv(odd)


def test_distance_is_integrated_when_not_logged(tmp_path, outing):
    columns = [c for c in outing.columns() if c[0] != "Distance"]
    path = write_motec_csv(tmp_path / "no_distance.csv", columns, outing.beacons)
    laps, notes = load_motec_csv(path)
    assert any("integrated from speed" in note for note in notes)
    assert laps[0].data["Distance"].iloc[-1] == pytest.approx(LAP_LENGTH, rel=1e-3)
    assert laps[0].lap_time == pytest.approx(outing.lap_times[0], abs=1e-3)


def test_a_distance_channel_that_resets_each_lap_is_unwrapped(tmp_path, outing):
    columns = [
        ("Lap Distance", "m", outing.position) if name == "Distance" else (name, u, v)
        for name, u, v in outing.columns()
    ]
    path = write_motec_csv(tmp_path / "lap_distance.csv", columns, outing.beacons)
    laps, _ = load_motec_csv(path)
    assert len(laps) == 2
    assert laps[0].data["Distance"].iloc[-1] == pytest.approx(LAP_LENGTH, abs=10.0)
    assert np.all(np.diff(laps[0].data["Time"].to_numpy()) > 0.0)


def test_unknown_channel_names_are_reported_and_can_be_mapped(tmp_path, outing):
    columns = [
        ("BRK", u, v) if name == "Brake Pos" else (name, u, v)
        for name, u, v in outing.columns()
    ]
    path = write_motec_csv(tmp_path / "custom.csv", columns, outing.beacons)

    with pytest.raises(MotecFormatError) as error:
        load_motec_csv(path)
    assert "Brake" in str(error.value) and "BRK" in str(error.value)

    laps, _ = load_motec_csv(path, channel_map={"Brake": "BRK"})
    assert laps[0].data["Brake"].max() > 50.0

    with pytest.raises(MotecFormatError, match="not in the file"):
        load_motec_csv(path, channel_map={"Brake": "Brake Pressure"})


def test_single_lap_exports(tmp_path, outing):
    """A one-lap range has no interior beacons: the file is the lap."""
    inside = (outing.time >= outing.beacons[0]) & (outing.time <= outing.beacons[1])
    columns = [(name, unit, values[inside]) for name, unit, values in outing.columns()]

    bare = write_motec_csv(tmp_path / "one_lap.csv", columns)
    laps, notes = load_motec_csv(bare)
    assert len(laps) == 1 and any("whole file" in note for note in notes)
    # Without beacons the ends are the first and last samples, so the
    # lap time is short of the truth by up to two sample periods.
    assert laps[0].lap_time == pytest.approx(outing.lap_times[0], abs=2.0 / RATE_HZ)

    one_beacon = write_motec_csv(
        tmp_path / "one_beacon.csv", outing.columns(), outing.beacons[:1]
    )
    with pytest.raises(MotecFormatError, match="single beacon"):
        load_motec_csv(one_beacon)
    laps, _ = load_motec_csv(one_beacon, whole_file=True)
    assert len(laps) == 1


def test_tolerates_what_real_exports_do(tmp_path, outing):
    """Windows code page, a degree sign, thousands separators."""
    columns = [
        (name, "°" if name == "Steering Angle" else unit, values)
        for name, unit, values in outing.columns()
    ]
    path = write_motec_csv(
        tmp_path / "windows.csv",
        columns,
        outing.beacons,
        encoding="cp1252",
        fmt="{:,.4f}",
    )
    laps, notes = load_motec_csv(path, steering_lock=STEERING_LOCK)
    assert len(laps) == 2 and notes == []
    assert laps[0].data["RPM"].max() > 10_000.0
    assert "Steering" in laps[0].channels


def test_files_that_are_not_exports_are_refused(tmp_path, export):
    assert is_motec_csv(export.read_bytes()[:4096])

    other = tmp_path / "other.csv"
    other.write_text("a,b,c\n1,2,3\n")
    assert not is_motec_csv(other.read_bytes())
    with pytest.raises(MotecFormatError, match="not a MoTeC CSV"):
        load_motec_csv(other)

    headless = tmp_path / "headless.csv"
    headless.write_text('"Format","MoTeC CSV File"\n"Venue","monza"\n')
    with pytest.raises(MotecFormatError, match="no channel table"):
        load_motec_csv(headless)


def test_the_engine_reads_motec_laps_unmodified(export, outing):
    """Third source, same engine: the marginal cost is the loader."""
    laps, _ = load_motec_csv(export)
    lap_a, lap_b = laps

    assert len(segment_lap(lap_a)) == 3
    delta = delta_time(lap_a, lap_b)
    expected = outing.lap_times[1] - outing.lap_times[0]
    assert float(delta.iloc[-1]) == pytest.approx(expected, abs=5e-3)

    def at(s):
        return float(np.interp(s, delta.index, delta.to_numpy()))

    assert at(1200.0) == pytest.approx(0.0, abs=1e-2)
    assert at(1800.0) == pytest.approx(expected, abs=5e-2)
