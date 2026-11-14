"""Lap tables: what is saved is what is loaded."""

import numpy as np
import pandas as pd
import pytest

from apextrace.delta import delta_time
from apextrace.storage import is_lap_table, load_laps, save_laps

DIST = np.arange(0.0, 2001.0, 5.0)


@pytest.fixture
def laps(make_lap, dipped_speed):
    """Two laps with different channel sets, as two sources would give."""
    rich = make_lap(DIST, dipped_speed(DIST), label="sim lap", source="f1_25")
    rich.track, rich.car = "Monza", None
    rich.data["Steering"] = np.sin(DIST / 300.0) * 40.0
    rich.data["X"] = np.cos(DIST / 318.0) * 318.0
    rich.data["Y"] = np.sin(DIST / 318.0) * 318.0
    plain = make_lap(
        DIST, dipped_speed(DIST, base=310.0), label="NOR 2024 Monza Q", source="fastf1"
    )
    plain.track, plain.car = "Monza", "McLaren"
    return [rich, plain]


@pytest.mark.parametrize("name", ["laps.csv", "laps.csv.gz"])
def test_round_trip_is_exact(tmp_path, laps, name):
    path = tmp_path / name
    save_laps(laps, path)
    loaded = load_laps(path)

    assert len(loaded) == 2
    for original, back in zip(laps, loaded, strict=True):
        assert (back.label, back.source, back.track, back.car) == (
            original.label,
            original.source,
            original.track,
            original.car,
        )
        # Bit for bit: full-precision floats, same columns, same dtypes.
        pd.testing.assert_frame_equal(back.data, original.data, check_exact=True)


def test_each_lap_keeps_only_its_own_channels(tmp_path, laps):
    """The table is the union of channels; a lap is not."""
    path = tmp_path / "laps.csv"
    save_laps(laps, path)
    rich, plain = load_laps(path)
    assert {"Steering", "X", "Y"} <= set(rich.channels)
    assert not {"Steering", "X", "Y"} & set(plain.channels)
    assert plain.data["Gear"].dtype == rich.data["Gear"].dtype == np.int64


def test_the_file_is_a_plain_table_anyone_can_open(tmp_path, laps):
    path = tmp_path / "laps.csv"
    save_laps(laps, path)

    assert is_lap_table(path.read_bytes()[:64])
    table = pd.read_csv(path, comment="#")
    assert list(table.columns[:3]) == ["Lap", "Distance", "Time"]
    assert len(table) == 2 * DIST.size
    assert set(table["Lap"]) == {0, 1}


def test_loaded_laps_go_straight_into_the_engine(tmp_path, laps):
    path = tmp_path / "laps.csv"
    save_laps(laps, path)
    before = delta_time(*laps)
    after = delta_time(*load_laps(path))
    assert np.array_equal(before.to_numpy(), after.to_numpy())


def test_files_that_are_not_lap_tables_are_refused(tmp_path, laps):
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n")
    assert not is_lap_table(other.read_bytes())
    with pytest.raises(ValueError, match="not an apextrace lap table"):
        load_laps(other)

    path = tmp_path / "laps.csv"
    save_laps(laps, path)
    lines = path.read_text().splitlines()
    damaged = tmp_path / "damaged.csv"
    damaged.write_text("\n".join([lines[0], "# not json", *lines[2:]]))
    with pytest.raises(ValueError, match="damaged lap list"):
        load_laps(damaged)

    with pytest.raises(ValueError, match="no laps"):
        save_laps([], tmp_path / "empty.csv")
