"""The app's backend: routes, payloads and the live hub.

Same rule as the rest of the suite: synthetic inputs with known
answers, no network, no game. The live path is driven by synthetic UDP
packets, through the hub directly and once end to end over a real
WebSocket with a capture being replayed.
"""

import asyncio

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the app needs the 'app' extra")

from fastapi.testclient import TestClient  # noqa: E402

from apextrace.app.analysis import (  # noqa: E402
    NotComparable,
    compare_payload,
    lap_payload,
)
from apextrace.app.cli import build_parser, main, serve  # noqa: E402
from apextrace.app.hub import LiveHub  # noqa: E402
from apextrace.app.server import LiveConfig, create_app  # noqa: E402
from apextrace.app.store import SessionStore  # noqa: E402
from apextrace.loaders.f1_25 import load_f1_25_laps, write_capture  # noqa: E402
from apextrace.loaders.f1_25.synthetic import synthetic_session  # noqa: E402

LAP_LENGTH = 3000.0
RATE_HZ = 20.0


def profile(mid_floor=130.0):
    """Three corners; the middle one's apex speed is the knob."""
    dips = ((600.0, 90.0, 90.0), (1500.0, mid_floor, 110.0), (2400.0, 70.0, 80.0))

    def _speed(dist):
        dist = np.asarray(dist, dtype=float)
        speed = np.full(dist.shape, 280.0)
        for centre, floor, width in dips:
            speed = speed - (280.0 - floor) * np.exp(-(((dist - centre) / width) ** 2))
        return speed

    return _speed


@pytest.fixture(scope="module")
def session():
    """Quick lap, slow lap, quickest lap, and a fourth to finish the third."""
    return synthetic_session(
        [profile(130.0), profile(105.0), profile(150.0), profile(130.0)],
        lap_length=LAP_LENGTH,
        rate_hz=RATE_HZ,
    )


@pytest.fixture
def capture(tmp_path, session):
    datagrams, _ = session
    path = tmp_path / "session.f1cap"
    write_capture(path, datagrams, rate_hz=3.0 * RATE_HZ)
    return path


# The test client calls the app "testserver"; a browser would use an IP
# address or localhost, which need no allowance.
TEST_HOSTS = ("testserver",)


@pytest.fixture
def client():
    with TestClient(create_app(allowed_hosts=TEST_HOSTS)) as test_client:
        yield test_client


def upload(client, path, name="session.f1cap"):
    with path.open("rb") as src:
        return client.post("/api/upload", files={"file": (name, src)})


# --- payloads ---------------------------------------------------------


def test_compare_payload_carries_the_engine_s_numbers(session):
    datagrams, lap_times = session
    laps, _ = load_f1_25_laps(datagrams)
    payload = compare_payload(laps[0], laps[1])

    assert payload["final_delta"] == pytest.approx(
        lap_times[1] - lap_times[0], abs=5e-3
    )
    assert len(payload["distance"]) == len(payload["delta"])
    assert payload["distance"][-1] == pytest.approx(LAP_LENGTH)
    assert {"Speed", "Throttle", "Brake", "Gear", "Steering"} <= set(
        payload["channels"]
    )
    assert len(payload["segments"]) == 3
    assert payload["cross_source"] is False

    # The corner table sums to the full-lap delta, as in the engine.
    assert payload["corners_error"] is None
    total = sum(row["total_s"] for row in payload["corners"])
    assert total == pytest.approx(payload["final_delta"], abs=1e-3)
    # All of it in the corner that differs.
    assert payload["corners"][1]["total_s"] == pytest.approx(total, abs=5e-3)


def test_gear_stays_an_integer_on_the_common_grid(session):
    """Discrete channels are held, not interpolated, in overlays too."""
    laps, _ = load_f1_25_laps(session[0])
    payload = compare_payload(laps[0], laps[1])
    for side in ("a", "b"):
        gears = payload["channels"]["Gear"][side]
        assert all(float(g).is_integer() for g in gears)


def test_map_colours_by_where_the_time_goes(session):
    laps, _ = load_f1_25_laps(session[0])
    payload = compare_payload(laps[0], laps[1])
    loss = np.array(payload["map"]["loss_per_100m"])
    dist = np.array(payload["distance"])
    # The slope of the delta peaks inside the corner that was slower.
    assert 1300.0 < dist[int(np.argmax(loss))] < 1700.0
    assert abs(loss[dist < 1000.0]).max() < 0.01


def test_payloads_are_strict_json(session, make_lap):
    """NaN is not JSON; a missing event must arrive as null."""
    import json

    laps, _ = load_f1_25_laps(session[0])
    json.dumps(compare_payload(laps[0], laps[1]), allow_nan=False)
    json.dumps(lap_payload(laps[0]), allow_nan=False)

    # A lap with no corners at all still produces a payload.
    flat = make_lap(np.arange(0.0, 1001.0, 5.0), np.full(201, 200.0))
    assert json.loads(json.dumps(lap_payload(flat), allow_nan=False))["corners"] == []


def test_behaviour_table_pairs_the_two_laps_corner_by_corner(session):
    laps, _ = load_f1_25_laps(session[0])
    behaviour = compare_payload(laps[0], laps[1])["behaviour"]

    assert [c["key"] for c in behaviour["columns"]][:2] == ["corner", "lap"]
    assert [(r["corner"], r["lap"]) for r in behaviour["rows"]] == [
        ("#1", "A"),
        ("#1", "B"),
        ("#2", "A"),
        ("#2", "B"),
        ("#3", "A"),
        ("#3", "B"),
    ]
    assert lap_payload(laps[0])["behaviour"]["rows"][0]["brake_peak"] > 0.0


def test_behaviour_is_withheld_from_laps_without_driver_inputs(make_lap, dipped_speed):
    """An on/off brake and no steering: metrics would be noise, so none."""
    dist = np.arange(0.0, 2001.0, 5.0)
    brake = np.where((dist > 850.0) & (dist < 990.0), 100.0, 0.0)
    timing_like = make_lap(dist, dipped_speed(dist), brake=brake)
    assert lap_payload(timing_like)["behaviour"] is None


def test_laps_of_different_circuits_are_not_comparable(make_lap):
    short = make_lap(np.arange(0.0, 3001.0, 5.0), np.full(601, 200.0), label="short")
    long = make_lap(np.arange(0.0, 5001.0, 5.0), np.full(1001, 200.0), label="long")
    with pytest.raises(NotComparable, match="not of the same circuit"):
        compare_payload(short, long)


# --- routes -----------------------------------------------------------


def test_upload_adds_every_lap_of_a_capture(client, capture):
    assert client.get("/api/laps").json() == []

    response = upload(client, capture)
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "F1 25 capture"
    assert len(body["added"]) == 3
    assert body["added"][0]["origin"] == "upload:session.f1cap"
    assert [lap["id"] for lap in client.get("/api/laps").json()] == [
        lap["id"] for lap in body["added"]
    ]


def test_upload_reads_a_motec_export_and_passes_the_steering_lock(client, tmp_path):
    """A third format reaches the app by one line in the ingest table."""
    from support_motec import Outing, write_motec_csv

    outing = Outing()
    path = write_motec_csv(tmp_path / "outing.csv", outing.columns(), outing.beacons)

    with path.open("rb") as src:
        plain = client.post("/api/upload", files={"file": ("outing.csv", src)}).json()
    assert plain["format"] == "MoTeC CSV export"
    assert len(plain["added"]) == 2
    assert "Steering" not in plain["added"][0]["channels"]
    assert any("steering_lock" in note for note in plain["notes"])

    with path.open("rb") as src:
        with_lock = client.post(
            "/api/upload",
            files={"file": ("outing.csv", src)},
            data={"steering_lock": "180"},
        ).json()
    assert "Steering" in with_lock["added"][0]["channels"]
    assert with_lock["notes"] == []

    # Laps from an export go through the same comparison as any other,
    # and with steering present they carry a behaviour table.
    ids = [lap["id"] for lap in with_lock["added"]]
    compared = client.get("/api/compare", params={"a": ids[0], "b": ids[1]}).json()
    assert compared["behaviour"] is not None


def test_laps_leave_as_a_table_and_come_back_the_same(client, capture, tmp_path):
    """Export is the extracted-table persistence; import is one more format."""
    added = upload(client, capture).json()["added"]

    exported = client.get("/api/export")
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("text/csv")
    assert "attachment" in exported.headers["content-disposition"]
    table = tmp_path / "laps.csv"
    table.write_bytes(exported.content)

    back = upload(client, table, name="laps.csv").json()
    assert back["format"] == "apex-trace lap table"
    assert [lap["lap_time"] for lap in back["added"]] == [
        lap["lap_time"] for lap in added
    ]
    assert back["added"][0]["channels"] == added[0]["channels"]

    one = client.get("/api/export", params={"ids": added[1]["id"]})
    assert one.text.count("\n") < exported.text.count("\n")
    assert client.get("/api/export", params={"ids": "nope"}).status_code == 404


def test_export_with_nothing_loaded_is_an_error(client):
    assert client.get("/api/export").status_code == 400


def test_upload_reports_skipped_laps(client, tmp_path):
    datagrams, _ = synthetic_session(
        [profile(), profile(), profile()],
        lap_length=LAP_LENGTH,
        rate_hz=RATE_HZ,
        invalid_laps=(1,),
    )
    path = tmp_path / "one_invalid.f1cap"
    write_capture(path, datagrams)

    body = upload(client, path).json()
    assert len(body["added"]) == 1
    assert len(body["notes"]) == 1 and "invalidated" in body["notes"][0]


def test_upload_of_an_unknown_format_is_refused(client, tmp_path):
    path = tmp_path / "holiday.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 definitely not telemetry")
    response = upload(client, path, name="holiday.jpg")
    assert response.status_code == 415
    assert "F1 25 capture" in response.json()["detail"]
    assert "MoTeC CSV export" in response.json()["detail"]


def test_upload_file_name_is_only_a_label(client, capture):
    """A hostile file name must not reach the filesystem."""
    body = upload(client, capture, name="../../etc/passwd").json()
    assert body["added"][0]["origin"] == "upload:passwd"


def test_compare_route(client, capture, session):
    _, lap_times = session
    first, second, _ = (lap["id"] for lap in upload(client, capture).json()["added"])

    response = client.get("/api/compare", params={"a": first, "b": second})
    assert response.status_code == 200
    assert response.json()["final_delta"] == pytest.approx(
        lap_times[1] - lap_times[0], abs=5e-3
    )

    assert (
        client.get("/api/compare", params={"a": first, "b": first}).status_code == 400
    )
    assert (
        client.get("/api/compare", params={"a": first, "b": "nope"}).status_code == 404
    )


def test_single_lap_and_delete_routes(client, capture):
    lap_id = upload(client, capture).json()["added"][0]["id"]

    body = client.get(f"/api/laps/{lap_id}").json()
    assert len(body["corners"]) == 3
    assert body["map"] is not None

    assert client.delete(f"/api/laps/{lap_id}").status_code == 200
    assert client.get(f"/api/laps/{lap_id}").status_code == 404
    assert client.delete(f"/api/laps/{lap_id}").status_code == 404


def test_requests_from_another_site_cannot_change_state(client, capture):
    """A page elsewhere in the browser must not be able to drive the app."""
    with capture.open("rb") as src:
        refused = client.post(
            "/api/upload",
            files={"file": ("x.f1cap", src)},
            headers={"origin": "https://evil.example"},
        )
    assert refused.status_code == 403
    assert client.get("/api/laps").json() == []

    with capture.open("rb") as src:
        allowed = client.post(
            "/api/upload",
            files={"file": ("x.f1cap", src)},
            headers={"origin": "http://testserver"},
        )
    assert allowed.status_code == 200


def test_a_foreign_host_name_is_refused(capture):
    """DNS rebinding: another site's name pointed at this machine."""
    with TestClient(create_app()) as strict:  # "testserver" not allowed
        assert strict.get("/api/laps").status_code == 400
        for host in ("127.0.0.1:8000", "localhost:8000", "[::1]:8000", "192.168.1.20"):
            assert strict.get("/api/laps", headers={"host": host}).status_code == 200
        rebound = strict.get("/api/laps", headers={"host": "evil.example:8000"})
        assert rebound.status_code == 400
        assert "--allow-host" in rebound.json()["detail"]


def circle_lap(make_lap, dipped_speed, floor, label):
    """A two-corner lap with position channels, on a circle."""
    dist = np.arange(0.0, 4001.0, 5.0)
    dips = ((1000.0, floor, 120.0), (2600.0, 150.0, 150.0))
    lap = make_lap(dist, dipped_speed(dist, dips=dips), label=label)
    angle = 2.0 * np.pi * dist / 4000.0
    lap.data["X"] = 4000.0 / (2.0 * np.pi) * np.cos(angle)
    lap.data["Y"] = 4000.0 / (2.0 * np.pi) * np.sin(angle)
    return lap


def test_official_marks_name_the_corners_and_anchor_the_alignment(
    make_lap, dipped_speed
):
    """As in the M3 corner report, now through the app's payload."""
    import pandas as pd

    lap_a = circle_lap(make_lap, dipped_speed, 100.0, "a")
    lap_b = circle_lap(make_lap, dipped_speed, 90.0, "b")
    radius = 4000.0 / (2.0 * np.pi)
    at = np.array([1000.0, 2600.0])
    marks = pd.DataFrame(
        {
            "Distance": at,
            "Name": ["T1", "T4"],
            "X": radius * np.cos(2.0 * np.pi * at / 4000.0),
            "Y": radius * np.sin(2.0 * np.pi * at / 4000.0),
        }
    )

    named = compare_payload(lap_a, lap_b, marks)
    assert [s["label"] for s in named["segments"]] == ["T1", "T4"]
    assert [row["corner"] for row in named["corners"]] == ["T1", "T4"]
    assert named["landmarks"] == 2
    assert lap_payload(lap_a, marks)["corners"][0]["label"] == "T1"

    plain = compare_payload(lap_a, lap_b)
    assert [s["label"] for s in plain["segments"]] == ["#1", "#2"]
    assert plain["landmarks"] == 0
    # The line crossings pin the total either way.
    assert named["final_delta"] == pytest.approx(plain["final_delta"], abs=1e-6)

    # Marks need a map to be projected onto; without one they are ignored.
    no_map = make_lap(lap_a.data["Distance"], lap_a.data["Speed"])
    assert compare_payload(no_map, lap_b, marks)["landmarks"] == 0


def test_fastf1_route_validates_and_stores(client, make_lap, monkeypatch):
    calls = []

    def fake_loader(year, gp, session, driver, lap="fastest"):
        calls.append((year, gp, session, driver, lap))
        if driver == "XXX":
            raise ValueError("XXX has no valid quick lap in this session")
        return make_lap(np.arange(0.0, 1001.0, 5.0), np.full(201, 200.0), label="NOR")

    def fake_marks(year, gp, session):
        raise RuntimeError("no network in the test suite")

    monkeypatch.setattr("apextrace.loaders.fastf1_loader.load_fastf1_lap", fake_loader)
    monkeypatch.setattr(
        "apextrace.loaders.fastf1_loader.load_fastf1_corner_marks", fake_marks
    )

    ok = client.post(
        "/api/fastf1",
        json={
            "year": 2024,
            "gp": "Italian Grand Prix",
            "session": "Q",
            "driver": "nor",
        },
    )
    assert ok.status_code == 200 and ok.json()["origin"] == "fastf1"
    assert calls[0] == (2024, "Italian Grand Prix", "Q", "NOR", "fastest")

    bad_driver = client.post(
        "/api/fastf1",
        json={
            "year": 2024,
            "gp": "Italian Grand Prix",
            "session": "Q",
            "driver": "XXX",
        },
    )
    assert bad_driver.status_code == 400
    assert "no valid quick lap" in bad_driver.json()["detail"]

    malformed = client.post("/api/fastf1", json={"year": 1066, "gp": "", "driver": 1})
    assert malformed.status_code == 422


def test_index_and_static_are_served(client):
    assert client.get("/").status_code == 200
    assert "apex-trace" in client.get("/").text


# --- live hub ---------------------------------------------------------


def drain(queue):
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    return messages


def run_through_hub(hub, datagrams):
    """Feed a whole stream, collecting everything a client would get."""
    queue = hub.subscribe()
    messages = []
    for datagram in datagrams:
        hub.handle_datagram(datagram)
        messages.extend(drain(queue))
    hub.unsubscribe(queue)
    return messages


def test_hub_follows_the_session_best_and_reads_the_delta(session):
    datagrams, lap_times = session
    hub = LiveHub(SessionStore())
    messages = run_through_hub(hub, datagrams)

    def samples(lap):
        return [m for m in messages if m["type"] == "sample" and m["lap"] == lap]

    # Lap 1 has nothing to be measured against.
    assert all(m["delta"] is None for m in samples(1))

    # Lap 2 is measured against lap 1, and ends behind by the difference.
    lap_two = [m for m in samples(2) if m["delta"] is not None]
    assert lap_two[-1]["delta"] == pytest.approx(lap_times[1] - lap_times[0], abs=5e-3)
    assert lap_two[-1]["predicted"] == pytest.approx(lap_times[1], abs=5e-3)

    # Lap 2 was slower, so lap 3 is still measured against lap 1...
    lap_three = [m for m in samples(3) if m["delta"] is not None]
    assert lap_three[-1]["delta"] == pytest.approx(
        lap_times[2] - lap_times[0], abs=5e-3
    )
    # ...and, being the quickest, becomes the reference for lap 4.
    laps = [m for m in messages if m["type"] == "lap"]
    assert [m["is_best"] for m in laps] == [True, False, True]
    assert hub.reference.lap.lap_time == pytest.approx(lap_times[2], abs=1e-3)
    assert [m["type"] for m in messages].count("reference") == 2


def test_a_pinned_reference_is_not_displaced_by_a_faster_lap(session):
    datagrams, lap_times = session
    store = SessionStore()
    laps, _ = load_f1_25_laps(datagrams)
    pinned = store.add(laps[1], origin="upload:earlier.f1cap")

    hub = LiveHub(store)
    hub.set_reference(pinned, pinned=True)
    messages = run_through_hub(hub, datagrams)

    assert hub.reference is pinned
    # Even lap 1 has a delta now: the reference was there from the start.
    lap_one = [m for m in messages if m["type"] == "sample" and m["lap"] == 1]
    assert lap_one[-1]["delta"] == pytest.approx(lap_times[0] - lap_times[1], abs=5e-3)

    # Unpinning returns to the best lap of the live session.
    hub.set_reference(None, pinned=False)
    assert hub.reference.origin == "live"
    assert hub.reference.lap.lap_time == pytest.approx(lap_times[2], abs=1e-3)


def test_a_slow_client_loses_samples_but_never_state(session):
    datagrams, _ = session
    hub = LiveHub(SessionStore())
    stalled = hub.subscribe()  # never drained
    for datagram in datagrams:
        hub.handle_datagram(datagram)

    queued = drain(stalled)
    assert len(queued) == stalled.maxsize
    # The last lap and reference messages are still in the queue even
    # though thousands of samples were produced after them.
    assert {"lap", "reference"} <= {m["type"] for m in queued}


def test_snapshot_lets_a_late_client_catch_up(session):
    datagrams, _ = session
    hub = LiveHub(SessionStore())
    half = len(datagrams) // 2
    for datagram in datagrams[:half]:
        hub.handle_datagram(datagram)

    snapshot = hub.snapshot()
    assert snapshot["type"] == "hello"
    assert snapshot["reference"] is not None
    assert snapshot["lap"] == 2
    assert len(snapshot["trace"]) > 100
    assert snapshot["status"]["track"] == "Monza"
    assert len(snapshot["laps"]) == 1


def test_the_automatic_reference_does_not_survive_a_change_of_circuit(session):
    """A best lap at Monza says nothing about a lap of Monaco."""
    datagrams, _ = session
    hub = LiveHub(SessionStore())
    run_through_hub(hub, datagrams)
    assert hub.reference is not None and hub.reference.lap.track == "Monza"

    elsewhere, _ = synthetic_session(
        [profile(), profile()], lap_length=LAP_LENGTH, rate_hz=RATE_HZ, track_id=5
    )
    messages = run_through_hub(hub, elsewhere[:200])
    assert hub.reference is None
    assert all(m["delta"] is None for m in messages if m["type"] == "sample")

    # A reference the user pinned is their call, and stays.
    pinned_hub = LiveHub(SessionStore())
    run_through_hub(pinned_hub, datagrams)
    pinned_hub.set_reference(pinned_hub.reference, pinned=True)
    run_through_hub(pinned_hub, elsewhere[:200])
    assert pinned_hub.reference is not None


def test_hub_counts_what_it_cannot_read(session):
    hub = LiveHub(SessionStore())
    hub.handle_datagram(b"not telemetry")
    hub.handle_datagram(session[0][0])
    assert (hub.packets, hub.bad_packets) == (2, 1)


def test_hub_records_the_raw_stream(tmp_path, session):
    datagrams, _ = session
    path = tmp_path / "recorded.f1cap"
    hub = LiveHub(SessionStore())
    hub.start_recording(path)
    for datagram in datagrams[:500]:
        hub.handle_datagram(datagram)
    hub.stop_recording()

    from apextrace.loaders.f1_25 import replay

    assert list(replay(path)) == datagrams[:500]


def test_hub_receives_over_udp(session):
    """The asyncio datagram endpoint feeds the same path as a replay."""
    import socket

    datagrams, _ = session

    async def scenario():
        hub = LiveHub(SessionStore())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        task = asyncio.create_task(hub.run_udp(port, host="127.0.0.1"))
        await asyncio.sleep(0.1)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for datagram in datagrams[:30]:
                sender.sendto(datagram, ("127.0.0.1", port))
        for _ in range(50):
            if hub.packets == 30:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return hub

    hub = asyncio.run(scenario())
    assert hub.packets == 30
    assert hub.source.startswith("udp:")
    assert hub.status()["track"] == "Monza"


# --- live, end to end -------------------------------------------------


def test_replay_reaches_the_browser_over_a_websocket(capture, session):
    """Capture file -> hub -> WebSocket, the path a tablet would be on."""
    _, lap_times = session
    live = LiveConfig(
        udp_port=None, replay=capture, replay_speed=400.0, replay_loop=True
    )

    with TestClient(create_app(live=live, allowed_hosts=TEST_HOSTS)) as client:
        with client.websocket_connect("/ws/live") as socket:
            hello = socket.receive_json()
            assert hello["type"] == "hello"
            assert hello["status"]["source"] == "replay:session.f1cap"

            seen_lap = None
            seen_delta = None
            for _ in range(50_000):
                message = socket.receive_json()
                if message["type"] == "lap" and seen_lap is None:
                    seen_lap = message
                if message["type"] == "sample" and message["delta"] is not None:
                    seen_delta = message
                if seen_lap and seen_delta:
                    break

        assert seen_lap is not None and seen_delta is not None
        assert any(
            seen_lap["lap"]["lap_time"] == pytest.approx(t, abs=1e-3) for t in lap_times
        )
        # Laps recorded live are in the same store the analysis reads.
        assert any(lap["origin"] == "live" for lap in client.get("/api/laps").json())


def test_websocket_from_another_site_is_refused(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/live", headers={"origin": "https://evil.example"}
        ) as socket:
            socket.receive_json()


def test_reference_route_pins_and_unpins(client, capture):
    lap_id = upload(client, capture).json()["added"][1]["id"]

    pinned = client.post("/api/live/reference", json={"lap_id": lap_id}).json()
    assert pinned["reference"]["id"] == lap_id
    assert client.get("/api/live").json()["pinned"] is True

    # Deleting the reference lap must not leave the hub holding it.
    client.delete(f"/api/laps/{lap_id}")
    state = client.get("/api/live").json()
    assert state["reference"] is None and state["pinned"] is False

    assert (
        client.post("/api/live/reference", json={"lap_id": "nope"}).status_code == 404
    )


# --- command line -----------------------------------------------------


def test_cli_parses_the_documented_invocations():
    parser = build_parser()
    args = parser.parse_args(["serve", "--replay", "x.f1cap", "--replay-speed", "2"])
    assert args.replay.name == "x.f1cap" and args.replay_speed == 2.0
    args = parser.parse_args(["serve", "--no-live", "a.f1cap", "b.f1cap"])
    assert args.no_live and [p.name for p in args.files] == ["a.f1cap", "b.f1cap"]
    assert parser.parse_args(["serve"]).host == "127.0.0.1"


def test_cli_extract_decodes_a_capture_into_a_table(capture, tmp_path, capsys):
    from apextrace.storage import load_laps

    out = tmp_path / "session.csv.gz"
    main(["extract", str(capture), "-o", str(out)])
    assert "wrote 3 laps" in capsys.readouterr().out
    laps = load_laps(out)
    assert len(laps) == 3 and laps[0].source == "f1_25"

    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"junk")
    with pytest.raises(SystemExit, match="junk.bin"):
        main(["extract", str(junk), "-o", str(tmp_path / "x.csv")])


def test_cli_record_reports_a_silent_port(tmp_path):
    with pytest.raises(SystemExit, match="no packets arrived"):
        main(
            [
                "record",
                str(tmp_path / "silent.f1cap"),
                "--udp-port",
                "0",
                "--timeout",
                "0.2",
            ]
        )


def test_cli_refuses_bad_arguments_before_starting_a_server(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit, match="no such capture"):
        serve(parser.parse_args(["serve", "--replay", str(tmp_path / "missing.f1cap")]))

    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"junk")
    with pytest.raises(SystemExit, match="junk.bin"):
        serve(parser.parse_args(["serve", "--no-live", str(junk)]))
