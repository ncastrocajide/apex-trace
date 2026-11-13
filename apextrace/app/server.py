"""The web app: post-session analysis and the live delta, in one place.

A thin layer on purpose. The analysis is the engine's, shaped for the
browser by analysis.py; the live path is the hub's. What is left for
this module is routing, validating what comes in from outside, and
owning the lifetime of the live source.

The app has no accounts and no authentication, and is meant to be run
on the machine it is used from, or on a home network to put the live
view on a tablet next to the wheel. The one protection it does carry is
a same-origin check on everything that changes state or opens the live
socket, so that a web page open in the same browser cannot drive it
behind the user's back.
"""

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from apextrace.app.analysis import NotComparable, compare_payload, lap_payload
from apextrace.app.hub import LiveHub
from apextrace.app.ingest import LoadOptions, UnsupportedFile, load_file
from apextrace.app.store import SessionStore, StoredLap

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Largest upload accepted. A capture is around 6 MB per minute at 60 Hz,
# so this is a long session with room to spare, and still a bound.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_CHUNK = 1024 * 1024


@dataclass
class LiveConfig:
    """Where the live view gets its stream from."""

    udp_port: int | None = 20777
    udp_host: str = "0.0.0.0"
    replay: Path | None = None
    replay_speed: float = 1.0
    replay_loop: bool = False
    record: Path | None = None


class FastF1Request(BaseModel):
    """A lap to download from official F1 timing."""

    year: int = Field(ge=2018, le=2100)
    gp: str = Field(min_length=1, max_length=80)
    session: str = Field(min_length=1, max_length=20)
    driver: str = Field(min_length=2, max_length=3)
    lap: int | Literal["fastest"] = "fastest"


class ReferenceRequest(BaseModel):
    """The lap to pin as live reference; null returns to session best."""

    lap_id: str | None = None


def _same_origin(origin: str | None, host: str | None) -> bool:
    """True unless a browser says the request came from another site.

    Non-browser clients send no Origin and are let through: the check
    exists to stop pages on other sites using the user's browser as a
    way in, which is the only cross-site route to a local app.
    """
    if origin is None:
        return True
    return host is not None and urlsplit(origin).netloc == host


def create_app(
    store: SessionStore | None = None, live: LiveConfig | None = None
) -> FastAPI:
    """Build the app. live=None serves analysis only, with no stream."""
    store = store if store is not None else SessionStore()
    hub = LiveHub(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task] = []
        if live is not None:
            if live.record is not None:
                hub.start_recording(live.record)
            if live.replay is not None:
                tasks.append(
                    asyncio.create_task(
                        hub.run_replay(live.replay, live.replay_speed, live.replay_loop)
                    )
                )
            elif live.udp_port is not None:
                tasks.append(
                    asyncio.create_task(hub.run_udp(live.udp_port, live.udp_host))
                )
            tasks.append(asyncio.create_task(hub.run_status()))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            hub.stop_recording()

    app = FastAPI(title="apex-trace", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.hub = hub

    @app.middleware("http")
    async def guard_origin(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(
            request.headers.get("origin"), request.headers.get("host")
        ):
            return JSONResponse(
                {"detail": "cross-origin request refused"}, status_code=403
            )
        return await call_next(request)

    def _lap_or_404(lap_id: str) -> StoredLap:
        stored = store.get(lap_id)
        if stored is None:
            raise HTTPException(404, f"no lap with id {lap_id!r}")
        return stored

    # --- laps ---------------------------------------------------------

    @app.get("/api/laps")
    def list_laps() -> list[dict]:
        return [stored.summary() for stored in store.all()]

    @app.get("/api/laps/{lap_id}")
    def get_lap(lap_id: str) -> dict:
        return lap_payload(_lap_or_404(lap_id).lap)

    @app.delete("/api/laps/{lap_id}")
    async def delete_lap(lap_id: str) -> dict:
        # async on purpose: it may touch the hub, which belongs to the
        # event loop.
        stored = _lap_or_404(lap_id)
        store.remove(lap_id)
        if hub.reference is stored:
            # Fall back to the session best rather than keep measuring
            # against a lap the user just threw away.
            hub.set_reference(None, pinned=False)
        return {"removed": lap_id}

    @app.get("/api/compare")
    def compare(a: str, b: str) -> dict:
        if a == b:
            raise HTTPException(400, "pick two different laps to compare")
        try:
            return compare_payload(_lap_or_404(a).lap, _lap_or_404(b).lap)
        except NotComparable as error:
            raise HTTPException(400, str(error)) from None

    # --- loading ------------------------------------------------------

    @app.post("/api/upload")
    def upload(
        file: UploadFile,
        steering_lock: float | None = Form(default=None, gt=0.0, le=1080.0),
    ) -> dict:
        """Load every lap from an uploaded telemetry file.

        The upload is spooled to a temporary file and handed to the
        loaders by path, the same way a file on disk would be. The
        client's file name is only ever used as a label. steering_lock
        [deg] is for exports that log steering as a wheel angle.
        """
        name = Path(file.filename or "upload").name
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload"
            size = 0
            with path.open("wb") as out:
                while chunk := file.file.read(_CHUNK):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file is too large")
                    out.write(chunk)
            try:
                kind, laps, notes = load_file(
                    path, LoadOptions(steering_lock=steering_lock)
                )
            except UnsupportedFile as error:
                raise HTTPException(415, str(error)) from None
            except ValueError as error:
                raise HTTPException(400, f"could not read {name}: {error}") from None

        added = [store.add(lap, origin=f"upload:{name}") for lap in laps]
        return {
            "format": kind,
            "added": [stored.summary() for stored in added],
            "notes": notes,
        }

    @app.post("/api/fastf1")
    def load_fastf1(request: FastF1Request) -> dict:
        """Download one lap from official F1 timing. Needs the network."""
        # Imported here: FastF1 is slow to import and only this endpoint
        # needs it, so the app starts fast and works offline otherwise.
        from apextrace.loaders.fastf1_loader import load_fastf1_lap

        try:
            lap = load_fastf1_lap(
                request.year,
                request.gp,
                request.session,
                request.driver.upper(),
                lap=request.lap,
            )
        except Exception as error:  # noqa: BLE001 - FastF1 raises many types
            raise HTTPException(
                400, f"could not load that lap from F1 timing: {error}"
            ) from None
        return store.add(lap, origin="fastf1").summary()

    # --- live ---------------------------------------------------------

    @app.get("/api/live")
    def live_state() -> dict:
        return hub.snapshot()

    @app.post("/api/live/reference")
    async def set_reference(request: ReferenceRequest) -> dict:
        # async on purpose: the hub belongs to the event loop.
        if request.lap_id is None:
            hub.set_reference(None, pinned=False)
        else:
            hub.set_reference(_lap_or_404(request.lap_id), pinned=True)
        reference = hub.reference
        return {"reference": None if reference is None else reference.summary()}

    @app.websocket("/ws/live")
    async def live_socket(socket: WebSocket) -> None:
        if not _same_origin(socket.headers.get("origin"), socket.headers.get("host")):
            await socket.close(code=1008)
            return
        await socket.accept()
        queue = hub.subscribe()

        async def send() -> None:
            await socket.send_json(hub.snapshot())
            while True:
                await socket.send_json(await queue.get())

        async def receive() -> None:
            # Clients send nothing; this only exists to notice them go.
            while True:
                await socket.receive_text()

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    task.result()
        finally:
            hub.unsubscribe(queue)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # --- frontend -----------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
