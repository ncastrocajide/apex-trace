"""Command line entry point: ``apextrace``.

apextrace serve                         # analysis + live F1 25 on UDP 20777
apextrace serve --record monza.f1cap    # same, keeping the raw stream
apextrace serve --replay monza.f1cap    # relive a recorded session
apextrace serve --no-live a.f1cap b.csv # analysis only, files preloaded
apextrace record monza.f1cap            # just record the F1 25 stream
apextrace extract monza.f1cap -o monza.csv   # decode to a table of laps

Only ``serve`` needs the app's optional dependencies; recording and
extracting run on the engine alone.
"""

import argparse
import sys
from pathlib import Path

from apextrace.app.ingest import LoadOptions, UnsupportedFile, load_file
from apextrace.lap import Lap
from apextrace.loaders.f1_25 import DEFAULT_PORT, record
from apextrace.storage import save_laps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apextrace", description="Source-agnostic race telemetry analysis."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser(
        "serve",
        help="run the web app",
        description="Run the web app: post-session analysis and live delta.",
    )
    serve.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="telemetry files to load at start-up (captures, exports)",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to serve the app on; use 0.0.0.0 to reach it from a "
        "tablet or phone on the same network (the app has no login)",
    )
    serve.add_argument("--port", type=int, default=8000, help="port for the app")
    serve.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="extra host name the app may be reached under (IP addresses, "
        "localhost and this machine's name are always accepted); repeatable",
    )
    serve.add_argument(
        "--udp-port",
        type=int,
        default=DEFAULT_PORT,
        help="UDP port the game sends telemetry to",
    )
    serve.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="FILE",
        help="also write the raw live stream to a capture file",
    )
    serve.add_argument(
        "--replay",
        type=Path,
        default=None,
        metavar="FILE",
        help="replay a capture file as the live stream instead of listening",
    )
    serve.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="pacing of the replay; 2 is twice as fast as recorded",
    )
    serve.add_argument(
        "--loop", action="store_true", help="restart the replay when it ends"
    )
    serve.add_argument(
        "--no-live", action="store_true", help="analysis only, no live stream"
    )
    serve.add_argument(
        "--steering-lock",
        type=float,
        default=None,
        metavar="DEG",
        help="steering wheel angle at full lock, for MoTeC exports that log "
        "steering in degrees (files given on the command line)",
    )

    rec = commands.add_parser(
        "record",
        help="record the F1 25 UDP stream to a capture file",
        description="Record the raw F1 25 UDP stream to a capture file. Nothing "
        "is decoded while recording; extract or serve the file afterwards.",
    )
    rec.add_argument("path", type=Path, help="capture file to write")
    rec.add_argument("--udp-port", type=int, default=DEFAULT_PORT)
    rec.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to record; omit to run until Ctrl-C",
    )
    rec.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds of silence after which the recording stops",
    )

    extract = commands.add_parser(
        "extract",
        help="decode telemetry files into one table of canonical laps",
        description="Decode captures and exports into a single table of "
        "canonical laps (CSV, or .csv.gz), which reloads without decoding.",
    )
    extract.add_argument("files", nargs="+", type=Path, help="files to decode")
    extract.add_argument(
        "-o", "--output", type=Path, required=True, help="table to write"
    )
    extract.add_argument("--steering-lock", type=float, default=None, metavar="DEG")
    return parser


def _load_files(paths: list[Path], steering_lock: float | None) -> list[Lap]:
    """Load every lap of every file, saying what each one gave."""
    laps: list[Lap] = []
    for path in paths:
        try:
            kind, found, notes = load_file(
                path, LoadOptions(steering_lock=steering_lock)
            )
        except (OSError, UnsupportedFile, ValueError) as error:
            raise SystemExit(f"{path}: {error}") from None
        print(f"{path}: {kind}, {len(found)} laps")
        for note in notes:
            print(f"  note: {note}")
        laps.extend(found)
    return laps


def run_record(args: argparse.Namespace) -> None:
    print(f"listening on UDP {args.udp_port}, writing {args.path} (Ctrl-C to stop)")
    count = record(
        args.path, port=args.udp_port, duration=args.duration, timeout=args.timeout
    )
    if count == 0:
        raise SystemExit(
            f"no packets arrived on port {args.udp_port} in {args.timeout:g} s. "
            "Check that UDP telemetry is on in the game, that the format is "
            "2025, and that the IP address points at this machine."
        )
    print(f"recorded {count} packets to {args.path}")


def run_extract(args: argparse.Namespace) -> None:
    laps = _load_files(args.files, args.steering_lock)
    if not laps:
        raise SystemExit("no complete laps found; nothing written")
    save_laps(laps, args.output)
    print(f"wrote {len(laps)} laps to {args.output}")


def serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from apextrace.app.server import LiveConfig, create_app
        from apextrace.app.store import SessionStore
    except ImportError as error:
        raise SystemExit(
            f"the app needs its optional dependencies ({error.name} is missing): "
            "run `uv sync --extra app`"
        ) from None

    if args.replay is not None and not args.replay.is_file():
        raise SystemExit(f"no such capture file: {args.replay}")
    if args.replay_speed <= 0.0:
        raise SystemExit("--replay-speed must be positive")
    if args.record is not None and args.replay is not None:
        raise SystemExit("--record and --replay cannot be combined")

    store = SessionStore()
    for path in args.files:
        for lap in _load_files([path], args.steering_lock):
            store.add(lap, origin=f"upload:{path.name}")

    live = None
    if not args.no_live:
        live = LiveConfig(
            udp_port=args.udp_port,
            replay=args.replay,
            replay_speed=args.replay_speed,
            replay_loop=args.loop,
            record=args.record,
        )
        if args.replay is not None:
            print(f"live view: replaying {args.replay}")
        else:
            print(f"live view: listening for F1 25 telemetry on UDP {args.udp_port}")

    print(f"apex-trace on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(
        create_app(store=store, live=live, allowed_hosts=args.allow_host),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        serve(args)
    elif args.command == "record":
        run_record(args)
    elif args.command == "extract":
        run_extract(args)


if __name__ == "__main__":
    main()
