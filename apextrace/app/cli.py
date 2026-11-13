"""Command line entry point: ``apextrace serve``.

apextrace serve                         # analysis + live F1 25 on UDP 20777
apextrace serve --record monza.f1cap    # same, keeping the raw stream
apextrace serve --replay monza.f1cap    # relive a recorded session
apextrace serve --no-live a.f1cap b.csv # analysis only, files preloaded
"""

import argparse
import sys
from pathlib import Path

from apextrace.loaders.f1_25 import DEFAULT_PORT


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
    return parser


def serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from apextrace.app.ingest import LoadOptions, UnsupportedFile, load_file
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
        try:
            kind, laps, notes = load_file(
                path, LoadOptions(steering_lock=args.steering_lock)
            )
        except (OSError, UnsupportedFile, ValueError) as error:
            raise SystemExit(f"{path}: {error}") from None
        for lap in laps:
            store.add(lap, origin=f"upload:{path.name}")
        print(f"{path}: {kind}, {len(laps)} laps")
        for note in notes:
            print(f"  note: {note}")

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


if __name__ == "__main__":
    main()
