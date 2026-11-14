"""M4: record the F1 25 UDP telemetry stream to a capture file.

Run this while driving. In the game: Settings > Telemetry Settings >
UDP Telemetry On, Format 2025, IP the address of this machine, Port
20777. Leave UDP Send Rate at 30 Hz or raise it to 60.

Nothing is parsed here on purpose. The stream is live and unrepeatable,
so the job during the session is to get the bytes onto disk; analysis
happens afterwards with f1_25_analyse.py, as many times as you like.

    uv run python examples/f1_25_record.py monza.f1cap
    uv run python examples/f1_25_record.py monza.f1cap --duration 600
"""

import argparse
from pathlib import Path

from apextrace.loaders.f1_25 import DEFAULT_PORT, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="capture file to write")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to record; omit to run until Ctrl-C",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds of silence after which the recording stops",
    )
    args = parser.parse_args()

    print(f"listening on UDP {args.port}, writing {args.path} (Ctrl-C to stop)")
    count = record(
        args.path, port=args.port, duration=args.duration, timeout=args.timeout
    )
    if count == 0:
        raise SystemExit(
            f"no packets arrived on port {args.port} in {args.timeout:g} s. Check "
            "that UDP telemetry is on in the game, that the format is 2025, and "
            "that the IP address points at this machine."
        )
    print(f"recorded {count} packets to {args.path}")


if __name__ == "__main__":
    main()
