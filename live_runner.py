# runner.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Iterable, Callable

from live_streamer import AutosysLogStreamer, _iter_entries


def make_txt_emitter(out_path: Path) -> Callable[[str], None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("a", encoding="utf-8")

    def emit(line: str) -> None:
        raw = line.rstrip("\n")
        # Write raw log line exactly as received
        f.write(f"{raw}\n")
        f.flush()

    return emit


def stream_with_clock_sync(
    streamer: AutosysLogStreamer,
    emit: Callable[[str], None],
) -> None:
    log_path = streamer.log_path
    speed = streamer.speed

    # Read all entries once (file is finite historic log)
    entries = list(_iter_entries(log_path))
    if not entries:
        return

    # Find first entry that has a timestamp
    first_ts: Optional[datetime] = None
    for e in entries:
        if e.timestamp is not None:
            first_ts = e.timestamp
            break

    if first_ts is None:
        # No timestamps in file → just emit without timing control
        for e in entries:
            if streamer._should_emit(e):
                emit(e.text)
        return

    start_wall = datetime.now()

    for entry in entries:
        if not streamer._should_emit(entry):
            continue

        if entry.timestamp is not None:
            # Offset in original Autosys time
            delta = (entry.timestamp - first_ts).total_seconds()
            # Apply speed factor
            logical_offset = delta / speed
            target_wall = start_wall + timedelta(seconds=logical_offset)

            now = datetime.now()
            wait = (target_wall - now).total_seconds()
            if wait > 0:
                time.sleep(wait)

        # At this point we are as close as possible to the target wall-clock time
        emit(entry.text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Autosys logs, clock-synced to system time, and store output in txt file."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(r"C:\Users\91810\autosys-aiml-job-monitor\src\logs.txt"),
        help="Input Autosys log file (historic).",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("replayed_with_system_time.txt"),
        help="Output txt file (raw log lines, no added timestamps).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = real time, 2.0 = twice as fast, 0.5 = half speed).",
    )
    return parser


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%m/%d/%Y %H:%M:%S")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    streamer = AutosysLogStreamer(
        log_path=args.log_file,
        speed=args.speed,
    )

    emit = make_txt_emitter(args.out_file)
    stream_with_clock_sync(streamer, emit)


if __name__ == "__main__":
    main()