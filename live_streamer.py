from __future__ import annotations
import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

_TIMESTAMP_RE = re.compile(r"\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\]")
_TS_FMT = "%m/%d/%Y %H:%M:%S"


@dataclass(frozen=True)
class LogEntry:
    """Represents one logical Autosys log entry."""

    timestamp: Optional[datetime]
    text: str


def _split_line_into_entries(line: str) -> Sequence[LogEntry]:
    """Split a physical line into logical entries if it embeds multiple timestamps."""

    matches = list(_TIMESTAMP_RE.finditer(line))
    if not matches:
        return (LogEntry(timestamp=None, text=line.strip()),)

    entries: list[LogEntry] = []
    for idx, match in enumerate(matches):
        ts = datetime.strptime(match.group(1), _TS_FMT)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        chunk = line[start:end].strip()
        entries.append(LogEntry(timestamp=ts, text=chunk))
    return entries


def _iter_entries(path: Path) -> Iterator[LogEntry]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.rstrip("\n")
            if not stripped:
                continue
            yield from _split_line_into_entries(stripped)


class AutosysLogStreamer:
    """Streams historic Autosys logs as if they were emitted live."""

    def __init__(
        self,
        log_path: Path,
        *,
        speed: float = 1.0,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        loop: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.log_path = log_path
        self.speed = speed
        self.since = since
        self.until = until
        self.loop = loop
        self._sleep = sleeper

    def _should_emit(self, entry: LogEntry) -> bool:
        ts = entry.timestamp
        if ts is None:
            return True
        if self.since and ts < self.since:
            return False
        if self.until and ts > self.until:
            return False
        return True

    def _compute_wait(self, prev_ts: Optional[datetime], current_ts: Optional[datetime]) -> float:
        if prev_ts is None or current_ts is None:
            return 0.0
        delta = (current_ts - prev_ts).total_seconds()
        if delta <= 0:
            return 0.0
        return delta / self.speed

    def stream(self, emit):

        entries = list(_iter_entries(self.log_path))
        if not entries:
            return

        # 1️⃣ Find first log timestamp
        first_ts = None
        for e in entries:
            if e.timestamp:
                first_ts = e.timestamp
                break

        if first_ts is None:
            # No timestamps → just emit
            for e in entries:
                emit(e.text)
            return

        # 2️⃣ Capture system start time
        wall_start = time.time()  # real wall clock seconds

        # 3️⃣ Replay with PERFECT time sync
        for entry in entries:

            if not self._should_emit(entry):
                continue

            if entry.timestamp:
                # Autosys elapsed time
                auto_offset = (entry.timestamp - first_ts).total_seconds()

                # Target system clock moment
                target_wall = wall_start + auto_offset

                # Current system time
                now = time.time()
                wait = target_wall - now

                if wait > 0:
                    time.sleep(wait)

            # Emit the actual log line
            emit(entry.text)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, _TS_FMT)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Autosys logs with their original timing offsets.",
    )
    parser.add_argument(
        "--log-file",
        default="D:\AUTOSYSLOG\event_demon.SBI.09012025.txt",
        type=Path,
    )
    parser.add_argument(
        "--speed",
        default=1.0,
        type=float,
        help="Speed multiplier relative to real time (e.g. 2.0 = twice as fast).",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Optional lower bound timestamp in MM/DD/YYYY HH:MM:SS.",
    )
    parser.add_argument(
        "--until",
        type=str,
        help="Optional upper bound timestamp in MM/DD/YYYY HH:MM:SS.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop back to the start once the end of the file is reached.",
    )
    return parser

def main(argv: Optional[Iterable[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    streamer = AutosysLogStreamer(
        log_path=args.log_file,
        speed=args.speed,
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
        loop=args.loop,
    )
    streamer.stream()

if __name__ == "__main__":
    main()
