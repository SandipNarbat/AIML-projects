import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, AsyncGenerator

import aiofiles
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
# BASE_PATH = Path("D:/AI SBI/Autosys/logs")
# FILE_PREFIX = "status_queue_8app.txt"
POLL_INTERVAL = 0.5  # seconds


app = FastAPI(title="Realtime Queue Monitor SSE")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
# def build_log_path() -> Path:
#     """Always points to the current file (no date logic here yet)."""
#     return BASE_PATH / FILE_PREFIX


async def read_state(path: Path) -> Dict[str, List[int]]:
    state = {}

    async with aiofiles.open(path, "r") as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            if "VV3Q" in line:
                continue
            line = line.replace("NQKE", "-1").strip()
            parts = line.split(",")
            queue = parts[0]
            metrics = []

            for v in parts[1:]:
                try:
                    num = int(v.strip())
                except (ValueError, TypeError):
                    num = 0

                metrics.append(0 if 1 <= num < 150 else num)
            state[queue] = metrics
    return state


def diff_states(
    old: Dict[str, List[int]],
    new: Dict[str, List[int]]
) -> List[dict]:
    """
    Computes delta between two states.
    Returns list of changed queues only.
    """
    deltas = []

    for queue, new_metrics in new.items():
        old_metrics = old.get(queue)

        # new queue appeared
        if old_metrics is None:
            deltas.append({
                "queue": queue,
                "type": "new",
                "metrics": new_metrics
            })
            continue

        changes = []
        for i, (o, n) in enumerate(zip(old_metrics, new_metrics)):
            if o != n:
                changes.append({
                    "index": i,
                    "old": o,
                    "new": n
                })

        if changes:
            deltas.append({
                "queue": queue,
                "type": "update",
                "changes": changes
            })

    return deltas


# --------------------------------------------------
# SSE STREAM
# --------------------------------------------------
async def event_stream() -> AsyncGenerator[str, None]:
    log_path = Path("queue_file.txt")

    # wait until file exists
    while not log_path.exists():
        await asyncio.sleep(1)

    prev_state: Dict[str, List[int]] = {}
    last_mtime = 0
    last_ping = time.time()

    # initial snapshot
    curr_state = await read_state(log_path)
    prev_state = curr_state

    yield (
        "event: snapshot\n"
        f"data: {json.dumps(curr_state)}\n\n"
    )

    while True:
        try:
            stat = log_path.stat()

            # file changed
            if stat.st_mtime != last_mtime:
                last_mtime = stat.st_mtime

                curr_state = await read_state(log_path)
                deltas = diff_states(prev_state, curr_state)

                for d in deltas:
                    yield (
                        "event: delta\n"
                        f"data: {json.dumps(d)}\n\n"
                    )

                prev_state = curr_state

            # keep connection alive
            if time.time() - last_ping > 10:
                yield ": ping\n\n"
                last_ping = time.time()

            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            yield (
                "event: error\n"
                f"data: {json.dumps({'error': str(e)})}\n\n"
            )
            await asyncio.sleep(1)


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@app.get("/events")
async def events():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
