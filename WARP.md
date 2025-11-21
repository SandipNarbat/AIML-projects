# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository currently consists of a single Python script, `preprocessor.py`, which processes Autosys log files to reconstruct job runs and compute baseline execution-duration statistics per job.

The script:
- Reads `.txt` log files from a configured directory (`INPUT_PATH`).
- Parses log lines into structured event records (timestamps, job IDs, run IDs, status, machine, etc.).
- Groups events into logical "runs" per `(JOID, RUNID)` and derives start, running, and end timestamps.
- Computes per-job duration statistics (median, mean, stdev, p95, MAD, and a sample of durations).

## How to run the code

### Prerequisites

- Python 3.x installed and available on the PATH.
- Autosys log files present in the directory specified by `INPUT_PATH` inside `preprocessor.py` (currently a hard-coded absolute path).

### Running the main script

From the repository root (`AIML-projects`), run:

```bash path=null start=null
python preprocessor.py
```

This will:
- Scan the `INPUT_PATH` directory for `.txt` log files.
- Stream and parse the logs into events.
- Build completed runs from those events.

Note: As written, the `__main__` block collects `events` and `completed_runs` but does not yet persist the computed baseline or print a summary. Future changes will likely want to call `compute_baseline(completed_runs)` and then store the result (e.g., to `OUT_BASELINE`).

### Basic syntax check ("lint")

There is no dedicated linter or style configuration in this repo. To perform a basic syntax check using the Python standard library, run:

```bash path=null start=null
python -m py_compile preprocessor.py
```

If this command exits without errors, the file is syntactically valid.

### Tests

There is currently no test suite or testing framework configured in this repository. If you add tests (for example, with `pytest`), document the exact commands here so future agents can run them.

## High-level architecture and data flow

### Modules and entry point

- `preprocessor.py` is both the main script and the only module.
- The `if __name__ == "__main__":` block is the runtime entry point and currently performs:
  - `events = list(preprocess_stream(INPUT_PATH))`
  - `completed_runs = build_runs(events)`

### Log parsing layer

Key elements:
- Global regex patterns (`TS_RE`, `STATUS_RE`, `JOB_RE`, `MACHINE_RE`, `JOID_RE`, `RUNID_RE`) define how to extract fields from each log line.
- `RUNNING_EQUIVALENTS` normalizes multiple running-like statuses (e.g., `RUN`, `LONG_RUNNING`) into a single logical `"RUNNING"` status.

`preprocess_stream(input_path)`:
- Iterates over all `.txt` files in the given directory in sorted order.
- For each line, extracts:
  - Timestamp (`ts`) via `TS_RE` and `parse_ts()`.
  - `Status`, `JobName`, `Machine`, `JOID`, and `RUNID` via the respective regexes.
- Normalizes `Status` to uppercase and consolidates running-like values to `"RUNNING"`.
- Yields a dictionary per log line with the keys:
  - `"JOID"`, `"JobName"`, `"Machine"`, `"RUNID"`, `"Status"`, `"ts"`.

This function is intentionally a generator: it can be consumed as a stream to avoid loading all logs into memory at once (though the current `__main__` block materializes it with `list(...)`).

### Run reconstruction layer

`build_runs(events)`:
- Consumes the event stream produced by `preprocess_stream`.
- Filters out events that lack either `JOID` or `RUNID` (these are currently ignored rather than attached to a run).
- Uses `(JOID, RUNID)` as the unique run key and maintains an in-memory `runs` dict of partially-constructed runs.
- For each event, updates the run record as follows:
  - `Status == "STARTING"`: sets `start_ts` if not already set.
  - `Status == "RUNNING"`:
    - If `running_ts` is unset, sets it.
    - If `start_ts` is missing, uses the `RUNNING` timestamp as both a synthetic start and running time.
  - `Status in {"SUCCESS", "FAILURE"}`:
    - Sets `end_ts` and `end_status`.
    - Backfills `running_ts` with `start_ts` or `end_ts` if it was never observed.
    - Computes `duration_sec` primarily as `(end_ts - start_ts).total_seconds()` when both timestamps are available; otherwise, falls back to `None`.
    - Appends a completed run record to `completed` and removes the run from the `runs` working set.

Return value:
- A list of normalized run records, each with:
  - `"JOID"`, `"RUNID"`, `"JobName"`, `"Machine"`,
  - `"start_ts"`, `"running_ts"`, `"end_ts"`, `"end_status"`,
  - `"duration_sec"` (may be `None` if duration cannot be computed).

This function encapsulates the temporal logic of job lifecycle reconstruction from raw log events.

### Baseline statistics layer

`compute_baseline(completed_runs)`:
- Aggregates `duration_sec` for each logical job key:
  - The key is `f"{JobName}|{JOID}"`, allowing differentiation between jobs that share a name but have different IDs.
- Excludes runs with `None` or negative durations.
- For each job key, computes:
  - `count`: number of valid duration samples.
  - `median`: median duration.
  - `mean`: arithmetic mean.
  - `stdev`: sample standard deviation (0.0 when fewer than 2 samples).
  - `p95`: 95th percentile using a simple linear interpolation helper `percentile`.
  - `mad`: median absolute deviation using the helper `mad`.
  - `sample_durations`: up to the first 50 sorted durations (for inspection or debugging).

Return value:
- A dictionary mapping `"{JobName}|{JOID}"` to a stats dictionary with the fields above.

Related helpers:
- `percentile(sorted_list, p)`: computes a simple p-th percentile on a pre-sorted list using linear interpolation between neighboring points.
- `mad(data)`: median absolute deviation around the median, used as a robust dispersion metric.

### Configuration and extension points

- `INPUT_PATH` is currently a hard-coded absolute path to the Autosys log directory. Future improvements may include:
  - Making this configurable via environment variable or command-line argument.
  - Supporting multiple input directories or glob patterns.
- `OUT_BASELINE` is defined but not yet used in the main block. When wiring in baseline computation and persistence, the intended flow is:
  - `completed_runs = build_runs(events)`
  - `baseline = compute_baseline(completed_runs)`
  - Serialize `baseline` to JSON and write to `OUT_BASELINE`.

Agents modifying this script should preserve the three logical layers (parsing → run reconstruction → baseline statistics) and keep the generator-based design of `preprocess_stream` where feasible, to maintain scalability on large log sets.
