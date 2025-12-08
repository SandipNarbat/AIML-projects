from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def _load_baselines(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _compute_stats(durations: List[float], stdev_multiplier: float) -> Dict:
    if not durations:
        return {
            "baseline_count": 0,
            "baseline_mean": 0.0,
            "baseline_stdev": 0.0,
            "baseline_p95": 0.0,
            "baseline_mad": 0.0,
            "baseline_statistical": 0.0,
            "durations": [],
        }

    arr = np.asarray(durations, dtype=float)
    mean = float(np.mean(arr))
    stdev = float(np.std(arr))
    p95 = _percentile(durations, 95)
    mad = _mad(durations)
    baseline_statistical = mean + stdev_multiplier * stdev

    return {
        "baseline_count": int(len(arr)),
        "baseline_mean": mean,
        "baseline_stdev": stdev,
        "baseline_p95": p95,
        "baseline_mad": mad,
        "baseline_statistical": baseline_statistical,
        "durations": list(durations),
    }


def update_baselines(
    baseline_path: Path,
    durations_by_job: Dict[str, List[float]],
    *,
    stdev_multiplier: float = 2.0,
) -> None:
    baselines = _load_baselines(baseline_path)

    for job_key, durations in durations_by_job.items():
        existing = baselines.get(job_key, {})
        existing_durations = existing.get("durations", [])
        combined = existing_durations + durations
        baselines[job_key] = _compute_stats(combined, stdev_multiplier)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(baselines, fh, indent=2)

