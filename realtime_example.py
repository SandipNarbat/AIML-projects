"""Example of using the trained model + per-JOID baselines for real-time scoring.

This script shows how to:
- Load `baseline_stats.json` (per-"JobName|JOID" baseline statistics)
- Load the trained XGBoost baseline-duration model and metadata
- Estimate the expected (baseline) duration for a given job (JobName + JOID)

It is intended as a reference for how your real-time service can integrate the
artifacts produced by `train_long_running_model.py`.

Usage examples (from repo root):

    python realtime_example.py --job-name "MY_JOB" --joid 12345

You can also override the default paths to the artifacts if needed:

    python realtime_example.py \
        --job-name "MY_JOB" --joid 12345 \
        --baseline-path baseline_stats.json \
        --model-path xgb_long_running_model.json \
        --meta-path xgb_long_running_meta.json
"""

import argparse
import json
from typing import Any, Dict

import numpy as np
import xgboost as xgb


def load_baselines(path: str) -> Dict[str, Dict[str, Any]]:
    """Load per-"JobName|JOID" baselines from JSON.

    This should be the same file written by `train_long_running_model.py`,
    usually `baseline_stats.json`.
    """

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_and_meta(model_path: str, meta_path: str) -> tuple[xgb.Booster, dict]:
    """Load the trained XGBoost booster and its metadata."""

    booster = xgb.Booster()
    booster.load_model(model_path)

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    return booster, meta


def build_feature_vector(baseline: Dict[str, Any], feature_columns: list[str]) -> np.ndarray:
    """Build a single feature vector from a baseline stats dict.

    The feature_columns list is expected to match what `train_long_running_model.py`
    used when training the model (e.g. ["baseline_count", "baseline_median", ...]).
    """

    values = []
    for col in feature_columns:
        values.append(float(baseline.get(col.split("baseline_")[-1], baseline.get(col, 0.0))))
    # Note: The above line is defensive; if the baseline dict already has
    # keys named exactly like feature_columns (which it does for this project),
    # the second branch `baseline.get(col, 0.0)` is used.

    return np.asarray([values], dtype=float)


def score_job(
    job_name: str,
    joid: int,
    baselines: Dict[str, Dict[str, Any]],
    booster: xgb.Booster,
    meta: dict,
) -> float:
    """Return the predicted baseline duration for the job, from its baseline stats.

    This uses only the per-job baseline statistics. The online system can later
    decide how to compare actual runtime vs. this predicted baseline.
    """

    key = f"{job_name}|{joid}"
    baseline = baselines.get(key)
    if baseline is None:
        raise KeyError(f"No baseline found for key {key!r}")

    feature_columns = meta["feature_columns"]
    x = build_feature_vector(baseline, feature_columns)

    dmatrix = xgb.DMatrix(x, feature_names=feature_columns)
    pred_duration = float(booster.predict(dmatrix)[0])
    return pred_duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Example: estimate a job's baseline duration (JobName + JOID) "
            "using per-JOID baselines and the trained XGBoost model."
        )
    )
    parser.add_argument("--job-name", required=True, help="JobName of the Autosys job")
    parser.add_argument("--joid", type=int, required=True, help="JOID of the Autosys job")
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="baseline_stats.json",
        help="Path to the per-JOID baseline JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="xgb_long_running_model.json",
        help="Path to the trained XGBoost model JSON file.",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="xgb_long_running_meta.json",
        help="Path to the model metadata JSON file.",
    )

    args = parser.parse_args()

    print(f"[realtime] Loading baselines from {args.baseline_path!r}")
    baselines = load_baselines(args.baseline_path)

    print(f"[realtime] Loading model from {args.model_path!r}")
    booster, meta = load_model_and_meta(args.model_path, args.meta_path)

    pred_duration = score_job(args.job_name, args.joid, baselines, booster, meta)

    print(f"[realtime] Job: JobName={args.job_name!r}, JOID={args.joid}")
    print(f"[realtime] Predicted baseline duration (seconds) = {pred_duration:.2f}")


if __name__ == "__main__":
    main()
