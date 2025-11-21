"""Train an XGBoost model to learn per-job baseline durations using preprocessor.py.

This script:
- Uses `preprocessor.preprocess_stream` and `build_runs` to reconstruct job runs.
- Uses `compute_baseline` to compute per-job baseline duration statistics.
- Builds a training set where the target is `duration_sec` for each completed run.
- Trains an XGBoost regressor using only baseline statistics as features
  (no long-running classification or thresholds are defined here).
- Saves the trained model and metadata for later real-time use.

Usage (from repo root):

    python train_long_running_model.py --input-path <LOG_DIR> \
        --model-out xgb_baseline_model.json \
        --meta-out xgb_baseline_meta.json

If --input-path is omitted, it defaults to `preprocessor.INPUT_PATH`.
"""

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import preprocessor


def load_runs_and_baseline(input_path: str) -> Tuple[List[dict], Dict[str, dict]]:
    """Load completed runs and compute baseline stats per job.

    Returns
    -------
    completed_runs : list of dict
        Output of preprocessor.build_runs.
    baseline_stats : dict
        Output of preprocessor.compute_baseline (keyed by "JobName|JOID").
    """

    events = list(preprocessor.preprocess_stream(input_path))
    completed_runs = preprocessor.build_runs(events)
    baseline_stats = preprocessor.compute_baseline(completed_runs)
    return completed_runs, baseline_stats


def make_training_dataframe(
    completed_runs: List[dict], baseline_stats: Dict[str, dict]
) -> pd.DataFrame:
    """Construct a pandas DataFrame for XGBoost regression.

    Target:
    - `duration_sec` for each completed run.

    Features (per run), based only on baseline statistics:
    - baseline_count, baseline_median, baseline_mean, baseline_stdev,
      baseline_p95, baseline_mad.
    """

    records: List[dict] = []

    for r in completed_runs:
        dur = r.get("duration_sec")
        if dur is None:
            continue

        job_name = r.get("JobName")
        joid = r.get("JOID")
        if job_name is None or joid is None:
            continue

        key = f"{job_name}|{joid}"
        base = baseline_stats.get(key)
        if not base:
            continue

        baseline_p95 = base.get("p95")
        if baseline_p95 is None:
            continue

        records.append(
            {
                # For reference/debugging only
                "JobName": job_name,
                "JOID": joid,

                # Target
                "duration_sec": float(dur),

                # Baseline-derived features
                "baseline_count": float(base.get("count", 0.0)),
                "baseline_median": float(base.get("median", 0.0)),
                "baseline_mean": float(base.get("mean", 0.0)),
                "baseline_stdev": float(base.get("stdev", 0.0)),
                "baseline_p95": float(baseline_p95),
                "baseline_mad": float(base.get("mad", 0.0)),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def train_model(df: pd.DataFrame) -> Tuple[XGBRegressor, dict]:
    """Train an XGBoost regressor to learn baseline durations.

    Uses only per-job baseline statistics as features.
    """

    feature_columns = [
        "baseline_count",
        "baseline_median",
        "baseline_mean",
        "baseline_stdev",
        "baseline_p95",
        "baseline_mad",
    ]

    X = df[feature_columns].astype(float).values
    y = df["duration_sec"].astype(float).values

    # Basic XGBoost regressor; tune as needed.
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
    )
    model.fit(X, y)

    meta = {
        "feature_columns": feature_columns,
    }

    return model, meta


def save_model(model: XGBRegressor, meta: dict, model_out: str, meta_out: str) -> None:
    """Persist XGBoost model and metadata to disk."""

    # Save the underlying booster in JSON format.
    booster = model.get_booster()
    booster.save_model(model_out)

    with open(meta_out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an XGBoost model that learns per-job baseline durations "
            "from historical Autosys runs."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help=(
            "Directory containing Autosys .txt logs. "
            "Defaults to preprocessor.INPUT_PATH if omitted."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="xgb_baseline_model.json",
        help="Path to write the trained XGBoost regressor (JSON format).",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default="xgb_baseline_meta.json",
        help="Path to write JSON metadata (feature columns only).",
    )

    args = parser.parse_args()

    input_path = args.input_path or preprocessor.INPUT_PATH

    print(f"[train] Using input_path={input_path!r}")
    completed_runs, baseline_stats = load_runs_and_baseline(input_path)

    if not completed_runs:
        raise SystemExit("No completed runs found; cannot train model.")

    print(f"[train] Loaded {len(completed_runs)} completed runs")

    # Build baseline.json from baseline_stats for reference/debugging.
    try:
        out_baseline = preprocessor.OUT_BASELINE
        with open(out_baseline, "w", encoding="utf-8") as fh:
            json.dump(baseline_stats, fh, indent=2)
        print(f"[train] Wrote baseline stats to {out_baseline!r}")
    except Exception as exc:  # baseline writing is optional
        print(f"[train] Warning: could not write baseline stats: {exc}")

    df = make_training_dataframe(completed_runs, baseline_stats)
    if df.empty:
        raise SystemExit(
            "Training DataFrame is empty after filtering; "
            "check that durations and baselines are available."
        )

    print(f"[train] Training on {len(df)} runs")

    model, meta = train_model(df)
    save_model(model, meta, args.model_out, args.meta_out)

    print(f"[train] Saved model to {args.model_out!r}")
    print(f"[train] Saved metadata to {args.meta_out!r}")


if __name__ == "__main__":
    main()
