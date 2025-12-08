   # code2.py

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import time

# ============================================================================
# ✅ UPDATED FEATURE COLUMNS WITH STATISTICAL BASELINE
# ============================================================================

FEATURE_COLUMNS = [
    "baseline_count",
    "baseline_mean",
    "baseline_stdev",           # ✅ NEW: Variability measure
    "baseline_p95",
    "baseline_mad",
    "baseline_statistical",     # ✅ NEW: Mean + 2*stdev (95.4% confidence)
]

# ============================================================================
# MODEL EVALUATION METRICS
# ============================================================================

@dataclass
class ModelMetrics:
    """✅ Comprehensive performance metrics for model evaluation"""
    rmse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error
    r2: float
    model_size_mb: float
    training_time_sec: float
    
    def __str__(self):
        return (f"RMSE: {self.rmse:.2f}s | MAE: {self.mae:.2f}s | "
                f"MAPE: {self.mape:.2%} | R²: {self.r2:.4f} | "
                f"Size: {self.model_size_mb:.2f}MB | Time: {self.training_time_sec:.2f}s")
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging"""
        return {
            'rmse': float(self.rmse),
            'mae': float(self.mae),
            'mape': float(self.mape),
            'r2': float(self.r2),
            'model_size_mb': float(self.model_size_mb),
            'training_time_sec': float(self.training_time_sec)
        }

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   training_time: float, model_size: float) -> ModelMetrics:
    """
    ✅ Compute comprehensive metrics for model evaluation
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        training_time: Training time in seconds
        model_size: Model size in MB
    
    Returns:
        ModelMetrics object with all metrics
    """
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = 0.0
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return ModelMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        model_size_mb=model_size,
        training_time_sec=training_time
    )

# ============================================================================
# XGBOOST MODEL TRAINER
# ============================================================================

class XGBoostTrainer:
    """Fast, accurate XGBoost model trainer"""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              eval_split: float = 0.2) -> Tuple[xgb.Booster, ModelMetrics]:
        """
        Train XGBoost model and return comprehensive metrics
        
        Args:
            X: Feature matrix
            y: Target values
            eval_split: Fraction of data for evaluation
        
        Returns:
            Tuple of (trained model, metrics)
        """
        start_time = time.time()
        
        # Split data
        n_samples = len(X)
        n_eval = int(n_samples * eval_split)
        
        X_train, X_eval = X[:-n_eval], X[-n_eval:]
        y_train, y_eval = y[:-n_eval], y[-n_eval:]
        
        # Optimized parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'tree_method': 'hist',
            'device': 'cpu',
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=self.feature_columns)
        
        # Early stopping
        evals = [(dtrain, 'train'), (deval, 'eval')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred_eval = model.predict(deval)
        
        # Get model size
        import sys
        model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
        
        metrics = compute_metrics(y_eval, y_pred_eval, training_time, model_size)
        
        return model, metrics

# ============================================================================
# LIGHTGBM MODEL TRAINER
# ============================================================================

class LightGBMTrainer:
    """Ultra-fast LightGBM for real-time retraining"""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
    
    def train(self, X: np.ndarray, y: np.ndarray,
              eval_split: float = 0.2) -> Tuple[lgb.Booster, ModelMetrics]:
        """
        Train LightGBM and return comprehensive metrics
        
        Args:
            X: Feature matrix
            y: Target values
            eval_split: Fraction of data for evaluation
        
        Returns:
            Tuple of (trained model, metrics)
        """
        start_time = time.time()
        
        # Split data
        n_samples = len(X)
        n_eval = int(n_samples * eval_split)
        
        X_train, X_eval = X[:-n_eval], X[-n_eval:]
        y_train, y_eval = y[:-n_eval], y[-n_eval:]
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'verbose': -1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_columns)
        valid_data = lgb.Dataset(X_eval, label=y_eval, feature_name=self.feature_columns, 
                                reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, valid_data],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred_eval = model.predict(X_eval)
        
        # Get model size
        import sys
        model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
        
        metrics = compute_metrics(y_eval, y_pred_eval, training_time, model_size)
        
        return model, metrics

# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory for creating and training models"""
    
    @staticmethod
    def train_xgboost(X: np.ndarray, y: np.ndarray,
                     feature_columns: List[str]) -> Tuple[xgb.Booster, ModelMetrics]:
        """Train XGBoost and return metrics"""
        trainer = XGBoostTrainer(feature_columns)
        return trainer.train(X, y)
    
    @staticmethod
    def train_lightgbm(X: np.ndarray, y: np.ndarray,
                      feature_columns: List[str]) -> Tuple[lgb.Booster, ModelMetrics]:
        """Train LightGBM and return metrics"""
        trainer = LightGBMTrainer(feature_columns)
        return trainer.train(X, y)

# ============================================================================
# BASELINE STATISTICS GENERATION - ✅ CRITICAL: Mean + 2*StdDev Method
# ============================================================================

def generate_baseline_statistics(job_durations_dict: dict) -> dict:
    """
    ✅ Generate baseline statistics using Mean + 2*StdDev method
    
    This captures 95.4% of normal execution times (assumes normal distribution).
    Any job exceeding mean + 2*stdev is statistically anomalous.
    
    Args:
        job_durations_dict: {"job_name|joid": [duration1, duration2, ...]}
    
    Returns:
        Baselines dict with statistical threshold
    """
    baselines = {}
    
    for key, durations in job_durations_dict.items():
        if not durations or len(durations) < 2:
            baselines[key] = {
                "count": len(durations),
                "mean": durations[0] if durations else 0,
                "median": durations[0] if durations else 0,
                "stdev": 0,
                "p95": durations[0] if durations else 0,
                "mad": 0,
                "baseline_statistical": durations[0] if durations else 0,
                "min": durations[0] if durations else 0,
                "max": durations[0] if durations else 0
            }
            continue
        
        durations_array = np.array(durations, dtype=float)
        
        mean = float(np.mean(durations_array))
        median = float(np.median(durations_array))
        stdev = float(np.std(durations_array, ddof=1))
        p95 = float(np.percentile(durations_array, 95))
        mad = float(np.mean(np.abs(durations_array - mean)))
        
        # ✅ CRITICAL: Statistical baseline = mean + 2*stdev (95.4% confidence interval)
        baseline_statistical = mean + (2 * stdev)
        
        baselines[key] = {
            "count": len(durations),
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "p95": p95,
            "mad": mad,
            "baseline_statistical": baseline_statistical,  # ✅ PRIMARY THRESHOLD
            "min": float(np.min(durations_array)),
            "max": float(np.max(durations_array))
        }
    
    return baselines

def save_baselines_to_json(baselines: dict, output_path: str):
    """Save baseline statistics to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baselines, f, indent=2)

def load_baselines_from_json(input_path: str) -> dict:
    """Load baseline statistics from JSON file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Baseline file not found: {input_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in baseline file: {input_path}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_features_from_baseline(job_name: str, joid: int, 
                                  baselines: Dict[str, Dict],
                                  feature_columns: List[str]) -> Optional[List[float]]:
    """
    Extract features from baseline statistics
    
    Args:
        job_name: Job name
        joid: Job ID
        baselines: Baseline statistics dictionary
        feature_columns: List of feature column names
    
    Returns:
        Feature vector or None if baseline not found
    """
    key = f"{job_name}|{joid}"
    baseline = baselines.get(key)
    
    if not baseline:
        return None
    
    features = []
    for col in feature_columns:
        col_name = col.split("baseline_")[-1]
        value = baseline.get(col_name, baseline.get(col, 0.0))
        features.append(float(value))
    
    return features

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(model1_metrics: ModelMetrics, model2_metrics: ModelMetrics) -> Dict:
    """Compare two models based on metrics"""
    comparison = {
        'model1': model1_metrics.to_dict(),
        'model2': model2_metrics.to_dict(),
        'better_model': 'model1' if model1_metrics.rmse < model2_metrics.rmse else 'model2',
        'rmse_difference': abs(model1_metrics.rmse - model2_metrics.rmse),
        'rmse_improvement_percent': (
            (model2_metrics.rmse - model1_metrics.rmse) / model2_metrics.rmse * 100
            if model2_metrics.rmse > 0 else 0
        ),
        'faster_training': (
            'model1' if model1_metrics.training_time_sec < model2_metrics.training_time_sec 
            else 'model2'
        )
    }
    
    return comparison

# ============================================================================
# VALIDATION
# ============================================================================

def validate_features(baselines: dict, feature_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required features exist in baselines
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not baselines:
        errors.append("Baselines dictionary is empty")
        return False, errors
    
    # Check first baseline entry
    sample_key = list(baselines.keys())[0]
    sample = baselines[sample_key]
    
    for feature in feature_columns:
        col_name = feature.split("baseline_")[-1]
        if col_name not in sample:
            errors.append(f"Missing feature: {col_name}")
    
    return len(errors) == 0, errors

def print_model_metrics(metrics: ModelMetrics, model_name: str = "Model"):
    """Pretty print model metrics"""
    print(f"\n{'='*60}")
    print(f"📊 {model_name} Metrics")
    print(f"{'='*60}")
    print(f"RMSE (Root Mean Squared Error):  {metrics.rmse:.2f}s")
    print(f"MAE (Mean Absolute Error):       {metrics.mae:.2f}s")
    print(f"MAPE (Mean Absolute % Error):    {metrics.mape:.2%}")
    print(f"R² Score:                         {metrics.r2:.4f}")
    print(f"Model Size:                       {metrics.model_size_mb:.2f} MB")
    print(f"Training Time:                    {metrics.training_time_sec:.2f}s")
    print(f"{'='*60}\n")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("✅ code2.py - Model Training Module (FIXED)")
    print("\nAvailable components:")
    print("  ✅ ModelMetrics - Comprehensive metrics dataclass")
    print("  ✅ compute_metrics() - Calculate all metrics")
    print("  ✅ XGBoostTrainer - Fast XGBoost implementation")
    print("  ✅ LightGBMTrainer - Ultra-fast LightGBM")
    print("  ✅ generate_baseline_statistics() - Mean + 2*StdDev method")
    print("  ✅ FEATURE_COLUMNS - Updated with baseline_stdev and baseline_statistical")

