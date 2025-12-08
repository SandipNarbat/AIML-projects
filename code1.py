# code1.py

from pathlib import Path
from datetime import date, datetime, timedelta
import json
import threading
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, asdict
import numpy as np
import xgboost as xgb

from preprocessing import preprocess_stream_line
from live_streamer import AutosysLogStreamer, _iter_entries
import baseline_updater
import code2

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths
    "baseline_path": "baseline_stats.json",
    "model_dir": "models",
    "log_dir": "logs",
    "dlq_dir": "dlq",
    "training_data_path": "processed",
    
    "log_file_path": r"D:\AUTOSYSLOG\test\event_demon.SBI.09012025.txt",
    "streaming_speed": 1.0,
    # Ingestion controls
    #   stream      -> read historical log via AutosysLogStreamer (current behavior)
    #   tail_file   -> follow the txt file produced by live_runner in real time
    "ingestion_mode": "tail_file",
    "live_output_path": "replayed_with_system_time.txt",
    "tail_start_at_end": False,

    "retrain_interval_hours": 24,
    "retrain_sample_size": 1000,
    "min_events_for_retrain": 100,

    "rmse_threshold": 500.0,
    "drift_threshold": 0.15,
    
    "baseline_exceeded_threshold_seconds": 60,
    "baseline_exceeded_threshold_percent": 0.20,
    "alert_on_baseline_exceeded": True,
    "alert_cooldown_seconds": 300,
    "baseline_threshold_mode": "both",
    
    "baseline_method": "statistical",
    "baseline_stdev_multiplier": 2.0,
    "baseline_update_interval_hours": 24,
}

# ============================================================================
# DEAD LETTER QUEUE
# ============================================================================

class DeadLetterQueue:
    """Persists failed records for replay and investigation"""
    
    def __init__(self, dlq_dir: str):
        self.dlq_dir = Path(dlq_dir)
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_file = self.dlq_dir / f"dlq_{datetime.now().strftime('%Y%m%d')}.txt"
        self.lock = threading.Lock()
        self.added_count = 0
    
    def add(self, line: str, error: str, context: Dict = None):
        """Add failed record to DLQ"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        context_str = str(context) if context else ""
        
        entry = f"[{timestamp}] ERROR: {error}\n"
        entry += f"  Line: {line[:500]}...\n"
        if context_str:
            entry += f"  Context: {context_str}\n"
        entry += "-" * 80 + "\n"
        
        with self.lock:
            try:
                with open(self.dlq_file, 'a', encoding='utf-8') as f:
                    f.write(entry)
                self.added_count += 1
            except Exception as e:
                print(f"CRITICAL: Failed to write DLQ: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        return {
            'dlq_file': str(self.dlq_file),
            'records_added': self.added_count,
            'file_exists': self.dlq_file.exists()
        }

class StructuredLogger:
    """✅ Simplified text logging with daily file rotation (date-based filenames)."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
    
    def _format_value(self, value: Any) -> str:
        """✅ Direct string conversion - no JSON needed"""
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return str(int(value))
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return f"{float(value):.2f}"
        elif isinstance(value, np.ndarray):
            return str(value.tolist())
        else:
            return str(value)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context as key=value pairs for text output"""
        if not context:
            return ""
        
        items = []
        for key, value in context.items():
            formatted_value = self._format_value(value)
            items.append(f"{key}={formatted_value}")
        
        return " | ".join(items)
    
    def _dated_path(self, stem: str, ext: str = ".txt") -> Path:
        """Return a date-suffixed file path (rotates daily by filename)."""
        day = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{stem}_{day}{ext}"
    
    def log_event(self, level: str, module: str, message: str, context: Dict[str, Any] = None):
        """Log event to text file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        context_str = self._format_context(context) if context else ""
        
        log_line = f"[{timestamp}] {level.ljust(7)} [{module.ljust(20)}] {message}"
        if context_str:
            log_line += f" | {context_str}"
        
        with self.lock:
            try:
                with open(self._dated_path("events"), 'a', encoding='utf-8') as f:
                    f.write(log_line + '\n')
            except Exception as e:
                print(f"ERROR: Failed to write log: {e}")
    
    def log_retraining(self, status: str, metrics: Dict[str, Any]):
        """Log retraining event"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metrics_str = []
        for key, value in metrics.items():
            formatted = self._format_value(value)
            metrics_str.append(f"  {key.ljust(25)}: {formatted}")
        
        log_line = f"[{timestamp}] RETRAINING {status.upper()}\n"
        log_line += "\n".join(metrics_str)
        
        with self.lock:
            try:
                with open(self._dated_path("retraining"), 'a', encoding='utf-8') as f:
                    f.write(log_line + '\n' + '='*80 + '\n')
            except Exception as e:
                print(f"ERROR: Failed to write retraining log: {e}")
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metric_parts = [f"{k}={self._format_value(v)}" for k, v in metrics.items()]
        log_line = f"[{timestamp}] " + " | ".join(metric_parts)
        
        with self.lock:
            try:
                with open(self._dated_path("performance"), 'a', encoding='utf-8') as f:
                    f.write(log_line + '\n')
            except Exception as e:
                print(f"ERROR: Failed to write performance log: {e}")
    
    def info(self, module: str, message: str, context: Dict = None):
        self.log_event('INFO', module, message, context)
    
    def warning(self, module: str, message: str, context: Dict = None):
        self.log_event('WARNING', module, message, context)
    
    def error(self, module: str, message: str, context: Dict = None):
        self.log_event('ERROR', module, message, context)
    
    def debug(self, module: str, message: str, context: Dict = None):
        self.log_event('DEBUG', module, message, context)


# Initialize components
logger = StructuredLogger(CONFIG["log_dir"])
dlq = DeadLetterQueue(CONFIG.get("dlq_dir", "dlq"))

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelVersion:
    """✅ Model versioning with validation"""
    
    def __init__(self, model_id: str, booster: xgb.Booster, meta: Dict, 
                 rmse: float, created_at: datetime):
        self.model_id = model_id
        self.booster = booster
        self.meta = meta
        self.rmse = rmse
        self.created_at = created_at
        self.predictions_count = 0
        self.alerts_triggered = 0
    
    def save(self, model_dir: str):
        """Save model and metadata"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{self.model_id}_model.json"
        meta_path = model_dir / f"{self.model_id}_meta.json"
        
        self.booster.save_model(str(model_path))
        
        metadata = {
            **self.meta,
            'rmse': float(self.rmse),
            'created_at': self.created_at.isoformat(),
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path, meta_path
    
    @staticmethod
    def load(model_dir: str, model_id: str):
        """✅ Load model with validation"""
        model_dir = Path(model_dir)
        
        model_path = model_dir / f"{model_id}_model.json"
        meta_path = model_dir / f"{model_id}_meta.json"
        
        if not model_path.exists() or not meta_path.exists():
            return None
        
        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            rmse = meta.pop('rmse', None)
            created_at_str = meta.pop('created_at', datetime.now().isoformat())
            created_at = datetime.fromisoformat(created_at_str)
            
            loaded_features = meta.get('feature_columns', [])
            if loaded_features != code2.FEATURE_COLUMNS:
                logger.warning('ModelVersion', 
                             f'Feature mismatch when loading {model_id}',
                             {'expected': code2.FEATURE_COLUMNS,
                              'got': loaded_features})
            
            return ModelVersion(model_id, booster, meta, rmse, created_at)
        
        except Exception as e:
            logger.error('ModelVersion', f'Failed to load {model_id}: {str(e)}')
            return None

class ModelManager:
    """Manages active model, versioning, and A/B testing"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.active_model: Optional[ModelVersion] = None
        self.candidate_model: Optional[ModelVersion] = None
        self.lock = threading.Lock()
        self.model_registry = {}
    
    def set_active_model(self, model: ModelVersion):
        """Set as active production model"""
        with self.lock:
            self.active_model = model
            self.model_registry[model.model_id] = model
            logger.info('ModelManager', f'Model {model.model_id} set as active', 
                       {'rmse': model.rmse})
    
    def set_candidate_model(self, model: ModelVersion):
        """Set candidate for evaluation"""
        with self.lock:
            self.candidate_model = model
            self.model_registry[model.model_id] = model
    
    def promote_candidate(self) -> bool:
        """Promote candidate to active if better than current"""
        with self.lock:
            if not self.candidate_model:
                return False
            
            if not self.active_model or self.candidate_model.rmse < self.active_model.rmse:
                old_active = self.active_model
                self.active_model = self.candidate_model
                self.candidate_model = None
                
                logger.info('ModelManager', 'Candidate promoted to active',
                           {'old_model': old_active.model_id if old_active else None,
                            'new_model': self.active_model.model_id,
                            'old_rmse': old_active.rmse if old_active else None,
                            'new_rmse': self.active_model.rmse})
                return True
            
            return False
    
    def predict(self, job_name: str, joid: int, baselines: Dict[str, Dict[str, Any]]) -> Optional[float]:
        """✅ Make prediction with performance logging"""
        with self.lock:
            if not self.active_model:
                logger.warning('ModelManager', 'No active model available')
                return None
            
            try:
                key = f"{job_name}|{joid}"
                baseline = baselines.get(key)
                
                if not baseline:
                    logger.warning('ModelManager', f'No baseline found for {key}')
                    return None
                
                feature_columns = self.active_model.meta.get('feature_columns', [])
                values = []
                for col in feature_columns:
                    col_name = col.split("baseline_")[-1]
                    values.append(float(baseline.get(col_name, baseline.get(col, 0.0))))
                
                x = np.asarray([values], dtype=float)
                dmatrix = xgb.DMatrix(x, feature_names=feature_columns)
                pred = float(self.active_model.booster.predict(dmatrix)[0])
                
                self.active_model.predictions_count += 1
                
                logger.log_performance({
                    'type': 'prediction',
                    'job_name': job_name,
                    'joid': joid,
                    'predicted_duration': pred,
                    'model_id': self.active_model.model_id,
                    'prediction_number': self.active_model.predictions_count
                })
                
                return pred
            
            except Exception as e:
                logger.error('ModelManager', f'Prediction failed: {str(e)}', 
                           {'job_name': job_name, 'joid': joid})
                return None

# ============================================================================
# STATE MANAGEMENT - ✅ CRITICAL FIX: Use Log Timestamps
# ============================================================================

@dataclass
class JobInstance:
    """✅ Tracks a running job instance with log-based timestamps"""

    joid: str
    runid: str
    job_name: str

    # ✅ CRITICAL FIX: Store log timestamp for accurate duration calculation
    start_time_log: datetime         # Timestamp from AutoSys log (ts field)
    start_time_real: datetime        # Real-time clock (when START log received)

    predicted_duration: Optional[float]
    baseline_seconds: Optional[float]

    alerted: bool = False
    last_alert_time: Optional[datetime] = None

    def duration_so_far_real(self) -> float:
        """Real-time duration (for immediate alerting)"""
        return (datetime.now() - self.start_time_real).total_seconds()

    def true_duration_from_log(self, end_time_log: datetime) -> float:
        """✅ CRITICAL FIX: Calculate TRUE duration using log timestamps only"""
        return (end_time_log - self.start_time_log).total_seconds()

    def exceeded_baseline(self, threshold_seconds: float, threshold_percent: float, mode: str = "either") -> bool:
        """Uses REAL-TIME duration to detect overrun immediately"""
        threshold = self.predicted_duration or self.baseline_seconds
        if not threshold or threshold <= 0:
            return False

        current_duration = self.duration_so_far_real()

        if current_duration <= threshold:
            return False

        exceeded_by_seconds = current_duration - threshold
        exceeded_by_percent = exceeded_by_seconds / threshold

        if mode == "both":
            return (exceeded_by_seconds >= threshold_seconds and exceeded_by_percent >= threshold_percent)
        else:
            return (exceeded_by_seconds >= threshold_seconds or exceeded_by_percent >= threshold_percent)

    def get_exceeded_amount(self) -> dict:
        """Amount exceeded (REAL TIME)"""
        current_duration = self.duration_so_far_real()
        if not self.predicted_duration:
            return {"seconds": 0, "percent": 0}

        exceeded_seconds = max(0, current_duration - self.predicted_duration)
        exceeded_percent = exceeded_seconds / self.predicted_duration if self.predicted_duration > 0 else 0

        return {
            "seconds": round(exceeded_seconds, 2),
            "percent": round(exceeded_percent * 100, 2)
        }

class StateManager:
    """Manages running job state"""
    
    def __init__(self, alert_cooldown: int = 300):
        self.running_jobs: Dict[Tuple[str, str], JobInstance] = {}
        self.lock = threading.Lock()
        self.alert_cooldown = alert_cooldown
    
    def start_job(self, joid: str, runid: str, job_name: str,
                  predicted_duration: float, baseline_seconds: float,
                  log_ts: datetime):
        """✅ Start tracking job with log timestamp for accuracy"""

        key = (runid, joid)

        with self.lock:
            self.running_jobs[key] = JobInstance(
                joid=str(joid),
                runid=str(runid),
                job_name=str(job_name),

                start_time_log=log_ts,           # ✅ CRITICAL: Use log timestamp
                start_time_real=datetime.now(),  # Real-time for alerting

                predicted_duration=float(predicted_duration or 0),
                baseline_seconds=float(baseline_seconds or 0),
            )
    
    def end_job(self, joid: str, runid: str) -> Optional[JobInstance]:
        """Remove job and return its instance"""
        key = (runid, joid)
        with self.lock:
            return self.running_jobs.pop(key, None)
    
    def get_active_jobs(self) -> list:
        """Get snapshot of active jobs"""
        with self.lock:
            return list(self.running_jobs.values())
    
    def can_alert_for_job(self, key: Tuple[str, str]) -> bool:
        """Check if job can be alerted (respects cooldown)"""
        with self.lock:
            job = self.running_jobs.get(key)
            if not job:
                return False
            
            if job.alerted and job.last_alert_time:
                if (datetime.now() - job.last_alert_time).total_seconds() < self.alert_cooldown:
                    return False
            
            job.alerted = True
            job.last_alert_time = datetime.now()
            return True

# Global state
model_manager = ModelManager(CONFIG["model_dir"])
state_manager = StateManager(alert_cooldown=CONFIG["alert_cooldown_seconds"])
event_buffer = []
buffer_lock = threading.Lock()
# Daily durations buffer for baseline auto-updates
daily_durations: Dict[str, list] = {}
daily_durations_lock = threading.Lock()

# ============================================================================
# MAIN PROCESSING - ✅ CRITICAL FIX: Use Log Timestamps
# ============================================================================

def handle_stream_output(line_text: str):
    """✅ CRITICAL FIX: Calculate duration using log timestamps, not system clock"""
    try:
        rec = preprocess_stream_line(line_text)
        if not rec:
            return
        
        # Validate required fields
        required_fields = ["JOID", "RUNID", "JobName", "Status", "timestamp"]
        missing = [f for f in required_fields if f not in rec]
        if missing:
            logger.warning('StreamHandler', 'Missing required fields', 
                         {'missing': missing})
            dlq.add(line_text, 'Missing required fields', {'missing': missing})
            return
        
        # Type conversion and validation
        try:
            joid = int(rec["JOID"]) if rec["JOID"] else None
            runid = int(rec["RUNID"]) if rec["RUNID"] else None
            job_name = str(rec["JobName"]).strip()
            status = str(rec["Status"]).strip().upper()
            ts = rec["timestamp"]  # ✅ CRITICAL: This is the log timestamp from "ts" field
        except (ValueError, TypeError) as e:
            logger.warning('StreamHandler', f'Type conversion error: {str(e)}')
            dlq.add(line_text, f'Type conversion error: {str(e)}', {'record': rec})
            return
        
        ts_str = ts.isoformat() if ts else None
        baselines = load_baselines(CONFIG["baseline_path"])
        
        if status == "STARTING":
            try:
                pred_duration = model_manager.predict(job_name, joid, baselines)
                if pred_duration is None:
                    pred_duration = 0
                
                # Use statistical baseline (mean + 2*stdev)
                baseline_key = f"{job_name}|{joid}"
                baseline_stats = baselines.get(baseline_key, {})
                baseline_seconds = baseline_stats.get("baseline_statistical") or \
                                 baseline_stats.get("median", 0)
                
                # ✅ CRITICAL FIX: Pass log timestamp for accurate tracking
                state_manager.start_job(
                    joid=str(joid),
                    runid=str(runid),
                    job_name=job_name,
                    predicted_duration=pred_duration,
                    baseline_seconds=baseline_seconds,
                    log_ts=ts  # ✅ CRITICAL: Use log timestamp, NOT datetime.now()
                )
                
                logger.info('StreamHandler', 'Job started',
                           {'ts': ts_str,
                            'joid': joid, 
                            'job_name': job_name, 
                            'pred_duration': round(pred_duration or 0, 2), 
                            'baseline_seconds': round(baseline_seconds, 2),
                            'baseline_method': 'mean+2*stdev',
                            'runid': runid})
            
            except Exception as e:
                logger.error('StreamHandler', f'Error handling STARTING: {str(e)}',
                            {'joid': joid, 'job_name': job_name})
                dlq.add(line_text, str(e), {'status': 'STARTING', 'joid': joid})
        
        elif status == "SUCCESS":
            try:
                job_instance = state_manager.end_job(str(joid), str(runid))
                if job_instance:
                    # ✅ CRITICAL FIX: Calculate duration using log timestamps ONLY
                    success_ts = ts  # Log timestamp from "ts" field
                    actual_duration = job_instance.true_duration_from_log(success_ts)
                    
                    # Atomic buffer operation
                    with buffer_lock:
                        event_buffer.append({
                            'joid': str(joid),
                            'runid': str(runid),
                            'job_name': job_name,
                            'actual_duration': actual_duration,  # ✅ TRUE measurement from log
                            'predicted_duration': job_instance.predicted_duration,
                            'timestamp': success_ts.isoformat()
                        })
                    
                    # Track durations for automatic baseline updates
                    key = f"{job_name}|{joid}"
                    with daily_durations_lock:
                        daily_durations.setdefault(key, []).append(actual_duration)
                    
                    logger.info('StreamHandler', 'Job succeeded',
                               {'ts': ts_str,
                                'joid': joid, 
                                'job_name': job_name,
                                'actual_duration': round(actual_duration, 2),
                                'predicted_duration': round(job_instance.predicted_duration or 0, 2),
                                'runid': runid})
            except Exception as e:
                logger.error('StreamHandler', f'Error handling SUCCESS: {str(e)}')
                dlq.add(line_text, str(e), {'status': 'SUCCESS'})
        
        elif status == "FAILURE":
            try:
                joid_str = str(joid) if joid else None
                runid_str = str(runid) if runid else None
                
                job_instance = state_manager.end_job(joid_str, runid_str) if joid_str and runid_str else None
                
                if job_instance:
                    # ✅ CRITICAL FIX: Use log timestamp for true duration
                    failure_ts = ts
                    actual_duration = job_instance.true_duration_from_log(failure_ts)
                    
                    with buffer_lock:
                        event_buffer.append({
                            'joid': joid_str,
                            'runid': runid_str,
                            'job_name': str(job_name),
                            'actual_duration': actual_duration,
                            'predicted_duration': job_instance.predicted_duration or 0.0,
                            'status': 'FAILURE',
                            'timestamp': failure_ts.isoformat()
                        })
                    
                    # Track durations for automatic baseline updates
                    key = f"{job_name}|{joid_str}"
                    with daily_durations_lock:
                        daily_durations.setdefault(key, []).append(actual_duration)
                    
                    logger.warning('StreamHandler', 'Job failed (tracked)', {
                        'ts': ts_str,
                        'joid': int(joid) if joid else None,
                        'job_name': str(job_name),
                        'duration': round(actual_duration, 2),
                        'predicted_duration': round(job_instance.predicted_duration or 0, 2),
                        'runid': int(runid) if runid else None
                    })
                else:
                    logger.warning('StreamHandler', 'Job failed (untracked)', {
                        'ts': ts_str,
                        'joid': int(joid) if joid else None,
                        'job_name': str(job_name),
                        'runid': int(runid) if runid else None
                    })
            except Exception as e:
                logger.error('StreamHandler', f'Error handling FAILURE: {str(e)}')
                dlq.add(line_text, str(e), {'status': 'FAILURE'})
    
    except Exception as e:
        logger.error('StreamHandler', f'Unexpected error: {str(e)}',
                    {'line_preview': line_text[:200]})
        dlq.add(line_text, str(e), {'type': 'unexpected_error'})

# ============================================================================
# MONITORING & RETRAINING
# ============================================================================

def monitor_baselines():
    """Monitor running jobs and alert on significant baseline overrun"""
    logger.info('BaselineMonitor', 'Baseline monitoring thread started')
    
    threshold_seconds = CONFIG.get("baseline_exceeded_threshold_seconds", 60)
    threshold_percent = CONFIG.get("baseline_exceeded_threshold_percent", 0.20)
    threshold_mode = CONFIG.get("baseline_threshold_mode", "either")
    
    logger.info('BaselineMonitor', 'Alert thresholds configured', {
        'threshold_seconds': threshold_seconds,
        'threshold_percent': f"{threshold_percent * 100}%",
        'mode': threshold_mode
    })
    
    while True:
        try:
            active_jobs = state_manager.get_active_jobs()
            
            for job in active_jobs:
                key = (job.runid, job.joid)
                
                if CONFIG["alert_on_baseline_exceeded"]:
                    if job.exceeded_baseline(
                        threshold_seconds=threshold_seconds,
                        threshold_percent=threshold_percent,
                        mode=threshold_mode
                    ):
                        if state_manager.can_alert_for_job(key):
                            exceeded_info = job.get_exceeded_amount()
                            
                            logger.warning('BaselineMonitor', 'Baseline exceeded', {
                                'joid': job.joid,
                                'runid': job.runid,
                                'job_name': job.job_name,
                                'baseline_seconds': round(job.predicted_duration, 2),
                                'current_duration': round(job.duration_so_far_real(), 2),
                                'exceeded_by': exceeded_info['seconds'],
                                'exceeded_by_percent': f"{exceeded_info['percent']}%"
                            })
                            
                            print(f"⚠️ ALERT: {job.job_name} exceeded prediction by {exceeded_info['seconds']}s ({exceeded_info['percent']}%)")
            
            time.sleep(5)
        
        except Exception as e:
            logger.error('BaselineMonitor', f'Monitor error: {str(e)}')
            time.sleep(5)

def retrain_model_periodic():
    """✅ Atomic buffer operations, metrics tracking"""
    logger.info('Retrainer', 'Model retraining thread started')
    
    last_retrain = datetime.now()
    
    while True:
        try:
            time.sleep(60)
            
            if (datetime.now() - last_retrain).total_seconds() < CONFIG["retrain_interval_hours"] * 3600:
                continue
            
            # Atomic operation - check AND clear together
            with buffer_lock:
                if len(event_buffer) < CONFIG["min_events_for_retrain"]:
                    logger.debug('Retrainer', 
                               f'Waiting for events: {len(event_buffer)}/{CONFIG["min_events_for_retrain"]}')
                    continue
                
                recent_events = event_buffer[-CONFIG["retrain_sample_size"]:]
                events_to_process = list(recent_events)
                event_buffer.clear()
            
            logger.info('Retrainer', f'Starting retraining with {len(events_to_process)} events')
            
            try:
                trained_model, metrics = train_new_model(events_to_process)
                
                if trained_model is None:
                    logger.error('Retrainer', 'Model training failed')
                    continue
                
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                candidate = ModelVersion(
                    model_id=model_id,
                    booster=trained_model,
                    meta={'feature_columns': code2.FEATURE_COLUMNS},
                    rmse=metrics.rmse,
                    created_at=datetime.now()
                )
                
                candidate.save(CONFIG["model_dir"])
                model_manager.set_candidate_model(candidate)
                promoted = model_manager.promote_candidate()
                
                logger.info('Retrainer', 'Retraining completed',
                           {'model_id': model_id, 'rmse': round(metrics.rmse, 2),
                            'promoted': promoted, 'events_used': len(events_to_process)})
                
                logger.log_retraining('success', {
                    'model_id': model_id,
                    'rmse': metrics.rmse,
                    'mae': metrics.mae,
                    'mape': metrics.mape,
                    'r2': metrics.r2,
                    'training_time': metrics.training_time_sec,
                    'promoted': promoted,
                    'events_used': len(events_to_process)
                })
                
                last_retrain = datetime.now()
            
            except Exception as e:
                logger.error('Retrainer', f'Retraining failed: {str(e)}')
                logger.log_retraining('failed', {'error': str(e)})
        
        except Exception as e:
            logger.error('Retrainer', f'Unexpected error: {str(e)}')
            time.sleep(5)

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_new_model(events: list) -> Tuple[Optional[xgb.Booster], Optional[code2.ModelMetrics]]:
    """✅ Return metrics object instead of just RMSE"""
    if not events or len(events) < CONFIG["min_events_for_retrain"]:
        return None, None
    
    try:
        X = []
        y = []
        baselines = load_baselines(CONFIG["baseline_path"])
        
        for event in events:
            job_name = event['job_name']
            joid = event['joid']
            actual_duration = event['actual_duration']
            
            key = f"{job_name}|{joid}"
            baseline = baselines.get(key)
            
            if not baseline:
                continue
            
            feature_columns = code2.FEATURE_COLUMNS
            features = []
            for col in feature_columns:
                col_name = col.split("baseline_")[-1]
                features.append(float(baseline.get(col_name, baseline.get(col, 0.0))))
            
            X.append(features)
            y.append(actual_duration)
        
        if len(X) < CONFIG["min_events_for_retrain"]:
            logger.warning('ModelTraining', f'Insufficient training samples: {len(X)}')
            return None, None
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'tree_method': 'hist',
            'device': 'cpu',
            'nthread': 1  # Single-threaded for thread safety
        }
        
        model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
        
        preds = model.predict(dtrain)
        mse = np.mean((preds - y) ** 2)
        rmse = np.sqrt(mse)
        
        metrics = code2.compute_metrics(y, preds, training_time=0.0, model_size=0.0)
        
        return model, metrics
    
    except Exception as e:
        logger.error('ModelTraining', f'Training error: {str(e)}')
        return None, None

def load_baselines(path: str) -> Dict[str, Dict[str, Any]]:
    """Load baseline statistics"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error('Baselines', f'Baseline file not found: {path}')
        return {}
    except json.JSONDecodeError:
        logger.error('Baselines', f'Invalid JSON in baseline file: {path}')
        return {}

# ============================================================================
# CLOCK-SYNCHRONIZED STREAMING
# ============================================================================

def stream_with_clock_sync(
    streamer: AutosysLogStreamer,
    emit: Callable[[str], None],
) -> None:
    """
    ✅ Clock-synchronized streaming that respects log timestamps
    This replays logs maintaining temporal relationships between events
    """
    log_path = streamer.log_path
    speed = streamer.speed

    # Read all entries once (file is finite historic log)
    entries = list(_iter_entries(log_path))
    if not entries:
        logger.warning('Streamer', 'No entries found in log file')
        return

    # Find first entry that has a timestamp
    first_ts: Optional[datetime] = None
    for e in entries:
        if e.timestamp is not None:
            first_ts = e.timestamp
            break

    if first_ts is None:
        # No timestamps in file → just emit without timing control
        logger.warning('Streamer', 'No timestamps found, streaming without clock sync')
        for e in entries:
            if streamer._should_emit(e):
                emit(e.text)
        return

    start_wall = datetime.now()
    logger.info('Streamer', 'Starting clock-synchronized streaming', {
        'first_log_timestamp': first_ts.isoformat(),
        'speed_multiplier': speed,
        'total_entries': len(entries)
    })

    entries_emitted = 0
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
        entries_emitted += 1
        
        # Progress logging every 100 entries
        if entries_emitted % 100 == 0:
            logger.info('Streamer', f'Streamed {entries_emitted}/{len(entries)} entries')
    
    logger.info('Streamer', 'Streaming completed', {
        'entries_emitted': entries_emitted,
        'total_entries': len(entries)
    })


def stream_from_txt_file(
    txt_path: Path,
    emit: Callable[[str], None],
    *,
    poll_interval: Optional[float] = None,
    start_at_end: bool = False,
) -> None:
    """
    Follow a growing txt file (written by live_runner) and emit each new line.
    This decouples streaming/writing (live_runner) from processing (code1).
    """
    path = Path(txt_path)
    logger.info('Streamer', 'Starting tail-file stream', {
        'txt_path': str(path),
        'poll_interval': poll_interval,
        'start_at_end': start_at_end,
    })

    # Wait for writer to create the file
    while not path.exists():
        logger.warning('Streamer', 'Waiting for txt stream file to appear', {'path': str(path)})
        time.sleep(1)

    with path.open("r", encoding="utf-8") as handle:
        if start_at_end:
            handle.seek(0, 2)  # jump to end so we only read new data

        while True:
            line = handle.readline()
            if not line:
                if poll_interval is not None:
                    time.sleep(poll_interval)
                continue
            emit(line.rstrip("\n"))

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """✅ Validate critical configuration values"""
    errors = []
    
    # Type checks
    if not isinstance(CONFIG["streaming_speed"], (int, float)) or CONFIG["streaming_speed"] <= 0:
        errors.append("streaming_speed must be positive number")

    ingestion_mode = CONFIG.get("ingestion_mode", "stream")
    if ingestion_mode not in ("stream", "tail_file"):
        errors.append("ingestion_mode must be 'stream' or 'tail_file'")
    
    if CONFIG.get("baseline_update_interval_hours", 24) <= 0:
        errors.append("baseline_update_interval_hours must be > 0")
    
    if CONFIG["baseline_exceeded_threshold_percent"] < 0 or CONFIG["baseline_exceeded_threshold_percent"] > 1:
        errors.append("baseline_exceeded_threshold_percent must be 0-1")
    
    if CONFIG["baseline_threshold_mode"] not in ["either", "both"]:
        errors.append("baseline_threshold_mode must be 'either' or 'both'")
    
    # File existence checks
    if not Path(CONFIG["baseline_path"]).exists():
        errors.append(f"baseline_path not found: {CONFIG['baseline_path']}")
    
    if ingestion_mode == "stream":
        if not Path(CONFIG["log_file_path"]).exists():
            errors.append(f"log_file_path not found: {CONFIG['log_file_path']}")
    else:
        live_out = CONFIG.get("live_output_path")
        if not live_out:
            errors.append("live_output_path must be set when ingestion_mode='tail_file'")
        else:
            # Ensure directory exists so live_runner can create the file
            Path(live_out).parent.mkdir(parents=True, exist_ok=True)
    
    if errors:
        for error in errors:
            logger.error('Config', error)
        raise ValueError(f"Configuration invalid: {errors}")
    
    logger.info('Config', 'Configuration validated successfully')

# ============================================================================
# INITIALIZATION & STARTUP
# ============================================================================

def initialize_system():
    """Initialize system and load or create initial model"""
    logger.info('System', 'Initializing anomaly detection system')
    
    Path(CONFIG["model_dir"]).mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(CONFIG["model_dir"])
    models = sorted(model_dir.glob("*_model.json"), reverse=True)
    
    if models:
        latest_model_id = models[0].stem.replace("_model", "")
        model = ModelVersion.load(CONFIG["model_dir"], latest_model_id)
        if model:
            model_manager.set_active_model(model)
            logger.info('System', f'Loaded model {latest_model_id}', 
                       {'rmse': model.rmse})
    
    if not model_manager.active_model:
        logger.warning('System', 'No initial model found')

    # Ensure baseline file exists
    baseline_path = Path(CONFIG["baseline_path"])
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if not baseline_path.exists():
        baseline_path.write_text("{}", encoding="utf-8")


def baseline_update_loop():
    """Periodically update baseline_stats.json with collected durations."""
    interval_hours = CONFIG.get("baseline_update_interval_hours", 24)
    interval_seconds = max(1, int(interval_hours * 3600))
    logger.info('BaselineUpdater', 'Starting baseline update loop', {
        'interval_hours': interval_hours
    })

    while True:
        time.sleep(interval_seconds)
        try:
            with daily_durations_lock:
                snapshot = {k: v[:] for k, v in daily_durations.items()}
                daily_durations.clear()

            if not snapshot:
                logger.info('BaselineUpdater', 'No durations collected this interval')
                continue

            baseline_updater.update_baselines(
                baseline_path=Path(CONFIG["baseline_path"]),
                durations_by_job=snapshot,
                stdev_multiplier=CONFIG.get("baseline_stdev_multiplier", 2.0),
            )
            logger.info('BaselineUpdater', 'Baseline stats updated', {
                'jobs_updated': len(snapshot),
                'total_durations': sum(len(v) for v in snapshot.values()),
            })
        except Exception as e:
            logger.error('BaselineUpdater', f'Baseline update failed: {str(e)}')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point with clock-synchronized streaming"""
    try:
        logger.info('Main', 'Starting anomaly detection system')
        
        validate_config()
        
        initialize_system()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_baselines, daemon=True)
        monitor_thread.start()
        logger.info('Main', 'Started baseline monitor thread')
        
        # Start retraining thread
        retrain_thread = threading.Thread(target=retrain_model_periodic, daemon=True)
        retrain_thread.start()
        logger.info('Main', 'Started retraining thread')

        # Start baseline update thread (daily by default)
        baseline_thread = threading.Thread(target=baseline_update_loop, daemon=True)
        baseline_thread.start()
        logger.info('Main', 'Started baseline updater thread')
        
        ingestion_mode = CONFIG.get("ingestion_mode", "stream")
        if ingestion_mode == "stream":
            # Create streamer (historic log replay)
            streamer = AutosysLogStreamer(
                log_path=Path(CONFIG["log_file_path"]),
                speed=CONFIG["streaming_speed"]
            )
            
            logger.info('Main', 'Starting clock-synchronized log stream', 
                       {'log_path': CONFIG["log_file_path"],
                        'speed': CONFIG["streaming_speed"],
                        'method': 'clock_sync'})
            
            # ✅ CRITICAL FIX: Use clock-synchronized streaming
            stream_with_clock_sync(streamer, emit=handle_stream_output)

        elif ingestion_mode == "tail_file":
            logger.info('Main', 'Starting tail-file ingestion', {
                'txt_path': CONFIG["live_output_path"],
                'start_at_end': CONFIG.get("tail_start_at_end", False),
            })
            stream_from_txt_file(
                txt_path=Path(CONFIG["live_output_path"]),
                emit=handle_stream_output,
                start_at_end=bool(CONFIG.get("tail_start_at_end", False)),
            )
        else:
            raise ValueError(f"Unsupported ingestion_mode: {ingestion_mode}")
        
        logger.info('Main', 'Streaming completed normally')
    
    except KeyboardInterrupt:
        logger.info('Main', 'System shutdown requested by user')
    except Exception as e:
        logger.error('Main', f'System error: {str(e)}')
        import traceback
        logger.error('Main', traceback.format_exc())
    finally:
        logger.info('Main', 'Anomaly detection system stopped')
        print(f"\n📊 Final Statistics:")
        print(f"   DLQ Records: {dlq.get_stats()['records_added']}")
        if model_manager.active_model:
            print(f"   Active Model: {model_manager.active_model.model_id}")
            print(f"   Predictions Made: {model_manager.active_model.predictions_count}")
            print(f"   Alerts Triggered: {model_manager.active_model.alerts_triggered}")

if __name__ == "__main__":
    main()
    