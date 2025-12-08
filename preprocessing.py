
import re
from datetime import datetime

# Regex patterns
TS_RE = re.compile(r"\[(?P<ts>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\]")
STATUS_RE = re.compile(r"\bSTATUS:\s*(?P<status>\w+)\b", re.IGNORECASE)
JOB_RE = re.compile(
    r"\bJOB:\s*(?P<job>.+?)(?=\s+(?:MACHINE:|JOID:|EXITCODE:|RUNID:|NTRY:|\Z))",
    re.IGNORECASE,
)
MACHINE_RE = re.compile(r"\bMACHINE:\s*(?P<machine>[^\s:]+)", re.IGNORECASE)
JOID_RE = re.compile(r"\bJOID:\s*(?P<joid>\d+)\b", re.IGNORECASE)
RUNID_RE = re.compile(r"\bRUNID:\s*(?P<runid>\d+)\b", re.IGNORECASE)


def parse_ts(ts_str):
    """Parse timestamp from AutoSys log format"""
    try:
        return datetime.strptime(ts_str, "%m/%d/%Y %H:%M:%S")
    except:
        return None


# State holders
first_event = {}
starting_done = set()


def preprocess_stream_line(line: str):
    
    if len(line.strip()) < 3:
        return None

    ts_m = TS_RE.search(line)
    status_m = STATUS_RE.search(line)
    job_m = JOB_RE.search(line)
    machine_m = MACHINE_RE.search(line)
    joid_m = JOID_RE.search(line)
    runid_m = RUNID_RE.search(line)

    # ✅ CRITICAL: Extract timestamp from log (the "ts" field)
    ts = parse_ts(ts_m.group("ts")) if ts_m else None
    status = status_m.group("status").upper() if status_m else None
    job = job_m.group("job").strip() if job_m else None
    machine = machine_m.group("machine").strip() if machine_m else ""
    joid = int(joid_m.group("joid")) if joid_m else None
    runid = int(runid_m.group("runid")) if runid_m else None

    rec = {
        "JOID": joid,
        "JobName": job,
        "Machine": machine,
        "RUNID": runid,
        "Status": status,
        "timestamp": ts,  # ✅ CRITICAL: This is the log timestamp, NOT system time
    }
    
    key = (joid, runid)
    
    # Only consider valid Autosys lifecycle events
    if status not in ("STARTING", "RUNNING", "SUCCESS", "FAILURE"):
        return None
    
    status_clean = status
    
    # If JOID missing → cannot reconstruct lifecycle → return as-is
    if key is None or joid is None:
        return rec
    
    # First event handling
    if key not in first_event:
        first_event[key] = status_clean
        
        if status_clean == "STARTING":
            starting_done.add(key)
            return rec
        
        if status_clean == "RUNNING":
            if key not in starting_done:
                # Inject fake STARTING with same timestamp
                fake = dict(rec)
                fake["Status"] = "STARTING"
                starting_done.add(key)
                return fake
            return rec
        
        # ✅ FIXED: Always return FAILURE events
        if status_clean == "FAILURE":
            return rec
        
        return rec
    
    # Prevent duplicate STARTING
    if status_clean == "STARTING" and key in starting_done:
        return None
    
    if status_clean == "STARTING":
        starting_done.add(key)
        return rec
    
    # ✅ CRITICAL: Always return SUCCESS and FAILURE events
    if status_clean in ("SUCCESS", "FAILURE"):
        return rec
    
    return rec