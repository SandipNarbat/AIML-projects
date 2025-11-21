# autosys_baseline.py
import os, re, json
from datetime import datetime
from collections import defaultdict
import statistics

INPUT_PATH = r"C:\Users\V1022819\SBI AI\AUTOSYSLOG"  
OUT_BASELINE = "baseline_stats.json"

TS_RE = re.compile(r"\[(?P<ts>[^\]]+)\]")
STATUS_RE = re.compile(r"\bSTATUS:\s*(?P<status>\w+)\b", re.IGNORECASE)
JOB_RE = re.compile(r"\bJOB:\s*(?P<job>.+?)(?=\s+(?:MACHINE:|JOID:|EXITCODE:|RUNID:|NTRY:|\Z))", re.IGNORECASE)
MACHINE_RE = re.compile(r"\bMACHINE:\s*(?P<machine>[^\s:]+)", re.IGNORECASE)
JOID_RE = re.compile(r"\bJOID:\s*(?P<joid>\d+)\b", re.IGNORECASE)
RUNID_RE = re.compile(r"\bRUNID:\s*(?P<runid>\d+)\b", re.IGNORECASE)

RUNNING_EQUIVALENTS = {"RUNNING","LONG_RUNNING","RUN"}

def parse_ts(ts_str):

    try:
        return datetime.strptime(ts_str.strip(), "%m/%d/%Y %H:%M:%S")
    except Exception:
        return None

def preprocess_stream(input_path):
    files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(".txt")])
    if not files:
        raise FileNotFoundError(f"No .txt logs in {input_path!r}")

    for fn in files:
        fp = os.path.join(input_path, fn)
        print(f"[stream] reading {fn}")
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(line.strip()) < 5:
                    continue
                ts_m = TS_RE.search(line)
                ts = parse_ts(ts_m.group("ts")) if ts_m else None
                status_m = STATUS_RE.search(line)
                job_m = JOB_RE.search(line)
                machine_m = MACHINE_RE.search(line)
                joid_m = JOID_RE.search(line)
                runid_m = RUNID_RE.search(line)

                status = status_m.group("status").upper() if status_m else None
                # normalize some variants
                if status and status in RUNNING_EQUIVALENTS:
                    status = "RUNNING"

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
                    "ts": ts
                }
                yield rec

def build_runs(events):

    runs = {}
    completed = []

    for ev in events:
        joid = ev["JOID"]
        runid = ev["RUNID"]
        status = ev["Status"]
        ts = ev["ts"]
        job = ev["JobName"] or f"JOID_{joid}"
        machine = ev["Machine"] or ""

        # if we don't have joid/runid, we still track by job name + ts fallback
        if joid is None or runid is None:
            # not enough info to tie to run; skip or log
            continue

        key = (joid, runid)
        if key not in runs:
            runs[key] = {
                "JOID": joid, "RUNID": runid, "JobName": job,
                "Machine": machine, "start_ts": None, "running_ts": None,
                "end_ts": None, "end_status": None
            }

        r = runs[key]

        if status == "STARTING":
            # record starting timestamp if not present
            if r["start_ts"] is None:
                r["start_ts"] = ts
        elif status == "RUNNING":
            # if no running_ts set, set it; if start_ts missing, we may create synthetic start
            if r["running_ts"] is None:
                # If we have a previous start_ts, use it; otherwise use RUNNING ts as both start+running
                if r["start_ts"] is None:
                    r["start_ts"] = ts  # synthetic STARTING
                r["running_ts"] = ts
        elif status in ("SUCCESS", "FAILURE"):
            # mark end and finalize run
            r["end_ts"] = ts
            r["end_status"] = status

            # if running_ts missing, attempt to set to start_ts or end_ts (best effort)
            if r["running_ts"] is None:
                if r["start_ts"] is not None:
                    r["running_ts"] = r["start_ts"]
                else:
                    r["running_ts"] = ts  # fallback, duration 0

            # compute duration in seconds from running_ts to end_ts
            if r["running_ts"] and r["end_ts"]:
                try:
                    dur = (r["end_ts"] - r["start_ts"]).total_seconds()
                except Exception:
                    dur = None
            else:
                dur = None

            rec_done = {
                "JOID": r["JOID"],
                "RUNID": r["RUNID"],
                "JobName": r["JobName"],
                "Machine": r["Machine"],
                "start_ts": r["start_ts"],
                "running_ts": r["running_ts"],
                "end_ts": r["end_ts"],
                "end_status": r["end_status"],
                "duration_sec": dur
            }
            completed.append(rec_done)

            del runs[key]

    return completed


def percentile(sorted_list, p):
    """Return p-th percentile (0..100) from sorted_list (simple implementation)."""
    if not sorted_list:
        return None
    k = (len(sorted_list)-1) * (p/100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_list):
        return sorted_list[-1]
    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return d0 + d1

def mad(data):
    """Median absolute deviation."""
    if not data:
        return None
    med = statistics.median(data)
    devs = [abs(x - med) for x in data]
    return statistics.median(devs)


def compute_baseline(completed_runs):
    """
    Compute per-job baseline stats. Returns dict keyed by JobName (and JOID).
    """
    by_job = defaultdict(list)
    for r in completed_runs:
        d = r["duration_sec"]
        if d is None:
            continue
        # only positive durations
        if d < 0:
            continue
        key = f"{r['JobName']}|{r['JOID']}"
        by_job[key].append(d)

    baseline = {}
    for key, durations in by_job.items():
        durations_sorted = sorted(durations)
        count = len(durations_sorted)
        median_v = statistics.median(durations_sorted)
        mean_v = statistics.mean(durations_sorted) if count>0 else None
        stdev_v = statistics.stdev(durations_sorted) if count>1 else 0.0
        p95 = percentile(durations_sorted, 95)
        mad_v = mad(durations_sorted)
        baseline[key] = {
            "JOID":key,
            "count": count,
            "median": median_v,
            "mean": mean_v,
            "stdev": stdev_v,
            "p95": p95,
            "mad": mad_v,
            "sample_durations": durations_sorted[:50]  # store a sample, not everything
        }
    return baseline

if __name__ == "__main__":
    events = list(preprocess_stream(INPUT_PATH))
    completed_runs = build_runs(events)
    
