from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import random

app = FastAPI()

# ---- CORS (important for frontend fetch) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request model ----
class CheckFileRequest(BaseModel):
    filename: str
    region: str


@app.post("/check-file")
def check_file(payload: CheckFileRequest):
    """
    Dummy API returning data IDENTICAL to what your JS expects
    """

    # simulate application status codes
    status_codes = [0, 10, 20, 30, 40]
    status_code = random.choice(status_codes)

    response = {
        "File_Name": payload.filename,
        "ftp": {
            "sftp_status": "CLOSE",
            "procesing_status": "Processing Completed",
            "transfered_status": {
                "transfered_to": f"{payload.region.lower()}-server-01",
                "transfered_at": "/fns/id/r/spool/Interfaces/BATCH-UPLOADS-TEMP"
            },
            "Archival_path": f"/archives/{payload.region}/BATCH-UPLOADS/20260110",
            "failure_status": None
        },
        "Application": {
            "status": status_code,
            "Report_status": [
                "Report generated",
                "Extraction file generated"
            ],
            "Report_details": [
                "Total records: 1500",
                "Successful records: 1495",
                "Failed records: 5"
            ]
        },
        "Error": None
    }

    return response
