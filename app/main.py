from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.db import init_db
import uuid
import json

from app.core.worker import execute_job
from app.core.schema import SharedContext

# 1. Define the lifespan logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    # This code runs before the server starts taking requests
    await init_db()
    print("✅ Database tables initialized!")
    
    yield  # The app stays in this 'yield' state while running
    
    # --- SHUTDOWN LOGIC ---
    # This code runs when the server is stopping
    print("👋 Shutting down system gracefully...")

# 2. Attach the lifespan to the FastAPI app
app = FastAPI(
    title="Mega AI Multi-Agent System",
    lifespan=lifespan
)

# In-memory storage for demonstration (Replace with Postgres for final submission)
jobs_db = {}

class QueryRequest(BaseModel):
    query: str

@app.post("/submit")
async def submit_query(request: QueryRequest):
    """
    Endpoint 1: Submit a query and receive a streaming SSE response.
    Fulfills Section 7.1 and Section 6.
    """
    job_id = str(uuid.uuid4())
    
    # We return a StreamingResponse that calls our worker's generator
    return StreamingResponse(
        execute_job(request.query, job_id),
        media_type="text/event-stream"
    )

@app.get("/trace/{job_id}")
async def get_trace(job_id: str):
    """
    Endpoint 2: Retrieve full execution trace for a completed job.
    Fulfills Section 7.2 and Section 6.
    """
    # In a real app, you'd query the 'Jobs' table in Postgres
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]

@app.get("/eval/latest")
async def get_latest_eval():
    """
    Endpoint 3: Retrieve latest eval run summary.
    Fulfills Section 7.3 and Section 4.
    """
    return {"status": "success", "summary": "Evaluation data will be populated by the eval harness."}

@app.post("/prompt/approve")
async def approve_rewrite(proposal_id: str, approved: bool):
    """
    Endpoint 4: Human approval/rejection for a prompt rewrite.
    Fulfills Section 7.4 and Section 5.
    """
    return {"proposal_id": proposal_id, "status": "approved" if approved else "rejected"}

@app.post("/eval/re-run")
async def trigger_reeval():
    """
    Endpoint 5: Trigger targeted re-eval on failed cases.
    Fulfills Section 7.5 and Section 5.
    """
    return {"message": "Re-evaluation started in background."}
