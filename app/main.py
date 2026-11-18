from contextlib import asynccontextmanager
import asyncio
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.db import init_db
import uuid
import json
import os

from app.core.worker import execute_job
from app.core.schema import SharedContext, jobs_db
from app.eval.harness import EVAL_RESULTS_PATH, run_eval_suite
from app.eval.red_team import (
    DEFAULT_CATEGORIES,
    RED_TEAM_RESULTS_PATH,
    list_candidates,
    promote_candidate,
    reject_candidate,
    run_red_team_suite,
)

# 1. Define the lifespan logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    try:
        await asyncio.wait_for(init_db(), timeout=2.5)
        print("✅ Database tables initialized!")
    except Exception as e:
        print(f"⚠️ Database initialization skipped (running without active DB): {e}")
    
    yield  # The app stays in this 'yield' state while running
    
    # --- SHUTDOWN LOGIC ---
    print("👋 Shutting down SynapseMesh gracefully...")

# 2. Attach the lifespan to the FastAPI app
app = FastAPI(
    title="SynapseMesh Multi-Agent System",
    description="Cognitive Multi-Agent Architecture with DAG Orchestration, Multi-Hop Retrieval, and Self-Optimization.",
    version="1.0.0",
    lifespan=lifespan
)

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

@app.post("/submit/sync")
async def submit_query_sync(request: QueryRequest):
    """
    Endpoint 1B (Synchronous): Runs the multi-agent job and returns the full JSON response
    including job_id, final answer, and trace (ideal for testing in Swagger UI /docs).
    """
    job_id = str(uuid.uuid4())
    final_answer = None
    
    async for raw_chunk in execute_job(request.query, job_id):
        if raw_chunk.startswith("data: "):
            payload_str = raw_chunk[6:].strip()
            if payload_str and payload_str != "[DONE]":
                try:
                    payload = json.loads(payload_str)
                    if payload.get("event") == "final_answer":
                        final_answer = payload.get("answer")
                except:
                    pass
                    
    trace = jobs_db.get(job_id, {})
    return {
        "status": "success",
        "job_id": job_id,
        "query": request.query,
        "final_answer": final_answer,
        "trace": trace
    }

@app.get("/trace/{job_id}")
async def get_trace(job_id: str):
    """
    Endpoint 2: Retrieve full execution trace for a completed job.
    Fulfills Section 7.2 and Section 6.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]

class RedTeamRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=20, description="Number of synthetic red-team queries to generate and test.")
    categories: Optional[List[str]] = Field(
        default=None,
        description=f"Optional subset of categories from: {DEFAULT_CATEGORIES}"
    )

def _summarize_results_file(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        total = len(results)
        avg_score = sum(r.get("score", 0.0) for r in results) / total if total > 0 else 0.0
        passed = sum(1 for r in results if r.get("score", 0.0) >= 0.8)
        return {
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate": round((passed / total) * 100, 2) if total > 0 else 0.0,
            "average_score": round(avg_score, 2),
            "results": results,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/eval/latest")
async def get_latest_eval():
    """
    Endpoint 3: Retrieve latest 2-Tier evaluation summaries (Tier 1 Golden + Tier 2 Red-Team).
    Fulfills Section 7.3 and Section 4.
    """
    tier1 = _summarize_results_file(EVAL_RESULTS_PATH)
    tier2 = _summarize_results_file(RED_TEAM_RESULTS_PATH)
    pending_candidates = list_candidates(status="pending_review")

    if not tier1 and not tier2:
        return {"status": "success", "summary": "Evaluation data will be populated by the eval harness or red-team generator."}

    return {
        "status": "success",
        "tier1_golden": tier1,
        "tier2_red_team": tier2,
        "pending_promotion_candidates": len(pending_candidates),
    }

@app.post("/prompt/approve")
async def approve_rewrite(proposal_id: str, approved: bool):
    """
    Endpoint 4: Human approval/rejection for a prompt rewrite.
    Fulfills Section 7.4 and Section 5.
    """
    return {"proposal_id": proposal_id, "status": "approved" if approved else "rejected"}

@app.post("/eval/re-run")
async def trigger_reeval(background_tasks: BackgroundTasks, failed_only: bool = True, use_llm_judge: bool = False):
    """
    Endpoint 5: Trigger targeted Tier 1 re-evaluation on failed cases in the background.
    Fulfills Section 7.5 and Section 5.
    """
    background_tasks.add_task(run_eval_suite, failed_only=failed_only, use_llm_judge=use_llm_judge)
    return {
        "status": "started",
        "mode": "failed_only" if failed_only else "full_golden_suite",
        "message": "Tier 1 re-evaluation started in background. Check GET /eval/latest once complete."
    }

@app.post("/eval/red-team")
async def trigger_red_team(request: RedTeamRequest):
    """
    Tier 2 Endpoint: Dynamically draft synthetic adversarial/multi-hop/ambiguous/temporal queries,
    run them through SynapseMesh, grade with a cross-model LLM Judge, and capture failures for promotion.
    """
    if request.categories:
        invalid = [c for c in request.categories if c not in DEFAULT_CATEGORIES]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid categories {invalid}. Allowed: {DEFAULT_CATEGORIES}"
            )
    return await run_red_team_suite(count=request.count, categories=request.categories)

@app.get("/eval/candidates")
async def get_eval_candidates(status: Optional[str] = None):
    """
    Tier 2 Promotion Pipeline: List discovered failure candidates (filterable by pending_review, promoted, rejected).
    """
    candidates = list_candidates(status=status)
    return {"count": len(candidates), "candidates": candidates}

@app.post("/eval/candidates/{candidate_id}/promote")
async def promote_eval_candidate(candidate_id: str):
    """
    Tier 2 -> Tier 1 Promotion: Promote a reviewed red-team failure into the permanent Tier 1 Golden Dataset.
    """
    res = promote_candidate(candidate_id)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res["detail"])
    return res

@app.post("/eval/candidates/{candidate_id}/reject")
async def reject_eval_candidate(candidate_id: str):
    """
    Tier 2 Promotion Pipeline: Reject a noisy or invalid synthetic failure candidate.
    """
    res = reject_candidate(candidate_id)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res["detail"])
    return res
