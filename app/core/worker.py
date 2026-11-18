import asyncio
import json
import os
import time
from app.core.schema import SharedContext, AgentAction, jobs_db
from app.core.llm import count_tokens, call_llm_with_backoff
from app.agents.orchestrator import run_orchestrator
from app.agents.decomposition import run_decomposition
from app.agents.retrieval import run_retrieval
from app.agents.critique import run_critique
from app.agents.synthesis import run_synthesis

MODEL_COMPRESSION = "nvidia/nemotron-3.5-bolt-e4b"

async def compress_context(context: SharedContext):
    prompt = f"Summarize history. Keep JSON/Citations [chunk_X].\n\nHistory: {context.history}"
    response, _ = await call_llm_with_backoff(MODEL_COMPRESSION, [{"role": "user", "content": prompt}])
    msg = response.choices[0].message
    summary = msg.content or getattr(msg, "reasoning_content", "") or "History compressed."
    context.history = [AgentAction(agent_id="compression_agent", thought_process="Budget exceeded.", output=summary, token_count=response.usage.total_tokens)]
    budget_cap = int(os.getenv("CONTEXT_BUDGET", "25000"))
    context.remaining_budget = max(500, budget_cap - response.usage.total_tokens)

async def execute_job(query: str, job_id: str):
    """The Main Execution Loop with SSE Streaming."""
    print("\n" + "="*60)
    print(f"🚀 [SynapseMesh] Job Started! ID: {job_id}")
    print(f"❓ Query: \"{query}\"")
    print("="*60 + "\n")

    context = SharedContext(job_id=job_id, original_query=query)
    
    # 0. Announce Job Initiation with Job ID
    yield f"data: {json.dumps({'event': 'job_started', 'job_id': job_id, 'query': query})}\n\n"

    # 1. Decomposition (Phase 1)
    yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'decomposition', 'job_id': job_id})}\n\n"
    
    plan_data = await run_decomposition(context)
    
    if isinstance(plan_data, dict) and "tasks" in plan_data:
        tasks = plan_data["tasks"]
    elif isinstance(plan_data, list):
        tasks = plan_data
    else:
        tasks = [
            {"id": "retrieval_1", "type": "retrieval", "description": query, "dependencies": []},
            {"id": "synthesis_1", "type": "synthesis", "description": "Synthesize final answer", "dependencies": ["retrieval_1"]}
        ]

    context.task_graph = tasks
    yield f"data: {json.dumps({'event': 'plan_created', 'tasks': tasks})}\n\n"

    # 2. Main Task Execution Loop
    completed_tasks = []
    max_iterations = max(10, len(context.task_graph) * 3)
    iteration = 0
    
    while len(completed_tasks) < len(context.task_graph) and iteration < max_iterations:
        iteration += 1
        if context.remaining_budget < 500:
            yield f"data: {json.dumps({'event': 'policy_violation', 'msg': 'Low budget.'})}\n\n"
            await compress_context(context)

        progress_made = False
        for task in context.task_graph:
            if not isinstance(task, dict):
                continue
            task_id = task.get('id', f"task_{len(completed_tasks)}")
            dependencies = task.get('dependencies', [])
            task_type = task.get('type', 'retrieval')

            if task_id not in completed_tasks and all(dep in completed_tasks for dep in dependencies):
                yield f"data: {json.dumps({'event': 'agent_start', 'agent': task_type, 'task_id': task_id})}\n\n"
                
                decision = await run_orchestrator(context)
                
                if task_type == 'retrieval':
                    result = await run_retrieval(context, task.get('description', query))
                elif task_type == 'synthesis':
                    result = await run_synthesis(context)
                else:
                    result = await run_synthesis(context)
                
                completed_tasks.append(task_id)
                progress_made = True
                yield f"data: {json.dumps({'event': 'task_complete', 'task_id': task_id, 'budget': context.remaining_budget})}\n\n"

        if not progress_made:
            # Force progress on stalled DAG
            for task in context.task_graph:
                if isinstance(task, dict) and task.get('id') not in completed_tasks:
                    completed_tasks.append(task['id'])

    # Guarantee final answer exists
    if not context.final_answer:
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'synthesis_fallback'})}\n\n"
        await run_synthesis(context)

    # 3. Final Critique & Refinement
    if context.final_answer:
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'critique'})}\n\n"
        evaluations = await run_critique(context, context.final_answer)
        
        has_disagreements = isinstance(evaluations, list) and any(
            isinstance(e, dict) and e.get("disagree") is True for e in evaluations
        )
        if has_disagreements:
            yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'synthesis_refinement'})}\n\n"
            await run_synthesis(context)

    # Store trace for GET /trace/{job_id}
    jobs_db[job_id] = context.model_dump() if hasattr(context, "model_dump") else context.dict()

    print("\n" + "="*60)
    print(f"✅ [SynapseMesh] Job {job_id} Finished!")
    print(f"🎯 Final Answer: {context.final_answer}")
    print("="*60 + "\n")

    yield f"data: {json.dumps({'event': 'final_answer', 'job_id': job_id, 'answer': context.final_answer})}\n\n"
    yield "data: [DONE]\n\n"