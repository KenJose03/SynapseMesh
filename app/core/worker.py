import asyncio
import json
import time
from app.core.schema import SharedContext, AgentAction
from app.core.llm import count_tokens, call_llm_with_backoff
from app.agents.orchestrator import run_orchestrator
from app.agents.decomposition import run_decomposition
from app.agents.retrieval import run_retrieval
from app.agents.critique import run_critique
from app.agents.synthesis import run_synthesis

MODEL_COMPRESSION = "meta/llama-3.1-8b-instruct"

async def compress_context(context: SharedContext):
    prompt = f"Summarize history. Keep JSON/Citations [chunk_X].\n\nHistory: {context.history}"
    response, _ = await call_llm_with_backoff(MODEL_COMPRESSION, [{"role": "user", "content": prompt}])
    summary = response.choices[0].message.content
    context.history = [AgentAction(agent_id="compression_agent", thought_process="Budget exceeded.", output=summary, token_count=response.usage.total_tokens)]
    context.remaining_budget = 4000 - response.usage.total_tokens

async def execute_job(query: str, job_id: str):
    """The Main Execution Loop with SSE Streaming."""
    context = SharedContext(job_id=job_id, original_query=query)
    
    # 1. Decomposition (Phase 1)
    yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'decomposition'})}\n\n"
    
    # 🛠️ THE FIX: Only call decomposition ONCE and ensure it's a list.
    plan_data = await run_decomposition(context)
    
    if isinstance(plan_data, dict) and "tasks" in plan_data:
        tasks = plan_data["tasks"]
    else:
        tasks = plan_data # decomposition.py now handles the unwrap via extract_json

    context.task_graph = tasks
    yield f"data: {json.dumps({'event': 'plan_created', 'tasks': tasks})}\n\n"

    # 2. Main Task Execution Loop
    completed_tasks = []
    while len(completed_tasks) < len(context.task_graph):
        if context.remaining_budget < 500:
            yield f"data: {json.dumps({'event': 'policy_violation', 'msg': 'Low budget.'})}\n\n"
            await compress_context(context)

        for task in context.task_graph:
            # Now 'task' is guaranteed to be a dict, so task['id'] won't fail
            if task['id'] not in completed_tasks and all(dep in completed_tasks for dep in task['dependencies']):
                yield f"data: {json.dumps({'event': 'agent_start', 'agent': task['type']})}\n\n"
                
                decision = await run_orchestrator(context)
                
                if task['type'] == 'retrieval':
                    result = await run_retrieval(context, task['description'])
                elif task['type'] == 'synthesis':
                    result = await run_synthesis(context)
                
                completed_tasks.append(task['id'])
                yield f"data: {json.dumps({'event': 'task_complete', 'task_id': task['id'], 'budget': context.remaining_budget})}\n\n"

    # 3. Final Critique & Refinement
    yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'critique'})}\n\n"
    await run_critique(context, context.final_answer)
    yield f"data: {json.dumps({'event': 'final_answer', 'answer': context.final_answer})}\n\n"