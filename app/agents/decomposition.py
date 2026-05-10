import json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction
from app.core.utils import extract_json

MODEL_DECOMPOSITION = "meta/llama-4-maverick-17b-128e-instruct"

SYSTEM_PROMPT = """
You are a Decomposition Agent. Your goal is to deconstruct complex queries into a Directed Acyclic Graph (DAG) of sub-tasks.

CRITICAL ARCHITECTURE:
1. Identify dependencies: If Task B requires information from Task A, Task B must list Task A's ID in its 'dependencies' array.
2. Output Format: You must return ONLY a JSON object with a root key "tasks" containing the list of DAG nodes.

JSON SCHEMA:
{
  "tasks": [
    {
      "id": "search_1",
      "type": "retrieval",
      "description": "Find recent interest rate data...",
      "dependencies": []
    },
    {
      "id": "analysis_1",
      "type": "synthesis",
      "description": "Analyze the impact based on search_1 results...",
      "dependencies": ["search_1"] 
    }
  ]
}

NO PREAMBLE. NO MARKDOWN.
"""

async def run_decomposition(context: SharedContext):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Decompose this query: {context.original_query}"}
    ]
    
    response, latency = await call_llm_with_backoff(MODEL_DECOMPOSITION, messages)
    
    # 🛠️ THE FIX: Use extract_json and REMOVE the subsequent json.loads call.
    # extract_json handles the "tasks" wrapper automatically.
    tasks = extract_json(response.choices[0].message.content)
    
    context.task_graph = tasks
    
    action = AgentAction(
        agent_id="decomposition",
        thought_process="Deconstructed query into a dependency-aware task graph.",
        output=f"Created {len(tasks)} sub-tasks.",
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)
    
    return tasks