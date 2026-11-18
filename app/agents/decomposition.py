import json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction
from app.core.utils import extract_json

MODEL_DECOMPOSITION = "nvidia/nemotron-3.5-lightning-30b-a3b"

SYSTEM_PROMPT = """
You are a Decomposition Agent. Your goal is to deconstruct complex queries into a Directed Acyclic Graph (DAG) of sub-tasks.

CRITICAL ARCHITECTURE:
1. Identify dependencies: If Task B requires information from Task A, Task B must list Task A's ID in its 'dependencies' array.
2. Output Format: You must return ONLY a JSON object with a root key "tasks" containing the list of DAG nodes.
3. Graph Completion: Every graph must terminate in a "synthesis" task that consolidates findings into the final answer.
4. Ambiguity & Missing Inputs: If the user query is ambiguous or missing essential inputs (e.g., "Calculate the ROI", "What's the weather like?", "Tell me about the merger" without specifying which numbers, location, or companies), do NOT create web retrieval tasks; create ONLY a single "synthesis" task instructing the Synthesis Agent to ask the user for the missing information.

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
    
    # Extract tasks list robustly (handling reasoning_content fallback)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "") or ""
    parsed = extract_json(content)
    
    if isinstance(parsed, dict) and "tasks" in parsed:
        tasks = parsed["tasks"]
    elif isinstance(parsed, list):
        tasks = parsed
    else:
        tasks = []

    # Ensure validity and guarantee termination with a synthesis task
    if not isinstance(tasks, list) or len(tasks) == 0:
        tasks = [
            {"id": "retrieval_1", "type": "retrieval", "description": context.original_query, "dependencies": []},
            {"id": "synthesis_1", "type": "synthesis", "description": "Synthesize the final answer.", "dependencies": ["retrieval_1"]}
        ]
    else:
        # Guarantee task IDs and dependency lists are well-formed
        has_synthesis = False
        task_ids = []
        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                continue
            if "id" not in t:
                t["id"] = f"task_{i+1}"
            if "type" not in t:
                t["type"] = "retrieval"
            if "dependencies" not in t or not isinstance(t["dependencies"], list):
                t["dependencies"] = []
            if t["type"] == "synthesis":
                has_synthesis = True
            task_ids.append(t["id"])

        if not has_synthesis:
            tasks.append({
                "id": f"synthesis_{len(tasks)+1}",
                "type": "synthesis",
                "description": "Consolidate and synthesize the final answer.",
                "dependencies": task_ids.copy()
            })
            
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