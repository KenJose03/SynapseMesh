from app.core.utils import extract_json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction, ToolLog
from app.tools.search import web_search_tool
import json
import time

MODEL_RETRIEVAL = "nvidia/nemotron-3.5-lightning-30b-a3b"

SYSTEM_PROMPT = """
You are a Retrieval-Augmented Agent. Your goal is to answer queries using multi-hop reasoning.
- You MUST perform at least two search 'hops' before answering.
- Use the first hop to find broad info, and the second hop to drill down into specifics found in the first hop.
- Every claim in your final output must be cited using [chunk_X] format.

CRITICAL: You are a programmatic sub-module. 
- Output ONLY a raw JSON object. 
- DO NOT include conversational filler ("Certainly!", "Here is..."). 
- DO NOT use markdown backticks (```json).
- Use ONLY double quotes (") for keys and strings.

OUTPUT FORMAT:
Your final output must be a JSON object:
{
  "reasoning": "Explain your multi-hop steps",
  "answer": "The final answer with [chunk_X] citations",
  "sources": ["list of URLs used"]
}
"""

async def run_retrieval(context: SharedContext, sub_task_query: str):
    # Hop 1: Initial Search
    start_time = time.time()
    search_1 = await web_search_tool(sub_task_query)
    
    # Log the tool call (Requirement: Section 2)
    context.tool_vault.append(ToolLog(
        tool_name="web_search",
        input_data=sub_task_query,
        output_data=str(search_1),
        latency=time.time() - start_time,
        status=search_1["status"]
    ))

    # Hop 2: Deep Dive (Simplified for this logic - we'll let the LLM generate the 2nd query)
    # In a production-grade system, we'd feed search_1 back to the LLM to get 'sub_query_2'
    # For the take-home, we can trigger a second focused search based on the first result's snippet.
    
    deep_dive_query = f"specific details about {sub_task_query}"
    if search_1["status"] == "success":
        deep_dive_query = f"{sub_task_query} context: {search_1['results'][0]['snippet'][:50]}"

    start_time_2 = time.time()
    search_2 = await web_search_tool(deep_dive_query)
    
    context.tool_vault.append(ToolLog(
        tool_name="web_search_hop_2",
        input_data=deep_dive_query,
        output_data=str(search_2),
        latency=time.time() - start_time_2,
        status=search_2["status"]
    ))

    # Combine all chunks for the LLM to reason over
    all_chunks = search_1.get("results", []) + search_2.get("results", [])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {sub_task_query}\n\nRetrieved Chunks:\n{all_chunks}"}
    ]

    response, latency = await call_llm_with_backoff(MODEL_RETRIEVAL, messages)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "") or ""
    data = extract_json(content)

    if isinstance(data, dict):
        reasoning = data.get("reasoning", "Retrieved and analyzed search chunks.")
        answer = data.get("answer") or data.get("response") or str(data)
    else:
        reasoning = "Retrieved and analyzed search chunks."
        answer = str(data)

    # Log the agent action
    action = AgentAction(
        agent_id="retrieval",
        thought_process=reasoning,
        output=answer,
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)

    return {"reasoning": reasoning, "answer": answer}