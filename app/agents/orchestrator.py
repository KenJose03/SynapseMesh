import json
from app.core.utils import extract_json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext

# We use the 70B model for the Brain
MODEL_ORCHESTRATOR = "nvidia/nemotron-3.5-lightning-30b-a3b"

SYSTEM_PROMPT = """
You are the Master Orchestrator of a multi-agent AI system. 
Your job is to analyze the user query and the current state, then decide which agent to invoke next.

AVAILABLE AGENTS:
1. decomposition: Use this first if the query is complex or ambiguous. It breaks the task into a plan.
2. retrieval: Use this to find facts, search the web, or query the database.
3. synthesis: Use this only when you have enough information to provide the final answer.

CRITICAL: You are a programmatic sub-module. 
- Output ONLY a raw JSON object. 
- DO NOT include conversational filler ("Certainly!", "Here is..."). 
- DO NOT use markdown backticks (```json).
- Use ONLY double quotes (") for keys and strings.

RESPONSE FORMAT:
You must respond ONLY with a valid JSON object:
{
  "next_agent": "agent_name",
  "reasoning": "Brief explanation of why this agent was chosen"
}
"""

async def run_orchestrator(context: SharedContext):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {context.original_query}\nCurrent History: {context.history}"}
    ]
    
    # FIX: Unpack the tuple (response, latency)
    response, latency = await call_llm_with_backoff(MODEL_ORCHESTRATOR, messages)
    
    # Now you can access .choices correctly
    decision = extract_json(response.choices[0].message.content)
    
    return decision