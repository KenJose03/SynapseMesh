from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction
from app.core.utils import extract_json

MODEL_SYNTHESIS = "meta/llama-4-maverick-17b-128e-instruct"

SYSTEM_PROMPT = """
You are a Synthesis Agent. Your goal is to merge agent outputs into a final answer.
- You must resolve any contradictions flagged by the Critique agent.
- You must create a 'Provenance Map' linking every sentence to its source agent and source chunk.

CRITICAL: You are a programmatic sub-module. 
- Output ONLY a raw JSON object. 
- DO NOT include conversational filler ("Certainly!", "Here is..."). 
- DO NOT use markdown backticks (```json).
- Use ONLY double quotes (") for keys and strings.

OUTPUT FORMAT:
Your final answer must be a JSON object:
{
  "final_answer": "Your detailed response...",
  "provenance_map": [
    {"sentence": "Paris is the capital.", "source_agent": "retrieval", "chunk_id": "chunk_1"}
  ]
}
"""

async def run_synthesis(context: SharedContext):
    # Assemble all relevant context for the final merge
    full_context_history = "\n".join([f"{a.agent_id}: {a.output}" for a in context.history])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Finalize the answer based on this history:\n{full_context_history}"}
    ]
    
    response, latency = await call_llm_with_backoff(MODEL_SYNTHESIS, messages)
    data = extract_json(response.choices[0].message.content)
    
    context.final_answer = data["final_answer"]
    
    # Log the final synthesis action
    action = AgentAction(
        agent_id="synthesis",
        thought_process="Merged outputs and resolved critique flags.",
        output=data["final_answer"],
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)
    
    return data