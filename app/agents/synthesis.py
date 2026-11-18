from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction
from app.core.utils import extract_json

MODEL_SYNTHESIS = "nvidia/nemotron-3.5-lightning-30b-a3b"

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
    # Assemble relevant evidence (excluding internal decomposition metadata)
    evidence_entries = [
        f"{a.agent_id}: {a.output}"
        for a in context.history
        if a.agent_id != "decomposition"
    ]
    full_context_history = "\n".join(evidence_entries) or "No prior agent evidence."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original User Query: {context.original_query}\n\n"
                f"Agent Evidence & Critique History:\n{full_context_history}\n\n"
                f"Directly answer the Original User Query using the verified evidence above."
            ),
        },
    ]
    
    response, latency = await call_llm_with_backoff(MODEL_SYNTHESIS, messages)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "") or ""
    data = extract_json(content)
    
    if isinstance(data, dict):
        final_answer = data.get("final_answer") or data.get("answer") or data.get("response") or str(data)
    elif isinstance(data, str) and data.strip():
        final_answer = data
    else:
        final_answer = str(data)
        
    context.final_answer = final_answer
    
    # Log the final synthesis action
    action = AgentAction(
        agent_id="synthesis",
        thought_process="Merged outputs and resolved critique flags.",
        output=final_answer,
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)
    
    return data