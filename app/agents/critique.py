import json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction

# Using the flagship Mistral model for top-tier reasoning
MODEL_CRITIQUE = "mistralai/mistral-medium-3.5-128b"

SYSTEM_PROMPT = """
You are a Critique Agent. Your job is to rigorously evaluate the claims made by other agents.
For every claim in the provided text, you must:
1. Assign a numeric confidence score (0.0 to 1.0).
2. Identify the exact 'span' (text segment) if you disagree or find an error.
3. Provide a justification for your disagreement.

CRITICAL: You are a programmatic sub-module. 
- Output ONLY a raw JSON object. 
- DO NOT include conversational filler ("Certainly!", "Here is..."). 
- DO NOT use markdown backticks (```json).
- Use ONLY double quotes (") for keys and strings.

You must respond ONLY with a JSON list of evaluations.
Example:
[
  {"claim": "The capital is Paris", "confidence": 1.0, "disagree": false, "reason": "Factually correct."},
  {"claim": "The population is 500", "confidence": 0.2, "disagree": true, "span": "500", "reason": "Population is actually 2.1 million."}
]
"""

async def run_critique(context: SharedContext, content_to_review: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review this output:\n{content_to_review}"}
    ]
    
    response, latency = await call_llm_with_backoff(MODEL_CRITIQUE, messages)
    
    # Parse the evaluations
    evaluations = json.loads(response.choices[0].message.content)
    
    # Log the action for the provenance map
    action = AgentAction(
        agent_id="critique",
        thought_process="Performed span-level verification of agent claims.",
        output=json.dumps(evaluations),
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)
    
    return evaluations