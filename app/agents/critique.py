from datetime import datetime
import json
from app.core.llm import call_llm_with_backoff
from app.core.schema import SharedContext, AgentAction
from app.core.utils import extract_json

# Using Nemotron 3.5 Super 120B MoE (~0.5s latency) to audit 30B Lightning outputs
MODEL_CRITIQUE = "nvidia/nemotron-3.5-super-vl-120b-a12b"

SYSTEM_PROMPT = """
You are a Critique Agent. Your job is to rigorously evaluate the claims made by other agents.
For every claim in the provided text, you must:
1. Assign a numeric confidence score (0.0 to 1.0).
2. Identify the exact 'span' (text segment) if you disagree or find an error.
3. Provide a justification for your disagreement.
4. IMPORTANT: Verify time-sensitive or post-2024 claims against the provided Live Retrieved Evidence and Today's Date. Do NOT reject a claim merely because it occurred after your training cutoff if it is supported by the Live Retrieved Evidence.

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
    if not content_to_review:
        return []

    today_date = datetime.now().strftime("%B %d, %Y")
    tool_snippets = "\n".join(
        f"[{t.tool_name} | {t.status}]: {t.output_data[:800]}"
        for t in context.tool_vault
    ) or "No live search evidence retrieved."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Today's Date: {today_date}\n"
                f"Original Query: {context.original_query}\n\n"
                f"Live Retrieved Evidence:\n{tool_snippets}\n\n"
                f"Output to Review:\n{content_to_review}"
            ),
        },
    ]
    
    response, latency = await call_llm_with_backoff(MODEL_CRITIQUE, messages)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "") or ""
    
    # Parse and normalize evaluations to a list so worker refinement always detects disagreements
    evaluations = extract_json(content)
    if isinstance(evaluations, dict):
        evaluations = [evaluations]
    elif not isinstance(evaluations, list):
        evaluations = []
    
    # Log the action for the provenance map
    action = AgentAction(
        agent_id="critique",
        thought_process="Performed span-level verification of agent claims.",
        output=json.dumps(evaluations) if not isinstance(evaluations, str) else evaluations,
        token_count=response.usage.total_tokens
    )
    context.history.append(action)
    context.add_tokens(response.usage.total_tokens)
    
    return evaluations