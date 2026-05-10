import json
import os
from app.core.llm import call_llm_with_backoff
from app.core.utils import extract_json

MODEL_META = "meta/llama-3.3-70b-instruct"

SYSTEM_PROMPT = """
You are a Meta-Optimization Agent. Analyze the provided LLM benchmark failures.
1. Identify the agent (Orchestrator, Retrieval, or Critique) that failed.
2. Propose a refined system prompt to fix the logical error.
3. Output ONLY valid JSON.
"""

async def run_meta_optimization():
    results_path = "app/eval/eval_results.json"
    if not os.path.exists(results_path):
        return "No evaluation data found. Run harness.py first."

    with open(results_path, "r") as f:
        failures = [r for r in json.load(f) if r['score'] < 1.0]

    if not failures:
        return "All tests passed! No optimization needed."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize based on these failures: {json.dumps(failures)}"}
    ]

    response, _ = await call_llm_with_backoff(MODEL_META, messages)
    proposal = extract_json(response.choices[0].message.content)
    
    print("\n" + "="*20 + " META-OPTIMIZATION PROPOSAL " + "="*20)
    print(json.dumps(proposal, indent=2))
    return proposal

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_meta_optimization())