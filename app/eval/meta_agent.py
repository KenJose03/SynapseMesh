import json
import os
from app.core.llm import call_llm_with_backoff
from app.core.utils import extract_json
from app.eval.red_team import RED_TEAM_RESULTS_PATH

MODEL_META = "nvidia/nemotron-3.5-super-vl-120b-a12b"

SYSTEM_PROMPT = """
You are a Meta-Optimization Agent. Analyze the provided LLM benchmark failures (from both Tier 1 Golden Regression and Tier 2 Dynamic Red-Teaming).
1. Identify the primary agent (orchestrator, retrieval, critique, or synthesis) that failed.
2. Propose a refined system prompt to fix the logical error or failure mode.
3. Output ONLY valid JSON matching this schema:

{
  "status": "optimization_proposed",
  "report": {
    "failing_agent": "retrieval",
    "root_cause": "Brief summary of failure mode",
    "proposed_prompt_diff": "The complete replacement system prompt for the failing agent."
  }
}
"""


async def run_meta_optimization():
    failures = []

    for label, path in [("tier1_golden", "app/eval/eval_results.json"), ("tier2_red_team", RED_TEAM_RESULTS_PATH)]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for r in data:
                    if r.get("score", 0.0) < 0.8:
                        failures.append({"tier": label, **r})
            except Exception:
                pass

    if not failures:
        return "All tests passed across Tier 1 and Tier 2! No optimization needed."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize based on these failures: {json.dumps(failures)}"}
    ]

    response, _ = await call_llm_with_backoff(MODEL_META, messages)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "") or ""
    proposal = extract_json(content)

    print("\n" + "=" * 20 + " META-OPTIMIZATION PROPOSAL " + "=" * 20)
    print(json.dumps(proposal, indent=2))
    return proposal


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_meta_optimization())
