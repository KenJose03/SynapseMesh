# app/eval/run_meta_loop.py
import asyncio
from app.eval.meta_agent import run_meta_optimization
from app.eval.apply_optimization import apply_optimization

async def main():
    # 1. Brain: Get the optimization proposal from the Meta Agent
    proposal = await run_meta_optimization()
    
    if isinstance(proposal, dict) and proposal.get("status") == "optimization_proposed":
        report = proposal["report"]
        # 2. Hands: Automatically apply the fix to the agent
        apply_optimization(report["failing_agent"], report["proposed_prompt_diff"])
        print(f"Loop Complete: Optimized {report['failing_agent']}")

if __name__ == "__main__":
    asyncio.run(main())