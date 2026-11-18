import json
import os
from app.eval.red_team import (
    FAILED_CANDIDATES_PATH,
    GOLDEN_DATASET_PATH,
    RED_TEAM_RESULTS_PATH,
)

EVAL_RESULTS_PATH = "app/eval/eval_results.json"


def _print_tier_summary(title: str, results: list):
    total_cases = len(results)
    if total_cases == 0:
        print(f"\n{title}: No results available yet.")
        return

    passed = len([r for r in results if r.get("score", 0.0) >= 0.8])
    success_rate = (passed / total_cases) * 100
    avg_score = sum(r.get("score", 0.0) for r in results) / total_cases

    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    print(f"Total Cases         : {total_cases}")
    print(f"Overall Pass Rate   : {success_rate:.2f}% ({passed}/{total_cases})")
    print(f"Average Score       : {avg_score:.2f} / 1.00")

    categories = {}
    failure_modes = {}
    for r in results:
        cat = r.get("type", "unknown")
        categories.setdefault(cat, []).append(r.get("score", 0.0))
        fm = r.get("failure_mode")
        if fm and fm != "none":
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

    print("\n--- Breakdown by Category ---")
    for cat, scores in categories.items():
        print(f"  {cat.upper():<15}: {sum(scores)/len(scores):.2f} ({len(scores)} cases)")

    if failure_modes:
        print("\n--- Discovered Failure Modes ---")
        for fm, cnt in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"  {fm:<20}: {cnt}")


def generate_performance_report():
    """
    2-Tier Performance Audit:
    - Tier 1: Golden Regression Suite (eval_results.json)
    - Tier 2: Dynamic Red-Teaming Suite (red_team_results.json) + Pending Candidates
    """
    if os.path.exists(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            _print_tier_summary("TIER 1: GOLDEN REGRESSION AUDIT", json.load(f))
    else:
        print("\n⚠️ Tier 1 results not found. Run: python -m app.eval.harness")

    if os.path.exists(RED_TEAM_RESULTS_PATH):
        with open(RED_TEAM_RESULTS_PATH, "r", encoding="utf-8") as f:
            _print_tier_summary("TIER 2: DYNAMIC RED-TEAM AUDIT", json.load(f))
    else:
        print("\nℹ️ Tier 2 Red-Team results not found yet. Run: python -m app.eval.red_team")

    if os.path.exists(FAILED_CANDIDATES_PATH):
        with open(FAILED_CANDIDATES_PATH, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        pending = [c for c in candidates if c.get("status") == "pending_review"]
        promoted = [c for c in candidates if c.get("status") == "promoted"]
        print("\n--- Failure Promotion Pipeline ---")
        print(f"  Pending Review Candidates : {len(pending)}")
        print(f"  Promoted to Tier 1 Golden : {len(promoted)}")
        if os.path.exists(GOLDEN_DATASET_PATH):
            with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
                print(f"  Current Tier 1 Suite Size : {len(json.load(f))} cases")


if __name__ == "__main__":
    generate_performance_report()
