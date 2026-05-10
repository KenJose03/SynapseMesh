import json

def generate_performance_report():
    """
    Fulfills Section 6: Metrics and Performance Audit.
    """
    with open("app/eval/eval_results.json", "r") as f:
        results = json.load(f)

    total_cases = len(results)
    success_rate = (len([r for r in results if r['score'] >= 1.0]) / total_cases) * 100
    
    print("\n" + "📊" + " SYSTEM PERFORMANCE AUDIT " + "📊")
    print(f"Overall Success Rate: {success_rate:.2f}%")
    print(f"Average Score: {sum(r['score'] for r in results)/total_cases:.2f}")
    
    # Categorize performance
    categories = {}
    for r in results:
        cat = r['type']
        categories[cat] = categories.get(cat, []) + [r['score']]
    
    print("\n--- Breakdown by Category ---")
    for cat, scores in categories.items():
        print(f"{cat.upper()}: {sum(scores)/len(scores):.2f}")

if __name__ == "__main__":
    generate_performance_report()