import json
import re

def extract_json(text: str):
    """
    Robustly extracts JSON and logs the raw output for debugging.
    Fulfills Section 2 (Failure Handling).
    """
    # 1. FORCED LOGGING: This will appear in your Docker terminal
    print("\n" + "="*50)
    print("🔍 RAW AI RESPONSE CAPTURED:")
    print(text)
    print("="*50 + "\n")

    if not text: return {}

    text = text.strip()
    
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n|```$', '', text, flags=re.MULTILINE | re.IGNORECASE).strip()
    try:
        data = json.loads(text)
        # AUTO-UNWRAP: If the AI wrapped our list in an object like {"tasks": [...]}
        # and we only see one key, return the inner list automatically.
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            if isinstance(data[key], list):
                return data[key]
                
        return data
    except json.JSONDecodeError:
        # 4. Regex fallback: Find the first { and last }
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            # Fix single quotes to double quotes for JSON compliance
            fixed = match.group(0).replace("'", '"')
            try:
                return json.loads(fixed)
            except:
                pass
        raise ValueError(f"Could not parse AI response as JSON: {text[:50]}")