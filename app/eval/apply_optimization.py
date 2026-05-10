import json
import os
import re

def apply_optimization(agent_name: str, new_prompt: str):
    """
    Automates Section 5: Overwrites the system prompt of a failing agent.
    """
    file_path = f"app/agents/{agent_name}.py"
    if not os.path.exists(file_path):
        print(f"❌ Agent {agent_name} not found.")
        return

    with open(file_path, "r") as f:
        content = f.read()

    # Regex to find and replace the SYSTEM_PROMPT string content
    pattern = r'(SYSTEM_PROMPT = """)(.*?)(""")'
    new_content = re.sub(pattern, f'SYSTEM_PROMPT = """\n{new_prompt}\n"""', content, flags=re.DOTALL)

    with open(file_path, "w") as f:
        f.write(new_content)
    
    print(f"[SUCCESS] Optimized {agent_name}.py with the new prompt.")

if __name__ == "__main__":
    # In a real loop, this would be fed by meta_agent.py
    # Example: Fix the 'Hallucination' issue in Retrieval
    new_retrieval_prompt = """
    You are a Fact-Checking Retrieval Agent.
    CRITICAL: If a query involves a specific event (e.g., '2025 Mars Strike'), 
    verify its existence. If no news exists, state 'NO SUCH EVENT FOUND' instead of guessing.
    """
    apply_optimization("retrieval", new_retrieval_prompt)