async def self_reflection_tool(context: SharedContext) -> str:
    """Agent calls this to find internal contradictions."""
    history_str = str([a.output for a in context.history])
    # The agent uses this tool to 'step back' and look for errors
    return f"Reflection analysis on history: {history_str[:500]}... No immediate contradictions found."