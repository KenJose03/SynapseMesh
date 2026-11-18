import asyncio
from ddgs import DDGS
from typing import List, Dict, Any

async def web_search_tool(query: str) -> Dict[str, Any]:
    """
    Section 2: Tool Calling with Failure Modes.
    Returns structured results with URLs and relevance scores.
    """
    # 1. Failure Mode: Malformed Input
    if not query or len(query) < 2:
        return {
            "status": "malformed_input",
            "error": "Query is too short or empty.",
            "results": []
        }

    try:
        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=3))

        async with asyncio.timeout(5.0): 
            raw_results = await asyncio.to_thread(_sync_search)
                
            # 2. Failure Mode: Empty Results
            if not raw_results:
                return {
                    "status": "empty",
                    "results": [],
                    "message": "No relevant documents found."
                }

            # Format results with relevance scores (simulated) and URLs
            structured_results = []
            for i, res in enumerate(raw_results):
                structured_results.append({
                    "url": res.get("href"),
                    "snippet": res.get("body"),
                    "relevance": round(1.0 - (i * 0.2), 2), # Simulated score
                    "chunk_id": f"chunk_{i}"
                })

            return {
                "status": "success",
                "results": structured_results
            }

    except asyncio.TimeoutError:
        # 3. Failure Mode: Timeout
        return {
            "status": "timeout",
            "error": "Search exceeded 5 second limit.",
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "results": []
        }