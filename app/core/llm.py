import os
import json
import asyncio
import sys
import time
import tiktoken
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file you just set up
load_dotenv()

# Initialize the OpenAI-compatible client for NVIDIA NIM
client = AsyncOpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_BASE_URL")
)

# Token counter for Llama-3/4 models (cl100k_base is the standard for most modern LLMs)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Helper to track context budget as per Section 3 of the assignment."""
    return len(tokenizer.encode(text))

async def call_llm_with_backoff(model: str, messages: list, temperature: float = 0.2):
    """
    Standard LLM call with Exponential Backoff for Rate Limits (Section 2).
    """
    max_retries = 5
    for i in range(max_retries):
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
    
            # 🕵️ EMERGENCY DEBUG: Print to stderr to bypass buffering
            sys.stderr.write("\n" + "!"*60 + "\n")
            sys.stderr.write(f"DEBUG: RAW RESPONSE FROM {model}:\n")
            sys.stderr.write(raw_content + "\n")
            sys.stderr.write("!"*60 + "\n")
            sys.stderr.flush() # Force it out now
            
            latency = time.time() - start_time
            
            # Log the token usage for the budget manager
            usage = response.usage.total_tokens
            
            return response, latency

        except Exception as e:
            # Catch 429 (Too Many Requests) for the 40 RPM free limit
            if "429" in str(e):
                wait_time = (2 ** i) + 1  # 1s, 3s, 5s, 9s, 17s...
                print(f"⚠️ Rate limit hit. Backing off for {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ LLM Error: {str(e)}")
                raise e

    raise Exception("Max retries exceeded for NVIDIA API.")

async def stream_llm_response(model: str, messages: list, agent_id: str):
    """
    SSE-compatible streaming for Section 6.
    Yields chunks formatted for Server-Sent Events.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                # Format required for SSE: "data: <payload>\n\n"
                payload = {
                    "agent": agent_id,
                    "token": content
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"