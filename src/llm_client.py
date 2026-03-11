"""
Async HTTP client for communicating with Ollama or Anthropic API.
Routes to Anthropic if ANTHROPIC_API_KEY is set, otherwise uses Ollama.
"""

import httpx
from config import Config


async def call_anthropic(
    messages: list[dict],
    config: Config,
) -> tuple[str, dict]:
    url = "https://api.anthropic.com/v1/messages"

    # Anthropic expects system prompt as top-level field, not a message
    system_prompt = ""
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": system_prompt,
        "messages": filtered_messages,
    }

    headers = {
        "x-api-key": config.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Anthropic returned {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()

    text: str = data["content"][0]["text"]
    usage: dict = {
        "eval_count": data["usage"].get("output_tokens", 0),
        "prompt_eval_count": data["usage"].get("input_tokens", 0),
    }

    return text, usage

async def call_ollama(
    messages: list[dict],
    config: Config,
) -> tuple[str, dict]:
    """
    POST a chat request to the Ollama /api/chat endpoint.
    """
    url = f"{config.api_base_url}/api/chat"

    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(url, json=payload)

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Ollama returned {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()

    text: str = data["message"]["content"]

    usage: dict = {
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
    }

    return text, usage


async def call_llm(
    messages: list[dict],
    config: Config,
) -> tuple[str, dict]:
    """
    Route to Anthropic or Ollama based on config.
    Use this instead of calling call_ollama directly.
    """
    if config.anthropic_api_key:
        return await call_anthropic(messages, config)
    return await call_ollama(messages, config)