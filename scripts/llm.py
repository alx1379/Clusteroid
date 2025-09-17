import openai
import httpx

from configs.settings import (
    OPENAI_API_KEY,
    USE_PROXY,
    PROXY_URL
)

# LLM usage statistics
stats = {
    "gen_tokens": 0,
    "embed_tokens": 0,
    "calls": 0
}

# Configure OpenAI client with proxy (if specified)
if USE_PROXY and PROXY_URL:
    transport = httpx.HTTPTransport(proxy=PROXY_URL, verify=False)
    http_client = httpx.Client(transport=transport)
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
else:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def generate_llm_answer(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
"You are a strictly functional assistant. Respond exclusively based on the provided context. "
                "Don't take initiative, express concern, or offer help. "
                "Don't add polite phrases at the end of your response (e.g., 'If you have any questions...'). "
                "Your response should be precise, concise, and strictly to the point, like in technical documentation."
            )},        
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    if response.usage:
        stats["gen_tokens"] += response.usage.total_tokens
        stats["calls"] += 1

    return response.choices[0].message.content.strip()


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    if response.usage:
        stats["gen_tokens"] += response.usage.total_tokens
        stats["calls"] += 1

    return response.choices[0].message.content.strip()
