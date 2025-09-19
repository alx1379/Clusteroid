# Copyright 2025 Alex Erofeev / AIGENTTO
# Created by Alex Erofeev at AIGENTTO (http://aigentto.com/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
