import json
import os

from pydantic import ValidationError

from schemas import AskResponse

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 2

def call_llm(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def call_llm_for_structured_response(prompt: str) -> AskResponse:
    structured_prompt = prompt + (
        "\n\nRespond with ONLY a JSON object matching exactly this schema: "
        '{"answer": string, "sources": array of strings, "confidence": float between 0 and 1}. '
        "No other text, no markdown code fences."
    )

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        raw = call_llm(structured_prompt)
        try:
            parsed = json.loads(raw)
            return AskResponse(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            structured_prompt = (
                prompt
                + "\n\nYour previous response did not match the required JSON schema "
                f"(error: {e}). Respond with ONLY a valid JSON object matching exactly: "
                '{"answer": string, "sources": array of strings, "confidence": float between 0 and 1}.'
            )

    return AskResponse(
        answer=f"[error] LLM failed to produce a schema-valid response after {MAX_RETRIES + 1} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )
