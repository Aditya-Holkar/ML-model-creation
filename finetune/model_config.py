import json
import requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are an ML engineer. Given a user's description of what they want their fine-tuned model to do, output a JSON config with exactly these fields:
- "base_model": one of ["TinyLlama-1.1B", "Phi-3-mini", "Mistral-7B", "Llama-3.1-8B"]
- "format": one of ["instruction", "chat"]
- "task_type": short description (e.g. "classification", "summarization", "Q&A", "generation")
- "system_prompt": system prompt for chat format (or "" if instruction)
- "description": one-line model card description
- "epochs_suggestion": integer 1-5
- "reasoning": one sentence why this config fits their use case

Return ONLY valid JSON, no markdown."""


def generate_config(description: str, api_key: str) -> dict:
    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())
