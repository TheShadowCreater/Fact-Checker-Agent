"""
claim_extractor.py
------------------
Uses Groq API (LLaMA 3) to extract verifiable factual claims from document text.
"""

import json
import os
from groq import Groq


EXTRACTION_SYSTEM_PROMPT = """You are a precise fact-extraction assistant.
Your job is to identify verifiable factual claims from documents.

Focus ONLY on claims that contain:
- Statistics and numerical data (e.g., "X% of users", "revenue grew by $Y")
- Dates and timelines (e.g., "launched in 2019", "by Q3 2024")
- Financial figures (e.g., "valued at $5 billion", "profit margin of 23%")
- Scientific or technical statements (e.g., "temperature increased by 1.5°C")
- Named entity facts (e.g., "Company X employs 10,000 people")
- Comparative claims (e.g., "fastest growing in the industry")

Do NOT include:
- Opinions or subjective statements
- Vague claims without specific data
- Questions or hypotheticals

Respond ONLY with a valid JSON array. No markdown, no explanation.
Each object must have exactly these fields:
{
  "id": <integer starting at 1>,
  "claim": "<the exact factual claim as a clear, standalone sentence>",
  "category": "<one of: statistic | date | financial | technical | comparative | other>",
  "source_snippet": "<verbatim short excerpt from the text that contains this claim>"
}
"""


def extract_claims(text: str, max_claims: int = 20) -> list[dict]:
    """
    Extract verifiable factual claims from document text using Groq (LLaMA 3).

    Args:
        text: The full document text
        max_claims: Maximum number of claims to extract

    Returns:
        List of claim dicts with keys: id, claim, category, source_snippet
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    client = Groq(api_key=api_key)

    # Truncate very long texts to avoid token limits
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Document truncated for processing]"

    user_message = f"""Extract up to {max_claims} verifiable factual claims from this document.
Return ONLY a JSON array — no markdown fences, no commentary.

DOCUMENT TEXT:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    claims = json.loads(raw)

    if not isinstance(claims, list):
        raise ValueError("Groq did not return a JSON array of claims.")

    # Normalise and validate
    validated = []
    for item in claims:
        if isinstance(item, dict) and "claim" in item:
            validated.append(
                {
                    "id": item.get("id", len(validated) + 1),
                    "claim": str(item["claim"]).strip(),
                    "category": item.get("category", "other"),
                    "source_snippet": item.get("source_snippet", ""),
                }
            )

    return validated
