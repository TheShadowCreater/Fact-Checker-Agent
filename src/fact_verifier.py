"""
fact_verifier.py
----------------
Verifies each extracted claim using Groq (LLaMA 3) for reasoning
and Tavily for live web search/retrieval.
Returns verdict, confidence score, reasoning, and source URLs.
"""

import json
import os
import re
from groq import Groq
from tavily import TavilyClient


VERIFICATION_SYSTEM_PROMPT = """You are a rigorous fact-checking assistant.
You will be given a claim and a set of web search results retrieved for that claim.

Your task:
1. Carefully read the search results provided.
2. Compare the claim against the evidence in the results.
3. Assign a verdict and confidence score.

VERDICT DEFINITIONS:
- "Verified"   : The claim is accurate and supported by credible sources.
- "Inaccurate" : The claim contains errors, outdated data, or significant exaggerations but has a factual basis.
- "False"      : The claim is contradicted by credible sources, or no credible evidence supports it.

CONFIDENCE SCORE: A float between 0.0 and 1.0 reflecting how certain you are in your verdict.
- 0.9–1.0 : Very strong evidence found
- 0.7–0.89: Good evidence found
- 0.5–0.69: Some evidence found; some uncertainty
- 0.3–0.49: Limited or conflicting evidence
- 0.0–0.29: Very little evidence found

Respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
Return exactly this structure:
{
  "verdict": "<Verified | Inaccurate | False>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<2–3 sentences explaining your verdict and what you found>",
  "sources": ["<URL 1>", "<URL 2>"]
}
"""


def _search_web(claim: str) -> tuple[str, list[str]]:
    """
    Use Tavily to search the web for evidence about a claim.

    Returns:
        (context_text, list_of_urls)
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in environment variables.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=claim,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
        include_raw_content=False,
    )

    urls = [r.get("url", "") for r in response.get("results", []) if r.get("url")]

    # Build a readable context block from results
    context_parts = []

    if response.get("answer"):
        context_parts.append(f"Summary: {response['answer']}")

    for i, result in enumerate(response.get("results", [])[:5], 1):
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        if content:
            context_parts.append(f"[Source {i}] {title}\nURL: {url}\n{content[:500]}")

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant search results found."
    return context_text, urls


def verify_claim(claim: str) -> dict:
    """
    Verify a single factual claim using Tavily (web search) + Groq (reasoning).

    Args:
        claim: The claim string to verify

    Returns:
        dict with keys: verdict, confidence, reasoning, sources
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return _fallback_result("GROQ_API_KEY is not set in environment variables.")

    # Step 1: Fetch web evidence via Tavily
    try:
        context_text, urls = _search_web(claim)
    except Exception as e:
        context_text = f"Web search failed: {e}"
        urls = []

    # Step 2: Ask Groq to reason over the evidence
    client = Groq(api_key=groq_key)

    user_message = f"""Fact-check the following claim using the web search results provided below.

CLAIM:
{claim}

WEB SEARCH RESULTS:
{context_text}

Based on these results, return your verdict as JSON.
The "sources" field should contain the most relevant URLs from the results.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.1,
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        result_text = response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_result(f"Groq inference error: {e}")

    # Strip markdown fences if present
    if result_text.startswith("```"):
        parts = result_text.split("```")
        result_text = parts[1] if len(parts) > 1 else result_text
        if result_text.startswith("json"):
            result_text = result_text[4:]
    result_text = result_text.strip()

    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return _fallback_result("Could not parse verification response.")
        else:
            return _fallback_result("Could not parse verification response.")

    # Merge Tavily URLs with any URLs the model extracted from the context
    model_sources = data.get("sources", [])
    merged_sources = list(dict.fromkeys(model_sources + urls))[:5]

    return {
        "verdict": data.get("verdict", "Inaccurate"),
        "confidence": float(data.get("confidence", 0.5)),
        "reasoning": data.get("reasoning", "No reasoning provided."),
        "sources": merged_sources,
    }


def _fallback_result(reason: str) -> dict:
    return {
        "verdict": "Inaccurate",
        "confidence": 0.3,
        "reasoning": reason,
        "sources": [],
    }
