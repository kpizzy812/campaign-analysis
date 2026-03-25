"""Part A: Creative effectiveness analyzer using Claude API."""

import asyncio
import json
import time

import re

import anthropic
from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 10  # parallel requests limit


def _parse_json(text: str) -> dict:
    """Parse JSON from Claude response, stripping markdown fences if present."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    return json.loads(cleaned)


def extract_creative_features(ad_text: str, client: anthropic.Anthropic = None) -> dict:
    """Extract features from a single ad text via Claude API (sync)."""
    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="""Ты — эксперт по анализу рекламных текстов в нише гемблинга и беттинга.
Извлеки признаки из текста объявления.
Отвечай ТОЛЬКО валидным JSON без пояснений.""",
        messages=[{"role": "user", "content": f"""Текст: {ad_text}

Верни JSON:
{{
  "has_number": bool,
  "has_urgency": bool,
  "has_social_proof": bool,
  "emotion": "fear|greed|excitement|neutral",
  "cta_strength": 1-5,
  "length_category": "short|medium|long",
  "key_benefit": "string"
}}"""}],
    )
    return _parse_json(response.content[0].text)


async def _extract_one(
    sem: asyncio.Semaphore,
    client: AsyncAnthropic,
    ad_text: str,
    idx: int,
) -> tuple[int, dict | None]:
    """Extract features for one ad with semaphore-limited concurrency."""
    async with sem:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=500,
                system="""Ты — эксперт по анализу рекламных текстов в нише гемблинга и беттинга.
Извлеки признаки из текста объявления.
Отвечай ТОЛЬКО валидным JSON без пояснений.""",
                messages=[{"role": "user", "content": f"""Текст: {ad_text}

Верни JSON:
{{
  "has_number": bool,
  "has_urgency": bool,
  "has_social_proof": bool,
  "emotion": "fear|greed|excitement|neutral",
  "cta_strength": 1-5,
  "length_category": "short|medium|long",
  "key_benefit": "string"
}}"""}],
            )
            features = _parse_json(response.content[0].text)
            return idx, features
        except Exception as e:
            print(f"  Error on ad {idx}: {e}")
            return idx, None


async def extract_batch_async(texts: list[str], max_concurrent: int = MAX_CONCURRENT) -> list[dict | None]:
    """Extract features for a batch of ads in parallel.

    Target: 100 ads < 60 seconds.
    """
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrent)

    tasks = [
        _extract_one(sem, client, text, i)
        for i, text in enumerate(texts)
    ]

    results = [None] * len(texts)
    for coro in asyncio.as_completed(tasks):
        idx, features = await coro
        results[idx] = features

    return results


def extract_batch(texts: list[str], max_concurrent: int = MAX_CONCURRENT) -> list[dict | None]:
    """Sync wrapper for batch extraction."""
    return asyncio.run(extract_batch_async(texts, max_concurrent))


if __name__ == "__main__":
    # Quick test with 5 ads
    import pandas as pd

    df = pd.read_csv("ads_dataset.csv")
    sample = df["ad_text"].head(5).tolist()

    print(f"Extracting features for {len(sample)} ads...")
    start = time.time()
    results = extract_batch(sample)
    elapsed = time.time() - start

    for i, (text, features) in enumerate(zip(sample, results)):
        print(f"\n--- Ad {i} ---")
        print(f"Text: {text[:80]}...")
        print(f"Features: {json.dumps(features, ensure_ascii=False, indent=2)}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(sample):.1f}s per ad)")
