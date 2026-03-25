"""Part B: Creative variant generator using Claude API with few-shot."""

import json
import re

import anthropic

MODEL = "claude-sonnet-4-20250514"


def _parse_json(text: str) -> list | dict:
    """Parse JSON from Claude response, stripping markdown fences if present."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    return json.loads(cleaned)


def generate_ad_variants(
    offer: str,
    geo: str,
    vertical: str,
    top_performers: list[str],
    n_variants: int = 5,
    client: anthropic.Anthropic = None,
) -> list[dict]:
    """Generate ad variants using few-shot prompting with top performers.

    Args:
        offer: Description of the offer/product
        geo: Target geography (e.g. 'US', 'DE')
        vertical: Ad vertical (gambling, betting, nutra)
        top_performers: List of top-performing ad texts (by CTR)
        n_variants: Number of variants to generate
        client: Anthropic client (optional)

    Returns:
        List of dicts with 'text', 'reasoning', 'predicted_ctr_percentile'
    """
    if client is None:
        client = anthropic.Anthropic()

    # Build few-shot examples
    examples_text = "\n".join(
        f"{i+1}. {text}" for i, text in enumerate(top_performers[:10])
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=f"""Ты — эксперт по созданию рекламных текстов в нише {vertical}.
Твоя задача: написать эффективные рекламные объявления для гео {geo}.

Ориентируйся на лучшие объявления по CTR (ниже).
Используй те же паттерны: числа, срочность, социальное доказательство, эмоции.
Но НЕ копируй — создай новые уникальные варианты.

Отвечай ТОЛЬКО валидным JSON-массивом без пояснений.""",
        messages=[{"role": "user", "content": f"""Оффер: {offer}
Гео: {geo}
Вертикаль: {vertical}

Топ-10 объявлений по CTR (для вдохновения):
{examples_text}

Сгенерируй {n_variants} новых вариантов. Верни JSON-массив:
[
  {{
    "text": "текст объявления",
    "reasoning": "почему этот вариант должен работать (1 предложение)",
    "predicted_ctr_percentile": 0-100
  }}
]"""}],
    )

    return _parse_json(response.content[0].text)


def score_variant(
    ad_text: str,
    top_features: dict,
    client: anthropic.Anthropic = None,
) -> dict:
    """Score a generated variant against top-performer patterns.

    Args:
        ad_text: Generated ad text to evaluate
        top_features: Aggregated features of top performers
        client: Anthropic client

    Returns:
        Dict with score, matches, suggestions
    """
    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="""Ты — эксперт по оценке рекламных текстов.
Оцени текст объявления и дай рекомендации.
Отвечай ТОЛЬКО валидным JSON без пояснений.""",
        messages=[{"role": "user", "content": f"""Текст объявления: {ad_text}

Паттерны лучших объявлений:
- has_number: {top_features.get('pct_has_number', 'N/A')}% топов имеют числа
- has_urgency: {top_features.get('pct_has_urgency', 'N/A')}% топов имеют срочность
- has_social_proof: {top_features.get('pct_has_social_proof', 'N/A')}% топов имеют соц. доказательство
- dominant_emotion: {top_features.get('dominant_emotion', 'N/A')}
- avg_cta_strength: {top_features.get('avg_cta_strength', 'N/A')}

Верни JSON:
{{
  "score": 1-10,
  "matches_top_patterns": ["список совпадающих паттернов"],
  "missing_patterns": ["список отсутствующих паттернов"],
  "suggestions": ["3 конкретных совета по улучшению"]
}}"""}],
    )

    return _parse_json(response.content[0].text)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("ads_dataset.csv")

    # Get top performers for gambling vertical
    gambling = df[df["vertical"] == "gambling"].nlargest(10, "ctr")
    top_texts = gambling["ad_text"].tolist()

    print("=== Generating 5 variants ===\n")
    variants = generate_ad_variants(
        offer="Казино с бонусом 200% на первый депозит",
        geo="US",
        vertical="gambling",
        top_performers=top_texts,
        n_variants=5,
    )

    for i, v in enumerate(variants):
        print(f"\n--- Variant {i+1} ---")
        print(f"Text: {v['text']}")
        print(f"Reasoning: {v['reasoning']}")
        print(f"Predicted CTR percentile: {v['predicted_ctr_percentile']}")
