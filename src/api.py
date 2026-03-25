"""FastAPI service for campaign ROI prediction."""

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import os
import redis as _redis

from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.creative_analyzer import extract_creative_features
from src.creative_generator import generate_ad_variants
from src.cached_predictor import CachedPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path('models/model.joblib')
METADATA_PATH = Path('models/metadata.json')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
CACHE_TTL = int(os.environ.get('CACHE_TTL', '300'))

model = None
metadata = None
cached_predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata, cached_predictor
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    # Init Redis cache
    if model is not None:
        try:
            redis_client = _redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            cached_predictor = CachedPredictor(model, redis_client, ttl_seconds=CACHE_TTL)
            logger.info(f"Redis cache connected: {REDIS_URL}, TTL={CACHE_TTL}s")
        except _redis.ConnectionError:
            logger.warning("Redis unavailable — running without cache")
            cached_predictor = None

    yield


app = FastAPI(title="Campaign ROI Predictor", lifespan=lifespan)


class CampaignInput(BaseModel):
    geo: str
    vertical: str
    traffic_source: str
    device: str
    os: str
    bid: float = Field(gt=0)
    daily_budget: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)


class PredictionOutput(BaseModel):
    probability: float
    prediction: int


class BatchOutput(BaseModel):
    predictions: list[PredictionOutput]
    count: int
    latency_ms: float


@app.get("/health")
def health():
    result = {
        "status": "healthy" if model is not None else "no_model",
        "model_version": metadata.get("model_type", "unknown") if metadata else "unknown",
        "trained_at": metadata.get("trained_at", "unknown") if metadata else "unknown",
        "metrics": metadata.get("metrics", {}) if metadata else {},
        "cache": cached_predictor.stats() if cached_predictor else {"status": "disabled"},
    }
    return result


def _prepare_input(items: list[CampaignInput]) -> pd.DataFrame:
    """Convert Pydantic models to DataFrame with engineered features."""
    records = [item.model_dump() for item in items]
    df = pd.DataFrame(records)
    df['budget_per_bid'] = df['daily_budget'] / df['bid'].clip(lower=0.001)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


@app.post("/predict", response_model=PredictionOutput)
def predict(item: CampaignInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = item.model_dump()

    # Use cached predictor if available
    if cached_predictor is not None:
        result = cached_predictor.predict(features)
        logger.info(
            "predict | geo=%s | prob=%.3f | cache=%s",
            item.geo, result['probability'], result['cache_hit'],
        )
        return PredictionOutput(
            probability=result['probability'], prediction=result['prediction'],
        )

    # Fallback: direct model call (no cache)
    start = time.perf_counter()
    df = _prepare_input([item])
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    latency = (time.perf_counter() - start) * 1000

    logger.info("predict | geo=%s | prob=%.3f | no_cache | %.1fms", item.geo, proba, latency)
    return PredictionOutput(probability=round(float(proba), 4), prediction=pred)


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(items: list[CampaignInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(items) > 10_000:
        raise HTTPException(status_code=422, detail="Batch size exceeds 10,000 limit")

    start = time.perf_counter()
    df = _prepare_input(items)
    probas = model.predict_proba(df)[:, 1]
    preds = (probas >= 0.5).astype(int)
    latency = (time.perf_counter() - start) * 1000

    logger.info(f"predict/batch | n={len(items)} | {latency:.1f}ms")

    predictions = [
        PredictionOutput(probability=round(float(p), 4), prediction=int(c))
        for p, c in zip(probas, preds)
    ]
    return BatchOutput(predictions=predictions, count=len(predictions), latency_ms=round(latency, 1))


# ============================================================
# Creative Analysis & Generation Endpoints
# ============================================================

class CreativeAnalyzeInput(BaseModel):
    ad_text: str


class CreativeAnalyzeOutput(BaseModel):
    features: dict
    predicted_ctr_percentile: int
    suggestions: list[str]


class CreativeGenerateInput(BaseModel):
    offer: str
    geo: str
    vertical: str
    n_variants: int = Field(default=5, ge=1, le=10)


class GeneratedVariant(BaseModel):
    text: str
    reasoning: str
    predicted_ctr_percentile: int


class CreativeGenerateOutput(BaseModel):
    variants: list[GeneratedVariant]
    count: int
    latency_ms: float


@app.post("/creatives/analyze", response_model=CreativeAnalyzeOutput)
def analyze_creative(item: CreativeAnalyzeInput):
    """Analyze ad text: extract features, predict CTR percentile, give suggestions."""
    import anthropic as _anthropic

    start = time.perf_counter()
    client = _anthropic.Anthropic()

    features = extract_creative_features(item.ad_text, client=client)

    # Simple CTR percentile based on feature count
    score = 0
    suggestions = []
    if features.get("has_number"):
        score += 25
    else:
        suggestions.append("Добавьте конкретные числа (бонус, скидка, количество пользователей)")
    if features.get("has_urgency"):
        score += 25
    else:
        suggestions.append("Добавьте элемент срочности ('только сегодня', 'осталось 24ч')")
    if features.get("has_social_proof"):
        score += 20
    else:
        suggestions.append("Добавьте социальное доказательство ('10,000+ уже выиграли')")
    if features.get("emotion") in ("excitement", "greed"):
        score += 15
    else:
        suggestions.append("Усильте эмоциональный заряд (excitement или greed)")
    if features.get("cta_strength", 0) >= 4:
        score += 15
    else:
        suggestions.append("Усильте призыв к действию (CTA)")

    latency = (time.perf_counter() - start) * 1000
    logger.info(f"creatives/analyze | {latency:.0f}ms")

    return CreativeAnalyzeOutput(
        features=features,
        predicted_ctr_percentile=min(score, 100),
        suggestions=suggestions[:3],
    )


@app.post("/creatives/generate", response_model=CreativeGenerateOutput)
def generate_creatives(item: CreativeGenerateInput):
    """Generate N ad variants using few-shot from top performers."""
    import anthropic as _anthropic
    import pandas as _pd

    start = time.perf_counter()
    client = _anthropic.Anthropic()

    # Load top performers from dataset
    ads_path = Path("ads_dataset.csv")
    if ads_path.exists():
        ads_df = _pd.read_csv(ads_path)
        vert_ads = ads_df[ads_df["vertical"] == item.vertical]
        top_texts = vert_ads.nlargest(10, "ctr")["ad_text"].tolist()
    else:
        top_texts = []

    variants_raw = generate_ad_variants(
        offer=item.offer,
        geo=item.geo,
        vertical=item.vertical,
        top_performers=top_texts,
        n_variants=item.n_variants,
        client=client,
    )

    variants = [
        GeneratedVariant(
            text=v.get("text", ""),
            reasoning=v.get("reasoning", ""),
            predicted_ctr_percentile=v.get("predicted_ctr_percentile", 50),
        )
        for v in variants_raw
    ]

    latency = (time.perf_counter() - start) * 1000
    logger.info(f"creatives/generate | n={len(variants)} | {latency:.0f}ms")

    return CreativeGenerateOutput(
        variants=variants,
        count=len(variants),
        latency_ms=round(latency, 1),
    )
