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

from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path('models/model.joblib')
METADATA_PATH = Path('models/metadata.json')

model = None
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
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
    return {
        "status": "healthy" if model is not None else "no_model",
        "model_version": metadata.get("model_type", "unknown") if metadata else "unknown",
        "trained_at": metadata.get("trained_at", "unknown") if metadata else "unknown",
        "metrics": metadata.get("metrics", {}) if metadata else {},
    }


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

    start = time.perf_counter()
    df = _prepare_input([item])
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    latency = (time.perf_counter() - start) * 1000

    logger.info(f"predict | geo={item.geo} | prob={proba:.3f} | {latency:.1f}ms")
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
