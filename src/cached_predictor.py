"""Cached predictor with Redis backend."""

import hashlib
import json
import logging
import time
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class CachedPredictor:
    """Wraps an sklearn pipeline with Redis caching layer.

    Cache key: md5 hash of sorted feature dict.
    TTL: configurable, default 300s (5 min).

    Tracks hit/miss stats for /health endpoint.
    """

    def __init__(self, model, redis_client: redis.Redis, ttl_seconds: int = 300):
        self.model = model
        self.redis = redis_client
        self.ttl = ttl_seconds

    @property
    def total_requests(self) -> int:
        try:
            hits = int(self.redis.get("stats:cache_hits") or 0)
            misses = int(self.redis.get("stats:cache_misses") or 0)
            return hits + misses
        except redis.ConnectionError:
            return 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_requests
        if total == 0:
            return 0.0
        try:
            hits = int(self.redis.get("stats:cache_hits") or 0)
            return hits / total
        except redis.ConnectionError:
            return 0.0

    def _make_key(self, features: dict) -> str:
        payload = json.dumps(features, sort_keys=True, default=str).encode()
        return "pred:" + hashlib.md5(payload).hexdigest()

    def predict(self, features: dict) -> dict:
        """Predict with cache lookup.

        Returns dict with 'probability', 'prediction', 'cache_hit', 'latency_ms'.
        """
        cache_key = self._make_key(features)

        # Try cache
        try:
            cached = self.redis.get(cache_key)
            if cached:
                self.redis.incr("stats:cache_hits")
                result = json.loads(cached)
                result['cache_hit'] = True
                return result
        except redis.ConnectionError:
            logger.warning("Redis unavailable, skipping cache")

        # Cache miss — run model
        try:
            self.redis.incr("stats:cache_misses")
        except redis.ConnectionError:
            pass
        start = time.perf_counter()
        result = self._run_model(features)
        result['latency_ms'] = round((time.perf_counter() - start) * 1000, 2)
        result['cache_hit'] = False

        # Store in cache
        try:
            cache_payload = {
                'probability': result['probability'],
                'prediction': result['prediction'],
            }
            self.redis.setex(cache_key, self.ttl, json.dumps(cache_payload))
        except redis.ConnectionError:
            logger.warning("Redis unavailable, skipping cache write")

        return result

    def _run_model(self, features: dict) -> dict:
        """Run sklearn pipeline on feature dict."""
        import pandas as pd
        from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

        df = pd.DataFrame([features])
        df['budget_per_bid'] = df['daily_budget'] / df['bid'].clip(lower=0.001)
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        proba = self.model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)

        return {
            'probability': round(float(proba), 4),
            'prediction': pred,
        }

    def stats(self) -> dict:
        try:
            hits = int(self.redis.get("stats:cache_hits") or 0)
            misses = int(self.redis.get("stats:cache_misses") or 0)
            total = hits + misses
            rate = hits / total if total > 0 else 0.0
        except redis.ConnectionError:
            hits, misses, total, rate = 0, 0, 0, 0.0
        return {
            'cache_hits': hits,
            'cache_misses': misses,
            'cache_total': total,
            'cache_hit_rate': round(rate, 4),
        }
