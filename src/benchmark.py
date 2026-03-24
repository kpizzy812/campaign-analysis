"""Inference speed benchmark."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def generate_batch(n: int) -> pd.DataFrame:
    """Generate a batch of random campaign features for benchmarking."""
    return pd.DataFrame({
        'bid': np.random.uniform(0.01, 3.0, n),
        'daily_budget': np.random.choice([10, 20, 30, 50, 100, 200, 500], n),
        'hour': np.random.randint(0, 24, n),
        'dow': np.random.randint(0, 7, n),
        'budget_per_bid': np.random.uniform(10, 5000, n),
        'geo': np.random.choice(['US', 'DE', 'GB', 'BR', 'IN'], n),
        'vertical': np.random.choice(['ecommerce', 'gambling', 'nutra'], n),
        'traffic_source': np.random.choice(['facebook', 'google', 'push'], n),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n),
        'os': np.random.choice(['android', 'ios', 'windows', 'macos'], n),
    })


def benchmark(model_path: str = 'models/model.joblib'):
    pipe = joblib.load(model_path)

    for batch_size in [1, 1_000, 10_000]:
        batch = generate_batch(batch_size)

        # Warmup
        pipe.predict_proba(batch)

        start = time.perf_counter()
        predictions = pipe.predict_proba(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_item_us = (elapsed_ms * 1000) / batch_size
        print(f"  {batch_size:>6,} predictions: {elapsed_ms:>8.1f}ms "
              f"({per_item_us:.1f}us/item)")

    # Verify 10K target
    batch_10k = generate_batch(10_000)
    start = time.perf_counter()
    pipe.predict_proba(batch_10k)
    elapsed = (time.perf_counter() - start) * 1000
    status = "PASS" if elapsed < 500 else "FAIL"
    print(f"\n  10K target (<500ms): {elapsed:.1f}ms [{status}]")


if __name__ == '__main__':
    print("=== Inference Benchmark ===\n")
    benchmark()
