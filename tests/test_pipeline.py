import pandas as pd
import numpy as np
from src.pipeline import prepare_features, build_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_prepare_features():
    df = pd.DataFrame({
        'campaign_id': ['c1'],
        'geo': ['US'], 'vertical': ['nutra'], 'traffic_source': ['facebook'],
        'device': ['mobile'], 'os': ['android'],
        'bid': [0.5], 'daily_budget': [100],
        'impressions': [10000], 'clicks': [500], 'conversions': [50],
        'spend': [250.0], 'revenue': [600.0],
        'created_at': ['2024-03-15 14:00:00'],
    })
    X, y = prepare_features(df)
    assert 'hour' in X.columns
    assert 'dow' in X.columns
    assert 'budget_per_bid' in X.columns
    assert 'revenue' not in X.columns
    assert 'spend' not in X.columns
    assert 'clicks' not in X.columns
    assert 'impressions' not in X.columns
    assert 'conversions' not in X.columns
    assert y.iloc[0] == 1  # revenue > spend -> profitable
    assert len(X.columns) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_pipeline_fit_transform():
    np.random.seed(42)
    df = pd.DataFrame({
        'campaign_id': [f'c{i}' for i in range(100)],
        'geo': np.random.choice(['US', 'DE', 'BR'], 100),
        'vertical': np.random.choice(['nutra', 'gambling'], 100),
        'traffic_source': np.random.choice(['facebook', 'google'], 100),
        'device': np.random.choice(['mobile', 'desktop'], 100),
        'os': np.random.choice(['android', 'ios'], 100),
        'bid': np.random.uniform(0.1, 1.0, 100),
        'daily_budget': np.random.choice([10, 50, 100], 100),
        'impressions': np.random.randint(1000, 10000, 100),
        'clicks': np.random.randint(10, 500, 100),
        'conversions': np.random.randint(0, 50, 100),
        'spend': np.random.uniform(10, 500, 100),
        'revenue': np.random.uniform(0, 1000, 100),
        'created_at': pd.date_range('2024-01-01', periods=100, freq='h'),
    })
    X, y = prepare_features(df)
    pipe = build_pipeline()
    X_transformed = pipe.fit_transform(X)
    assert X_transformed.shape[0] == 100
    assert X_transformed.shape[1] == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_no_leakage_columns():
    df = pd.DataFrame({
        'campaign_id': ['c1'],
        'geo': ['US'], 'vertical': ['nutra'], 'traffic_source': ['facebook'],
        'device': ['mobile'], 'os': ['android'],
        'bid': [0.5], 'daily_budget': [100],
        'impressions': [10000], 'clicks': [500], 'conversions': [50],
        'spend': [250.0], 'revenue': [600.0],
        'created_at': ['2024-03-15 14:00:00'],
    })
    X, y = prepare_features(df)
    leakage = {'revenue', 'spend', 'clicks', 'conversions', 'impressions',
               'roi', 'profit', 'cr', 'ctr', 'is_profitable', 'campaign_id'}
    assert leakage.isdisjoint(set(X.columns))
