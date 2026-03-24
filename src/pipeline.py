"""Preprocessing pipeline for campaign ROI prediction."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

NUMERIC_FEATURES = ['bid', 'daily_budget', 'hour', 'dow', 'budget_per_bid']
CATEGORICAL_FEATURES = ['geo', 'vertical', 'traffic_source', 'device', 'os']

LEAKAGE_COLS = ['revenue', 'spend', 'clicks', 'conversions', 'impressions',
                'roi', 'profit', 'cr', 'ctr', 'is_profitable', 'campaign_id']


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from raw campaign data.

    Returns (X, y) where X contains only safe pre-launch features.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour'] = df['created_at'].dt.hour
    df['dow'] = df['created_at'].dt.dayofweek
    df['budget_per_bid'] = df['daily_budget'] / df['bid'].clip(lower=0.001)

    y = (df['revenue'] > df['spend']).astype(int)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()

    return X, y


def build_pipeline(model=None) -> Pipeline:
    """Build sklearn Pipeline with ColumnTransformer preprocessing.

    If model is None, returns preprocessing-only pipeline.
    """
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    steps = [('preprocessor', preprocessor)]
    if model is not None:
        steps.append(('model', model))

    return Pipeline(steps)
