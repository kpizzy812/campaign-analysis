"""Training script: baseline + LightGBM + Optuna + MLflow."""

import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.pipeline import (
    prepare_features, build_pipeline,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

DATA_PATH = Path('campaigns_synthetic.csv')
MODELS_DIR = Path('models')


def load_and_split() -> tuple:
    """Load data and do temporal split: train=Jan-Apr, test=May-Jun."""
    df = pd.read_csv(DATA_PATH)
    df['created_at'] = pd.to_datetime(df['created_at'])

    train_df = df[df['created_at'] < '2024-05-01']
    test_df = df[df['created_at'] >= '2024-05-01']

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    return X_train, y_train, X_test, y_test


def run_baseline(X_train, y_train) -> float:
    """LogisticRegression baseline with StratifiedKFold(5)."""
    numeric_pipe = Pipeline([('scaler', StandardScaler())])
    categorical_pipe = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, NUMERIC_FEATURES),
        ('cat', categorical_pipe, CATEGORICAL_FEATURES),
    ])
    baseline_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(baseline_pipe, X_train, y_train,
                             cv=cv, scoring='roc_auc')

    print(f"Baseline ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    with mlflow.start_run(run_name="baseline_logreg"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("roc_auc_mean", scores.mean())
        mlflow.log_metric("roc_auc_std", scores.std())

    return scores.mean()


def run_optuna(X_train, y_train, n_trials: int = 30) -> dict:
    """Tune LightGBM with Optuna, log every trial to MLflow."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="optuna_lgbm") as parent_run:

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
                'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
            }

            model = LGBMClassifier(
                **params,
                class_weight='balanced',
                random_state=42,
                verbosity=-1,
            )
            pipe = build_pipeline(model=model)

            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring='roc_auc',
            )

            with mlflow.start_run(
                run_name=f"trial_{trial.number}", nested=True
            ):
                mlflow.log_params(params)
                mlflow.log_metric('roc_auc_mean', scores.mean())
                mlflow.log_metric('roc_auc_std', scores.std())

            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric('best_roc_auc', study.best_value)
        print(f"Best ROC-AUC = {study.best_value:.4f}")
        print(f"Best params: {best}")

    return best


def train_final(X_train, y_train, X_test, y_test, best_params: dict):
    """Train final model on full train set, evaluate on hold-out test."""
    lgbm_params = {
        'n_estimators': best_params['n_estimators'],
        'learning_rate': best_params['lr'],
        'num_leaves': best_params['num_leaves'],
        'min_child_samples': best_params['min_child_samples'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample'],
    }

    model = LGBMClassifier(
        **lgbm_params,
        class_weight='balanced',
        random_state=42,
        verbosity=-1,
    )
    pipe = build_pipeline(model=model)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
    }

    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(lgbm_params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    print(f"\n=== Hold-out Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return pipe, metrics


def save_model(pipe: Pipeline, metrics: dict):
    """Save model and metadata to models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / 'model.joblib')

    metadata = {
        'model_type': 'LGBMClassifier',
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'features_numeric': NUMERIC_FEATURES,
        'features_categorical': CATEGORICAL_FEATURES,
    }
    with open(MODELS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODELS_DIR}/")


def main():
    mlflow.set_experiment("campaign_roi_prediction")

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_and_split()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n--- Baseline ---")
    run_baseline(X_train, y_train)

    print("\n--- Optuna Tuning (LightGBM) ---")
    best_params = run_optuna(X_train, y_train, n_trials=30)

    print("\n--- Final Model ---")
    pipe, metrics = train_final(X_train, y_train, X_test, y_test, best_params)

    save_model(pipe, metrics)

    return pipe, metrics


if __name__ == '__main__':
    main()
