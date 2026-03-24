# Campaign ROI Predictor

ML-модель для предсказания прибыльности рекламных кампаний (ROI > 0). От EDA до рабочего API.

## Структура

```
src/
  pipeline.py       # sklearn Pipeline + ColumnTransformer
  train.py          # Baseline + LightGBM + Optuna + MLflow
  api.py            # FastAPI: /predict, /predict/batch, /health
  benchmark.py      # Скорость инференса
tests/              # Тесты pipeline и API
models/             # Обученная модель + метаданные
Dockerfile          # Контейнеризация
```

## Быстрый старт

### Обучение

```bash
# Сгенерировать данные (если нет campaigns_synthetic.csv)
python generate_data.py

# Обучить модель (baseline + Optuna + final)
python -m src.train
```

### Запуск API (локально)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Запуск через Docker

```bash
docker build -t ml-roi-predictor:v1 .
docker run -p 8000:8000 ml-roi-predictor:v1
```

### Тесты

```bash
python -m pytest tests/ -v
```

## API

### `POST /predict` — одно предсказание

```json
{
  "geo": "US", "vertical": "nutra", "traffic_source": "facebook",
  "device": "mobile", "os": "android",
  "bid": 0.5, "daily_budget": 100, "hour": 14, "dow": 2
}
```

Ответ:
```json
{"probability": 0.9234, "prediction": 1}
```

### `POST /predict/batch` — массив до 10 000 объектов

Тело: массив объектов того же формата. Ответ содержит `predictions`, `count`, `latency_ms`.

### `GET /health` — статус сервиса

```json
{
  "status": "healthy",
  "model_version": "LGBMClassifier",
  "trained_at": "2026-03-24 16:48:58",
  "metrics": {"roc_auc": 0.94, "precision": 0.96, "recall": 0.84, "f1": 0.89}
}
```

## Финальные метрики (hold-out test: May-Jun 2024)

| Метрика | Значение |
|---------|----------|
| **ROC-AUC** | **0.9404** |
| Precision | 0.9578 |
| Recall | 0.8382 |
| F1 | 0.8940 |
| Baseline (LogReg) ROC-AUC | 0.9420 +/- 0.0092 |

## Benchmark скорости

| Batch size | Время | Per item |
|------------|-------|----------|
| 1 | ~8ms | 8ms |
| 1,000 | ~17ms | 17us |
| 10,000 | ~115ms | 11.5us |

**10K target (<500ms): 118ms [PASS]**

## MLflow

Все эксперименты логируются в MLflow (локальный `mlruns/`). Запуск UI:

```bash
mlflow ui --port 5000
```

- `baseline_logreg` — LogisticRegression baseline
- `optuna_lgbm` — 30 Optuna trials (nested runs)
- `final_model` — лучшие параметры, метрики на hold-out

## Feature Store — архитектурная заметка

### Проблема

Агрегатные признаки (средний CTR по гео за 7 дней, средний CR по вертикали за 30 дней) нельзя считать в реальном времени при каждом запросе — это обращение к миллионам строк в event log.

### Решение

**Предрасчёт агрегатов по расписанию, хранение в быстром key-value store.**

#### Хранилище

| Компонент | Что хранит | Почему |
|-----------|-----------|--------|
| **Redis** | Горячие агрегаты (avg CTR by geo, avg CR by vertical) | Latency < 1ms на чтение. Модель дёргает Redis при инференсе |
| **ClickHouse** | Исторические агрегаты, сырые события | Быстрая аналитика по столбцам, оконные функции |
| **PostgreSQL** | Метаданные кампаний, конфиг модели | ACID, реляционные связи |

#### Пайплайн обновления

```
Event Log (Kafka/Kinesis)
    │
    ▼
Scheduled Job (Airflow / cron)         ← каждый час
    │
    ├── SELECT geo, AVG(ctr) FROM events
    │   WHERE created_at > now() - interval '7 days'
    │   GROUP BY geo
    │
    ├── Записать в Redis: feature:ctr_7d:{geo} = value
    │
    └── Записать в ClickHouse: feature_history table
```

#### Периодичность

| Признак | Обновление | TTL в Redis |
|---------|-----------|-------------|
| Avg CTR by geo (7d) | Каждый час | 2 часа |
| Avg CR by vertical (30d) | Каждые 6 часов | 12 часов |
| Avg ROI by geo+vertical (30d) | Ежедневно | 48 часов |
| Campaign count by source (7d) | Каждый час | 2 часа |

#### При инференсе

```python
# В API handler:
geo_ctr = redis.get(f"feature:ctr_7d:{campaign.geo}") or default
vertical_cr = redis.get(f"feature:cr_30d:{campaign.vertical}") or default
# Добавить к feature vector перед predict
```

Latency overhead: ~1-2ms на Redis lookup. Общий inference < 50ms.
