# Churn Prediction ETL & ML Pipeline

An end-to-end data engineering and machine learning project that predicts which bank customers are likely to leave (churn). It covers the whole path: raw CSV → cleaning and validation → Parquet + BigQuery → SQL exploration → feature engineering → model training and tuning → a prediction API → Docker → scheduled orchestration with Airflow.

Built by **AURA Labs**.

---

## What it does

| Stage | What happens | Where |
|---|---|---|
| Extract | Reads the raw CSV (path comes from `config/config.yaml`) | `src/data/extract.py` |
| Transform | Snake-case columns, type downcasting, range and category fixes | `src/data/transform.py` |
| Validate | Hard quality gate: schema, nulls, duplicates, allowed categories, churn-rate sanity. If it fails, the pipeline stops before loading | `src/data/validate.py` |
| Load | Writes Parquet locally and (optionally) loads to BigQuery | `src/data/load.py` |
| Explore | SQL analysis with DuckDB (local) and BigQuery (cloud) | `src/data/explore_sql.py`, `explore_bigquery.py` |
| Features | One-hot encoding + engineered features, saves the exact column list used for training | `src/features/build_features.py` |
| Models | Random Forest, XGBoost, Logistic Regression, threshold analysis, hyper-parameter search | `src/models/` |
| API | FastAPI service that returns a churn probability and a decision | `src/api/` |
| Orchestration | Airflow DAG that runs the ETL daily | `dags/churn_pipeline_dag.py` |

---

## Key decisions (and why)

- **Data validation is a gate, not a report.** A custom `DataValidationError` halts the pipeline so bad data never reaches Parquet or BigQuery.
- **The `complain` column is excluded from the final models.** It is almost a copy of the target (customers who complained had already decided to leave), so using it would inflate the metrics without helping to predict churn early. The model is built on real behavioural and profile signals instead.
- **Class imbalance (about 20% churn) is handled in the model**, not by resampling: `class_weight="balanced"` for Random Forest and Logistic Regression, and `scale_pos_weight` computed from the data (about 3.9) for XGBoost.
- **Accuracy is not the main metric.** With 80/20 classes, accuracy is misleading. I use ROC-AUC (threshold-independent), plus recall, precision and the confusion matrix at a chosen threshold.
- **Models are compared at the same recall, not at the same threshold.** Each model spreads its probabilities differently, so one threshold favours some models unfairly. Fixing the share of churners to catch and counting the false alarms each model needs is the fair comparison (see Results).
- **The API serves Random Forest at a threshold of 0.24.** At the same recall it raises fewer false alarms than XGBoost (about 251 vs 296 at 75% recall on the test set). Missing a customer who is about to leave costs more than a false alarm, so the threshold is set far below 0.50: it catches about 75% of churners with about 55% precision.
- **No training-serving skew, and a test that proves it.** The exact feature columns used in training are saved to `feature_columns.json`, and the API builds the one-hot columns by hand to match them. `tests/test_serving_consistency.py` checks that the API path gives the same probability as the training matrix for 200 random customers. (It caught a real bug: `get_dummies(drop_first=True)` on a single row silently erased the country, gender and card type.)
- **Unknown categories are rejected.** The request schema only accepts the countries, genders and card types seen in training, so a typo returns a clear 422 instead of a silent wrong prediction.
- **Absolute imports (`from src.data.x import y`).** This lets the same code run as a script, from Docker, and from Airflow.

---

## Results

Test set: 2,000 customers, 408 of them churners. Produced by `python -m src.models.compare`; the raw numbers are in `reports/model_comparison.json`. Recall and precision are for the churner class.

| Model | ROC-AUC | Recall @ 0.50 | Precision @ 0.50 | Recall @ 0.40 | Precision @ 0.40 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7794 | 72.6% | 38.7% | 84.6% | 32.4% |
| Random Forest | 0.8695 | 44.1% | 79.3% | 56.1% | 71.3% |
| XGBoost | 0.8594 | 69.9% | 55.0% | 75.5% | 50.2% |

How to read it: ROC-AUC measures how well a model ranks customers and does not depend on a threshold. Random Forest ranks slightly better, but at the default 0.50 threshold it is cautious and misses most churners. XGBoost (trained with class weighting) flags more customers at the same threshold, so its recall is higher and its precision lower. Part of that gap is the threshold, not the model quality, which is why the models are also compared at the same recall. Logistic Regression ranks clearly worse, which fits the non-linear effect of the number of products.

**False alarms at the same recall** (customers flagged as churners who stayed; fewer is better):

| Churners caught | Random Forest | XGBoost | Logistic Regression |
|---|---|---|---|
| 70% | **193** | 236 | 428 |
| 75% | **251** | 296 | 513 |
| 80% | **379** | 392 | 595 |

Random Forest needs the fewest false alarms at every target, with a clear gap at 70-75% and a small one at 80%. The threshold was picked on this same test set, so the numbers are slightly optimistic; choosing it with cross-validation is a next step.

---

## Project structure

```
.
├── config/
│   └── config.yaml              # paths, BigQuery project/dataset/table
├── dags/
│   └── churn_pipeline_dag.py    # Airflow DAG: churn_etl_pipeline
├── data/
│   ├── raw/                     # put the Kaggle CSV here (not in git)
│   └── processed/               # generated Parquet files (not in git)
├── joblib/                      # trained models + feature_columns.json (not in git)
├── src/
│   ├── data/                    # extract, transform, validate, load, pipeline, explore_*
│   ├── features/                # build_features
│   ├── models/                  # train, evaluate, tune
│   └── api/                     # FastAPI app (main.py, schemas.py)
├── Dockerfile                   # API image
├── Dockerfile.airflow           # Airflow image
├── docker-compose.yml
└── requirements.txt
```

Every folder under `src/` needs an `__init__.py` so that `python -m src....` works.

---

## Getting started

### 1. Get the data

Download the dataset from Kaggle and place the CSV in `data/raw/`:

https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

The file name must match `raw_filename` in `config/config.yaml`.

### 2. Set up the environment

Python 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the ETL pipeline

Run everything from the project root with `-m`:

```bash
python -m src.data.pipeline
```

This extracts, cleans, validates and writes `data/processed/customer_churn_clean.parquet`.

### 4. (Optional) Load to BigQuery

1. Install the Google Cloud CLI and authenticate once:
   ```bash
   gcloud auth application-default login
   ```
2. Set `bigquery.project_id` (and dataset/table) in `config/config.yaml`.
3. Call `run_pipeline(upload_to_bigquery=True)` or run `python -m src.data.load`.

If `project_id` is empty, the BigQuery step is skipped.

### 5. Explore with SQL

```bash
python -m src.data.explore_sql        # DuckDB on the Parquet file
python -m src.data.explore_bigquery   # BigQuery (requires step 4)
```

### 6. Build features and train

```bash
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
python -m src.models.tune             # optional: Random Forest hyper-parameter search
```

Models and `feature_columns.json` are saved to `joblib/`. They are not committed, so run training once after cloning.

### 7. Run the API

```bash
uvicorn src.api.main:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Interactive docs (try `/predict` here): http://127.0.0.1:8000/docs

---

## Docker

```bash
# Prediction API (needs trained models in ./joblib)
docker compose up -d --build api

# Run the ETL once inside a container
docker compose --profile etl run --rm etl

# Airflow
docker compose up -d --build airflow
```

Note: BigQuery credentials are not available inside the containers, so `upload_to_bigquery` stays `False` there.

---

## Airflow

The DAG `churn_etl_pipeline` runs `run_pipeline()` daily (`catchup=False`).

1. Start it: `docker compose up -d airflow` (the first start takes a few minutes).
2. Get the admin password:
   ```bash
   docker logs churn_airflow 2>&1 | grep -i password
   ```
   On Windows PowerShell: `docker logs churn_airflow 2>&1 | Select-String "password"`.
3. Open http://localhost:8080 and sign in as `admin`.
4. Un-pause the DAG and trigger it, from the UI or:
   ```bash
   docker exec churn_airflow airflow dags unpause churn_etl_pipeline
   docker exec churn_airflow airflow dags trigger churn_etl_pipeline
   docker exec churn_airflow airflow dags list-runs -d churn_etl_pipeline
   ```

Airflow runs in `standalone` mode (SQLite, one container). That is fine for development and demos, but not for production.

### Screenshots

The DAG list and a successful run history (both runs finished in about 19 seconds):

![Airflow DAGs list](docs/images/airflow_dags.png)

![Airflow Grid view, all runs successful](docs/images/airflow_grid.png)

---

## Tech stack

Python · pandas · PyArrow / Parquet · DuckDB · Google BigQuery · scikit-learn · XGBoost · FastAPI · Pydantic · Uvicorn · Docker / Docker Compose · Apache Airflow

---

## Possible next steps

- Pick the decision threshold with cross-validation instead of the test set
- Hyper-parameter tuning for the served model (for example with Optuna)
- Unit tests for the transform and validate steps
- Write pipeline logs and validation reports to files
- Pass BigQuery credentials to containers through secrets
- CI with GitHub Actions

---

## Data source

Bank Customer Churn dataset on Kaggle: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn