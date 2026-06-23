<<<<<<< HEAD
# Financial Fraud Detection App

Dashboard Link:https://shettysumantha-fraud-detection-app-dashboardapp-5l9ve5.streamlit.app/

## Step 1: Data Preprocessing & ETL

This project includes a preprocessing pipeline that cleans the raw transaction dataset, performs feature engineering, and writes cleaned records to both a CSV file and an SQLite database.

### Run preprocessing

```bash
cd "c:\Fraud Detection App"
python ml_pipeline/preprocess.py --input-csv data/credit_card_fraud_dataset.csv --output-csv data/processed_transactions.csv --sqlite-db database/fraud_detection.db --sqlite-table transactions
```

### Output

- `data/processed_transactions.csv` — cleaned and feature-engineered dataset
- `database/fraud_detection.db` — SQLite database with the `transactions` table

### Notes

- The pipeline handles duplicate removal, missing value cleaning, date parsing, categorical normalization, and fraud-related aggregate features.
- This is the first development step toward model training, API serving, and dashboard visualization.

## Step 2: Train the model

```bash
python ml_pipeline/train_model.py --input-csv data/processed_transactions.csv --output-model models/fraud_pipeline.joblib --output-metrics models/model_metrics.json
```

This training pipeline evaluates:
- supervised learning: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- unsupervised anomaly detection: Isolation Forest, One-Class SVM, Autoencoder (optional; install TensorFlow/Keras locally for autoencoder support)
- graph-based entity feature extraction from merchants, locations, and transaction types
- risk scoring and anomaly monitoring for dashboard analytics

The Streamlit dashboard displays dataset insights, fraud distribution charts, model evaluation comparison, anomaly analytics, and graph-based feature summaries.

## Step 3: Run the API

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Use POST `/api/predict` with transaction JSON to get fraud predictions.

Optional: view evaluation metrics via GET `/api/metrics`.

## Step 4: Streaming skeleton & alerting

A lightweight stream processor skeleton is included in `streaming/stream_processor.py`. It can run in batch mode by feeding newline-delimited JSON (`.jsonl`) or connect to Kafka when `kafka-python` is installed.

```bash
python streaming/stream_processor.py --mode batch --input-file data/stream_events.jsonl
```

## Step 5: Open the dashboard

```bash
streamlit run dashboard/app.py
```

## Step 6: Run tests

```bash
pytest -q
```
=======
# Fraud_Detection_App
>>>>>>> 0247286babbcbabe269df3c61af71b8050960f91
