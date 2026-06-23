import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_COLUMNS = [
    "Amount",
    "TransactionDay",
    "TransactionMonth",
    "TransactionWeekday",
    "TransactionQuarter",
    "TransactionYear",
    "AmountLog",
    "HighAmount",
    "merchant_transaction_count",
    "merchant_fraud_rate",
    "location_transaction_count",
    "location_fraud_rate",
    "type_transaction_count",
    "type_fraud_rate",
    "merchant_degree",
    "location_degree",
    "type_degree",
    "merchant_betweenness",
    "location_betweenness",
    "type_betweenness",
    "TransactionType",
    "Location",
]

NUMERIC_FEATURES = [
    "Amount",
    "TransactionDay",
    "TransactionMonth",
    "TransactionWeekday",
    "TransactionQuarter",
    "TransactionYear",
    "AmountLog",
    "HighAmount",
    "merchant_transaction_count",
    "merchant_fraud_rate",
    "location_transaction_count",
    "location_fraud_rate",
    "type_transaction_count",
    "type_fraud_rate",
    "merchant_degree",
    "location_degree",
    "type_degree",
    "merchant_betweenness",
    "location_betweenness",
    "type_betweenness",
]

CATEGORICAL_FEATURES = ["TransactionType", "Location"]

MODEL_PATH = Path("models/fraud_pipeline.joblib")
MODEL_METRICS_PATH = Path("models/model_metrics.json")
MODEL_REGISTRY_PATH = Path("models/model_registry.json")
AUTOENCODER_PATH = Path("models/autoencoder.h5")


def load_processed_data(path: Path) -> pd.DataFrame:
    logging.info("Loading processed dataset from %s", path)
    df = pd.read_csv(path)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_supervised_models(rf_estimators: int = 200, n_jobs: Optional[int] = None) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    models: Dict[str, Pipeline] = {}

    models["logistic_regression"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))]
    )
    models["decision_tree"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)),
        ]
    )
    models["random_forest"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=rf_estimators, class_weight="balanced", random_state=42, n_jobs=n_jobs),
            ),
        ]
    )
    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=n_jobs),
                ),
            ]
        )
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
                ),
            ]
        )
    return models


def build_unsupervised_models(n_jobs: Optional[int] = None) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["isolation_forest"] = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=n_jobs)
    models["one_class_svm"] = OneClassSVM(gamma="scale", nu=0.02)
    return models


def build_autoencoder(input_shape: int) -> Optional[keras.Model]:
    if not TENSORFLOW_AVAILABLE:
        return None

    encoder = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_shape,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
        ]
    )
    decoder = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(16,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(input_shape, activation="linear"),
        ]
    )
    inputs = keras.Input(shape=(input_shape,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = keras.Model(inputs, decoded, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def evaluate_model(name: str, model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    logging.info("Training model: %s", name)
    logging.info("Pipeline steps for %s: %s", name, model.steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logging.info("%s metrics: %s", name, {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def evaluate_unsupervised(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Dict[str, float]:
    logging.info("Training unsupervised model: %s", name)

    if preprocessor is not None:
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test

    model.fit(X_train_transformed)
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test_transformed)
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X_test_transformed)
    else:
        raise ValueError(f"Model {name} does not support anomaly scoring")

    anomaly_score = -raw_scores
    predicted = model.predict(X_test_transformed)
    predicted_labels = np.where(predicted == -1, 1, 0)

    metrics = {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(y_test, predicted_labels, zero_division=0),
        "recall": recall_score(y_test, predicted_labels, zero_division=0),
        "f1": f1_score(y_test, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(y_test, anomaly_score),
    }

    logging.info("%s metrics: %s", name, {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def train_autoencoder(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Optional[Dict[str, float]]:
    if not TENSORFLOW_AVAILABLE:
        return None

    if preprocessor is not None:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    autoencoder = build_autoencoder(X_train_scaled.shape[1])
    if autoencoder is None:
        return None

    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=128, verbose=0)
    reconstructed = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.square(X_test_scaled - reconstructed), axis=1)
    predicted_labels = np.where(mse > np.percentile(mse, 95), 1, 0)

    metrics = {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(y_test, predicted_labels, zero_division=0),
        "recall": recall_score(y_test, predicted_labels, zero_division=0),
        "f1": f1_score(y_test, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(y_test, mse),
    }

    autoencoder.save(AUTOENCODER_PATH)
    logging.info("Autoencoder saved to %s", AUTOENCODER_PATH)
    logging.info("Autoencoder metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def select_best_model(results: Dict[str, Dict[str, float]]) -> Tuple[str, Pipeline, Dict[str, float]]:
    best_name = None
    best_score = -1.0
    best_model = None
    best_metrics = {}

    for name, result in results.items():
        model, metrics = result["model"], result["metrics"]
        if metrics["roc_auc"] > best_score:
            best_name = name
            best_score = metrics["roc_auc"]
            best_model = model
            best_metrics = metrics

    assert best_name is not None
    return best_name, best_model, best_metrics


def load_registry(path: Path = MODEL_REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"models": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"models": []}


def save_registry_entry(
    model_output: Path,
    model_name: str,
    metrics: Dict[str, float],
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> Dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    version = len(registry.get("models", [])) + 1
    entry = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_file": str(model_output),
        "model_name": model_name,
        "metrics": metrics,
    }
    registry.setdefault("models", []).append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logging.info("Saved model registry entry to %s", registry_path)
    return entry


def save_metrics(report: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved model metrics to %s", path)


def train_model(
    processed_csv: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    model_output: Path = MODEL_PATH,
    metrics_output: Path = MODEL_METRICS_PATH,
    rf_estimators: int = 200,
    n_jobs: Optional[int] = None,
) -> None:
    df = load_processed_data(processed_csv)
    df = df.dropna(subset=FEATURE_COLUMNS + ["IsFraud"])

    X = df[FEATURE_COLUMNS]
    y = df["IsFraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    training_results: Dict[str, Dict[str, object]] = {}
    metrics_report: Dict[str, Dict[str, Dict[str, float]]] = {"supervised": {}, "unsupervised": {}}

    supervised_models = build_supervised_models(rf_estimators=rf_estimators, n_jobs=n_jobs)
    for name, model in supervised_models.items():
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        training_results[name] = {"model": model, "metrics": metrics}
        metrics_report["supervised"][name] = metrics

    best_name, best_model, best_metrics = select_best_model(training_results)
    logging.info("Selected best model: %s", best_name)

    metrics_report["best_supervised"] = {"name": best_name, "metrics": best_metrics}

    unsupervised_models = build_unsupervised_models(n_jobs=n_jobs)
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask] if normal_mask.sum() > 0 else X_train

    preprocessor = build_preprocessor()
    for name, model in unsupervised_models.items():
        metrics = evaluate_unsupervised(name, model, X_train_normal, X_test, y_test, preprocessor=preprocessor)
        metrics_report["unsupervised"][name] = metrics

    if TENSORFLOW_AVAILABLE:
        auto_metrics = train_autoencoder(X_train_normal, X_test, y_test, preprocessor=preprocessor)
        if auto_metrics is not None:
            metrics_report["unsupervised"]["autoencoder"] = auto_metrics

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name, "metrics": best_metrics}, model_output)
    logging.info("Saved best supervised model to %s", model_output)

    save_metrics(metrics_report, metrics_output)
    save_registry_entry(model_output, best_name, best_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model from processed data.")
    parser.add_argument(
        "--input-csv",
        default="data/processed_transactions.csv",
        help="Path to processed training CSV file.",
    )
    parser.add_argument(
        "--output-model",
        default=str(MODEL_PATH),
        help="File path to save the trained model pipeline.",
    )
    parser.add_argument(
        "--output-metrics",
        default=str(MODEL_METRICS_PATH),
        help="File path to save model evaluation metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--rf-estimators", type=int, default=200, help="Number of trees for RandomForest.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of jobs to run in parallel (pass 1 for single-threaded).")
    args = parser.parse_args()

    train_model(
        Path(args.input_csv),
        test_size=args.test_size,
        random_state=args.random_state,
        model_output=Path(args.output_model),
        metrics_output=Path(args.output_metrics),
        rf_estimators=args.rf_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
