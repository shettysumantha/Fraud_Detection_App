import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = Path("models/fraud_pipeline.joblib")
HISTORICAL_DATA = Path("data/processed_transactions.csv")
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


def load_pipeline(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_history(path: Path = HISTORICAL_DATA) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        for col in ["MerchantID", "TransactionType", "Location"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
        if "IsFraud" in df.columns:
            df["IsFraud"] = pd.to_numeric(df["IsFraud"], errors="coerce").fillna(0).astype(int)
        return df
    return pd.DataFrame()


def build_aggregate_lookup(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df["MerchantID"] = df["MerchantID"].astype(str).fillna("")
    df["Location"] = df["Location"].astype(str).fillna("")
    df["TransactionType"] = df["TransactionType"].astype(str).fillna("")

    return {
        "merchant": df.groupby("MerchantID")["IsFraud"].agg(["count", "mean"]).rename(columns={"count": "merchant_transaction_count", "mean": "merchant_fraud_rate"}),
        "location": df.groupby("Location")["IsFraud"].agg(["count", "mean"]).rename(columns={"count": "location_transaction_count", "mean": "location_fraud_rate"}),
        "type": df.groupby("TransactionType")["IsFraud"].agg(["count", "mean"]).rename(columns={"count": "type_transaction_count", "mean": "type_fraud_rate"}),
    }


def compute_high_amount_threshold(history_df: pd.DataFrame) -> float:
    if history_df.empty or "Amount" not in history_df.columns:
        return 0.0
    return float(history_df["Amount"].quantile(0.90))


def build_graph_features(history_df: pd.DataFrame, row: pd.Series) -> Dict[str, float]:
    merchant_node = f"merchant:{row['MerchantID']}"
    location_node = f"location:{row['Location']}"
    type_node = f"type:{row['TransactionType']}"

    graph = nx.Graph()
    if not history_df.empty:
        for _, record in history_df.iterrows():
            graph.add_edge(f"merchant:{record['MerchantID']}", f"location:{record['Location']}")
            graph.add_edge(f"merchant:{record['MerchantID']}", f"type:{record['TransactionType']}")
            graph.add_edge(f"location:{record['Location']}", f"type:{record['TransactionType']}")

    graph.add_edge(merchant_node, location_node)
    graph.add_edge(merchant_node, type_node)
    graph.add_edge(location_node, type_node)

    betweenness = nx.betweenness_centrality(graph)
    degrees = dict(graph.degree())

    return {
        "merchant_degree": int(degrees.get(merchant_node, 0)),
        "location_degree": int(degrees.get(location_node, 0)),
        "type_degree": int(degrees.get(type_node, 0)),
        "merchant_betweenness": float(betweenness.get(merchant_node, 0.0)),
        "location_betweenness": float(betweenness.get(location_node, 0.0)),
        "type_betweenness": float(betweenness.get(type_node, 0.0)),
    }


def build_input_dataframe(data: Dict[str, Any], history_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([data]).copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    df["TransactionType"] = df["TransactionType"].astype(str).str.lower().str.strip()
    df["Location"] = df["Location"].astype(str).str.title().str.strip()
    df["MerchantID"] = df["MerchantID"].astype(str)

    df["TransactionDay"] = df["TransactionDate"].dt.day
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionWeekday"] = df["TransactionDate"].dt.weekday
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["AmountLog"] = np.log1p(df["Amount"])

    high_amount_threshold = compute_high_amount_threshold(history_df)
    if high_amount_threshold > 0:
        df["HighAmount"] = (df["Amount"] >= high_amount_threshold).astype(int)
    else:
        df["HighAmount"] = 0

    if not history_df.empty:
        history_df = history_df.copy()
        history_df["MerchantID"] = history_df["MerchantID"].astype(str).fillna("")
        history_df["Location"] = history_df["Location"].astype(str).fillna("")
        history_df["TransactionType"] = history_df["TransactionType"].astype(str).fillna("")

        lookups = build_aggregate_lookup(history_df)
        df = df.merge(lookups["merchant"], how="left", left_on="MerchantID", right_index=True)
        df = df.merge(lookups["location"], how="left", on="Location")
        df = df.merge(lookups["type"], how="left", on="TransactionType")

    df["merchant_transaction_count"] = df.get("merchant_transaction_count", 0).fillna(0).astype(int)
    df["merchant_fraud_rate"] = df.get("merchant_fraud_rate", 0.0).fillna(0.0)
    df["location_transaction_count"] = df.get("location_transaction_count", 0).fillna(0).astype(int)
    df["location_fraud_rate"] = df.get("location_fraud_rate", 0.0).fillna(0.0)
    df["type_transaction_count"] = df.get("type_transaction_count", 0).fillna(0).astype(int)
    df["type_fraud_rate"] = df.get("type_fraud_rate", 0.0).fillna(0.0)

    graph_features = build_graph_features(history_df, df.iloc[0])
    for key, value in graph_features.items():
        df[key] = value

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    for col in missing_columns:
        df[col] = 0

    return df[FEATURE_COLUMNS]


def predict_transaction(data: Dict[str, Any], model_path: Path = MODEL_PATH, history_path: Path = HISTORICAL_DATA) -> Dict[str, Any]:
    pipeline_data = load_pipeline(model_path)
    model = pipeline_data["model"] if isinstance(pipeline_data, dict) else pipeline_data
    history_df = load_history(history_path)
    X = build_input_dataframe(data, history_df)

    score = float(model.predict_proba(X)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.decision_function(X)[0])
    label = int(model.predict(X)[0])

    return {
        "fraud_probability": round(score, 4),
        "label": label,
        "input": data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fraud for a single transaction JSON payload.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to saved model pipeline.")
    parser.add_argument("--history", default=str(HISTORICAL_DATA), help="Processed history CSV used to compute aggregate features.")
    parser.add_argument("--transaction-json", required=True, help="JSON string or file path for the transaction.")
    args = parser.parse_args()

    if Path(args.transaction_json).exists():
        transaction = json.loads(Path(args.transaction_json).read_text())
    else:
        transaction = json.loads(args.transaction_json)

    result = predict_transaction(transaction, model_path=Path(args.model), history_path=Path(args.history))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
