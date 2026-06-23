import argparse
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from ml_pipeline.etl import persist_to_sqlite
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from etl import persist_to_sqlite


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    logging.info("Loading raw data from %s", csv_path)
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data cleaning")

    df = df.copy()
    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()
    df = df.dropna(subset=["TransactionID", "TransactionDate", "Amount", "TransactionType", "Location", "IsFraud"])

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["TransactionDate"])
    df["TransactionDate"] = df["TransactionDate"].dt.tz_localize(None)

    df["IsFraud"] = df["IsFraud"].astype(int)
    df["MerchantID"] = df["MerchantID"].astype(str)
    df["TransactionType"] = df["TransactionType"].astype(str).str.lower().str.strip()
    df["Location"] = df["Location"].astype(str).str.title().str.strip()

    df["TransactionDay"] = df["TransactionDate"].dt.day
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionWeekday"] = df["TransactionDate"].dt.weekday
    df["TransactionQuarter"] = df["TransactionDate"].dt.quarter
    df["TransactionYear"] = df["TransactionDate"].dt.year

    df["AmountLog"] = np.log1p(df["Amount"])
    df["HighAmount"] = (df["Amount"] >= df["Amount"].quantile(0.90)).astype(int)

    df = add_aggregate_features(df)
    df = add_graph_features(df)
    df["RiskScore"] = _calculate_risk_score(df)

    logging.info("Data cleaning completed: %s rows", len(df))
    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    merchant_stats = (
        df.groupby("MerchantID")["IsFraud"]
        .agg(merchant_transaction_count="count", merchant_fraud_rate="mean")
        .reset_index()
    )

    location_stats = (
        df.groupby("Location")["IsFraud"]
        .agg(location_transaction_count="count", location_fraud_rate="mean")
        .reset_index()
    )

    type_stats = (
        df.groupby("TransactionType")["IsFraud"]
        .agg(type_transaction_count="count", type_fraud_rate="mean")
        .reset_index()
    )

    df = df.merge(merchant_stats, on="MerchantID", how="left")
    df = df.merge(location_stats, on="Location", how="left")
    df = df.merge(type_stats, on="TransactionType", how="left")

    df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0.0)
    df["location_fraud_rate"] = df["location_fraud_rate"].fillna(0.0)
    df["type_fraud_rate"] = df["type_fraud_rate"].fillna(0.0)

    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating graph-based features")

    edges = []
    for _, row in df.iterrows():
        merchant = f"merchant:{row['MerchantID']}"
        location = f"location:{row['Location']}"
        tx_type = f"type:{row['TransactionType']}"
        edges.append((merchant, location))
        edges.append((merchant, tx_type))
        edges.append((location, tx_type))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    merchant_degree = {}
    location_degree = {}
    type_degree = {}
    merchant_betweenness = {}
    location_betweenness = {}
    type_betweenness = {}

    bc = nx.betweenness_centrality(graph)

    for node, degree in graph.degree():
        if node.startswith("merchant:"):
            merchant_degree[node.split(":", 1)[1]] = degree
            merchant_betweenness[node.split(":", 1)[1]] = bc.get(node, 0.0)
        elif node.startswith("location:"):
            location_degree[node.split(":", 1)[1]] = degree
            location_betweenness[node.split(":", 1)[1]] = bc.get(node, 0.0)
        elif node.startswith("type:"):
            type_degree[node.split(":", 1)[1]] = degree
            type_betweenness[node.split(":", 1)[1]] = bc.get(node, 0.0)

    df["merchant_degree"] = df["MerchantID"].map(merchant_degree).fillna(0).astype(int)
    df["location_degree"] = df["Location"].map(location_degree).fillna(0).astype(int)
    df["type_degree"] = df["TransactionType"].map(type_degree).fillna(0).astype(int)
    df["merchant_betweenness"] = df["MerchantID"].map(merchant_betweenness).fillna(0.0)
    df["location_betweenness"] = df["Location"].map(location_betweenness).fillna(0.0)
    df["type_betweenness"] = df["TransactionType"].map(type_betweenness).fillna(0.0)

    return df


def _calculate_risk_score(df: pd.DataFrame) -> pd.Series:
    amount_score = (df["AmountLog"] - df["AmountLog"].min()) / (df["AmountLog"].max() - df["AmountLog"].min() + 1e-9)
    fraud_rate_score = (
        0.5 * df["merchant_fraud_rate"]
        + 0.3 * df["location_fraud_rate"]
        + 0.2 * df["type_fraud_rate"]
    )
    return (0.6 * amount_score + 0.4 * fraud_rate_score).round(4)


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    logging.info("Writing cleaned data to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %s rows to %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud detection data preprocessing and ETL.")
    parser.add_argument("--input-csv", default="data/credit_card_fraud_dataset.csv", help="Raw transaction CSV path.")
    parser.add_argument("--output-csv", default="data/processed_transactions.csv", help="Cleaned output CSV path.")
    parser.add_argument("--sqlite-db", default="database/fraud_detection.db", help="SQLite database path for ETL.")
    parser.add_argument("--sqlite-table", default="transactions", help="SQLite table name.")
    parser.add_argument("--no-sqlite", action="store_true", help="Skip persisting cleaned data to SQLite.")
    args = parser.parse_args()

    raw_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    sqlite_path = Path(args.sqlite_db)

    raw_df = load_raw_data(raw_path)
    processed_df = clean_data(raw_df)
    save_processed_data(processed_df, output_path)

    if not args.no_sqlite:
        persist_to_sqlite(processed_df, sqlite_path, args.sqlite_table)


if __name__ == "__main__":
    main()
