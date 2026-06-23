import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

import pandas as pd
import plotly.express as px
import streamlit as st

from ml_pipeline.predict import predict_transaction

DATA_PATH = Path("data/processed_transactions.csv")
MODEL_PATH = Path("models/fraud_pipeline.joblib")
METRICS_PATH = Path("models/model_metrics.json")


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Financial Fraud Detection Dashboard")

if not DATA_PATH.exists():
    st.warning("Processed dataset not found. Run preprocessing first.")
    st.stop()


df = pd.read_csv(DATA_PATH)
model_metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as fh:
        model_metrics = json.load(fh)

fraud_rate = df["IsFraud"].mean()

st.sidebar.header("Dataset Summary")
st.sidebar.metric("Total transactions", len(df))
st.sidebar.metric("Fraud cases", int(df["IsFraud"].sum()))
st.sidebar.metric("Fraud rate", f"{fraud_rate:.2%}")

st.sidebar.markdown("---")
st.sidebar.header("Quick Insights")
if "best_supervised" in model_metrics:
    best = model_metrics["best_supervised"]["name"]
    st.sidebar.markdown(f"**Best supervised model:** {best}")

st.sidebar.markdown("---")
st.sidebar.header("Predict a new transaction")
with st.sidebar.form("transaction_form"):
    transaction_id = st.text_input("Transaction ID", value="100001")
    transaction_date = st.text_input("Transaction Date", value="23-06-2026")
    amount = st.number_input("Amount", min_value=0.0, value=199.0)
    merchant_id = st.text_input("Merchant ID", value="101")
    transaction_type = st.selectbox("Transaction Type", ["purchase", "refund", "withdrawal", "transfer"])
    location = st.text_input("Location", value="New York")
    submit = st.form_submit_button("Predict fraud")

    if submit:
        transaction_payload = {
            "TransactionID": int(transaction_id) if transaction_id.isdigit() else transaction_id,
            "TransactionDate": transaction_date,
            "Amount": float(amount),
            "MerchantID": merchant_id,
            "TransactionType": transaction_type,
            "Location": location,
        }

        if MODEL_PATH.exists():
            prediction = predict_transaction(transaction_payload)
            st.success(f"Fraud probability: {prediction['fraud_probability']:.4f}")
            st.json(prediction)
        else:
            st.error("Trained model not found. Train the model before using prediction.")


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Metrics", "Anomaly Analytics", "Graph Features"])

with tab1:
    st.header("Transaction Insights")
    col1, col2 = st.columns(2)
    with col1:
        monthly = (
            df.groupby("TransactionMonth")["IsFraud"].agg(total="size", fraud="sum").reset_index()
        )
        monthly["fraud_rate"] = monthly["fraud"] / monthly["total"]
        st.subheader("Monthly Fraud Rate")
        st.plotly_chart(px.line(monthly, x="TransactionMonth", y="fraud_rate", markers=True, title="Monthly Fraud Rate"), use_container_width=True)

    with col2:
        st.subheader("Top Transaction Types")
        type_counts = df["TransactionType"].value_counts().reset_index()
        type_counts.columns = ["TransactionType", "Count"]
        st.plotly_chart(px.bar(type_counts, x="TransactionType", y="Count", title="Transaction Type Breakdown", color="TransactionType"), use_container_width=True)

    st.subheader("Transaction Type Fraud Distribution")
    type_fraud = df.groupby("TransactionType")["IsFraud"].sum().reset_index().sort_values("IsFraud", ascending=False)
    st.plotly_chart(px.pie(type_fraud, names="TransactionType", values="IsFraud", title="Fraud Case Distribution by Transaction Type"), use_container_width=True)

    st.subheader("Transaction Amount Distribution")
    st.plotly_chart(px.histogram(df, x="Amount", nbins=40, title="Transaction Amount Histogram", marginal="box", color="IsFraud"), use_container_width=True)

    st.subheader("Fraud Share by Location")
    fraud_by_location = (
        df.groupby("Location")["IsFraud"].mean().reset_index().sort_values("IsFraud", ascending=False).head(10)
    )
    fraud_by_location.columns = ["Location", "FraudRate"]
    st.plotly_chart(px.bar(fraud_by_location, x="Location", y="FraudRate", title="Top Fraud Rate Locations", color="FraudRate"), use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_cols = [
        "Amount",
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
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Evaluation Dashboard")
    if model_metrics:
        best_details = model_metrics.get("best_supervised", {})
        if best_details:
            col1, col2 = st.columns(2)
            col1.metric("Best supervised model", best_details.get("name", "N/A"))
            col2.metric("Best ROC AUC", f"{best_details.get('metrics', {}).get('roc_auc', 0):.4f}")

        if "supervised" in model_metrics:
            supervised = model_metrics["supervised"]
            metric_df = pd.DataFrame(supervised).T.reset_index().rename(columns={"index": "Model"})
            st.subheader("Supervised Model Comparison")
            st.dataframe(metric_df[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]].sort_values("roc_auc", ascending=False))
            supervised_melt = metric_df.melt(id_vars="Model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="Metric", value_name="Value")
            st.plotly_chart(
                px.bar(supervised_melt, x="Model", y="Value", color="Metric", barmode="group", title="Supervised Metrics Comparison"),
                use_container_width=True,
            )

        if "unsupervised" in model_metrics:
            unsupervised = model_metrics["unsupervised"]
            unsup_df = pd.DataFrame(unsupervised).T.reset_index().rename(columns={"index": "Model"})
            st.subheader("Unsupervised Anomaly Detection")
            st.dataframe(unsup_df[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]].sort_values("roc_auc", ascending=False))
            unsupervised_melt = unsup_df.melt(id_vars="Model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="Metric", value_name="Value")
            st.plotly_chart(
                px.bar(unsupervised_melt, x="Model", y="Value", color="Metric", barmode="group", title="Unsupervised Metrics Comparison"),
                use_container_width=True,
            )

        if best_details:
            st.markdown(f"**Best supervised model:** {best_details.get('name', 'N/A')} with ROC AUC = {best_details.get('metrics', {}).get('roc_auc', 0):.4f}")
    else:
        st.warning("No model metrics found. Train the model to visualize algorithm performance.")

with tab3:
    st.header("Anomaly & Risk Monitoring")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="RiskScore", nbins=40, title="Risk Score Distribution", color="IsFraud"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df[df["IsFraud"] == 1], x="Amount", nbins=40, title="Fraudulent Transaction Amounts", color="TransactionType"), use_container_width=True)

    st.plotly_chart(px.line(df.groupby("TransactionMonth")["IsFraud"].mean().reset_index(), x="TransactionMonth", y="IsFraud", title="Monthly Fraud Incidence", markers=True), use_container_width=True)
    st.plotly_chart(px.box(df, x="TransactionType", y="RiskScore", title="Risk Score by Transaction Type"), use_container_width=True)

with tab4:
    st.header("Graph-Based Features")
    if all(col in df.columns for col in ["merchant_degree", "location_degree", "type_degree"]):
        top_merchants = df.groupby("MerchantID")["merchant_degree"].mean().reset_index().sort_values("merchant_degree", ascending=False).head(10)
        st.subheader("Top merchant degree")
        st.plotly_chart(px.bar(top_merchants, x="MerchantID", y="merchant_degree", title="Merchant Graph Degree"), use_container_width=True)

        st.subheader("Risk vs Graph Centrality")
        st.plotly_chart(px.scatter(df, x="merchant_betweenness", y="merchant_fraud_rate", color="IsFraud", title="Merchant Betweenness vs Fraud Rate"), use_container_width=True)
    else:
        st.info("Graph features are not available in the current dataset. Run preprocessing with graph feature extraction enabled.")
