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
REGISTRY_PATH = Path("models/model_registry.json")

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TransactionDate" in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    else:
        df["TransactionDate"] = pd.NaT
    return df

@st.cache_data
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.markdown("""
<style>
section.main header {display:none}
.stApp {background: linear-gradient(180deg, #eef2ff 0%, #ffffff 45%, #f8fafc 100%);} 
[data-testid="stSidebar"] {background: rgba(255,255,255,0.96) !important;}
.css-17lntkn.e1fqkh3o3 {border-radius: 1rem; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);}
.stButton>button {background-color: #2563eb; color: white;}
.stButton>button:hover {background-color: #1d4ed8;}
</style>
""", unsafe_allow_html=True)

st.markdown("# Financial Fraud Detection Dashboard")
st.markdown("### Interactive analytics, model metrics, and fraud monitoring in one polished view.")
st.markdown("---")

if not DATA_PATH.exists():
    st.warning("Processed dataset not found. Run preprocessing first.")
    st.stop()


df = load_dataset(DATA_PATH)
model_metrics = load_json(METRICS_PATH)
model_registry = load_json(REGISTRY_PATH)

fraud_rate = df["IsFraud"].mean()

st.sidebar.header("Dataset Summary")
st.sidebar.metric("Total transactions", len(df))
st.sidebar.metric("Fraud cases", int(df["IsFraud"].sum()))
st.sidebar.metric("Fraud rate", f"{fraud_rate:.2%}")

st.sidebar.markdown("---")
st.sidebar.header("Filters")
min_date = df["TransactionDate"].min()
max_date = df["TransactionDate"].max()
if pd.notnull(min_date) and pd.notnull(max_date):
    date_range = st.sidebar.date_input("Transaction date range", value=(min_date.date(), max_date.date()))
else:
    date_range = None

min_amt, max_amt = float(df["Amount"].min()), float(df["Amount"].max())
amount_range = st.sidebar.slider("Transaction amount range", min_value=0.0, max_value=max_amt, value=(min_amt, max_amt))

DEFAULT_TRANSACTION_TYPES = ["purchase", "refund", "withdrawal", "transfer"]
available_types = sorted(set(df["TransactionType"].dropna().unique().tolist()) | set(DEFAULT_TRANSACTION_TYPES))
select_all_types = st.sidebar.checkbox("Select all transaction types", value=True)
if select_all_types:
    st.sidebar.markdown("**All transaction types selected**")
    selected_types = available_types
else:
    selected_types = st.sidebar.multiselect(
        "Transaction types",
        options=available_types,
        default=[],
        key="transaction_type_selector",
    )

locations = sorted(df["Location"].dropna().unique().tolist())
select_all_locations = st.sidebar.checkbox("Select all locations", value=True)
if select_all_locations:
    st.sidebar.markdown("**All locations selected**")
    selected_locations = locations
else:
    selected_locations = st.sidebar.multiselect(
        "Locations",
        options=locations,
        default=[],
        key="location_selector",
    )

st.sidebar.markdown("---")
st.sidebar.header("Quick Insights")
if "best_supervised" in model_metrics:
    best = model_metrics["best_supervised"]["name"]
    st.sidebar.markdown(f"**Best supervised model:** {best}")
if model_registry.get("models"):
    latest = model_registry["models"][-1]
    st.sidebar.markdown(f"**Latest registry model:** {latest.get('model_name', 'N/A')} v{latest.get('version', 'N/A')}")

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

PALETTE = ["#2563eb", "#fb923c", "#14b8a6", "#8b5cf6", "#ef4444", "#0f766e", "#38bdf8", "#fbbf24"]
TEMPLATE = "plotly_white"

filtered = df.copy()
if date_range:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["TransactionDate"] >= start) & (filtered["TransactionDate"] <= end)]
if amount_range:
    filtered = filtered[(filtered["Amount"] >= amount_range[0]) & (filtered["Amount"] <= amount_range[1])]
if selected_types:
    filtered = filtered[filtered["TransactionType"].isin(selected_types)]
if selected_locations:
    filtered = filtered[filtered["Location"].isin(selected_locations)]

if filtered.empty:
    st.sidebar.error("No data matches filters. Adjust selections to view charts.")
    st.stop()

with tab1:
    st.header("Transaction Insights")
    kpi1, kpi2, kpi3 = st.columns([1,1,1])
    kpi1.metric("Total transactions", f"{len(filtered):,}")
    kpi2.metric("Fraud cases", f"{int(filtered['IsFraud'].sum()):,}")
    kpi3.metric("Fraud rate", f"{filtered['IsFraud'].mean():.2%}")

    col1, col2 = st.columns(2)
    with col1:
        monthly = (
            filtered.groupby("TransactionMonth")["IsFraud"].agg(total="size", fraud="sum").reset_index()
        )
        monthly["fraud_rate"] = monthly["fraud"] / monthly["total"]
        fig_month = px.line(monthly, x="TransactionMonth", y="fraud_rate", markers=True, title="Monthly Fraud Rate", template=TEMPLATE)
        fig_month.update_traces(line=dict(color=PALETTE[0], width=3))
        fig_month.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        st.subheader("Top Transaction Types")
        type_counts = filtered["TransactionType"].value_counts().reset_index()
        type_counts.columns = ["TransactionType", "Count"]
        fig_type = px.bar(type_counts, x="TransactionType", y="Count", title="Transaction Type Breakdown", color="TransactionType", template=TEMPLATE, color_discrete_sequence=PALETTE)
        fig_type.update_layout(showlegend=False, plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
        st.plotly_chart(fig_type, use_container_width=True)

    st.subheader("Transaction Type Fraud Distribution")
    type_fraud = filtered.groupby("TransactionType")["IsFraud"].sum().reset_index().sort_values("IsFraud", ascending=False)
    fig_pie = px.pie(type_fraud, names="TransactionType", values="IsFraud", title="Fraud Case Distribution by Transaction Type", template=TEMPLATE, color_discrete_sequence=PALETTE)
    fig_pie.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Transaction Amount Distribution")
    fig_hist = px.histogram(filtered, x="Amount", nbins=40, title="Transaction Amount Histogram", marginal="box", color="IsFraud", template=TEMPLATE, color_discrete_sequence=["#2563eb", "#ef4444"])
    fig_hist.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("High Risk Transactions")
    fig_scatter = px.scatter(filtered, x="Amount", y="RiskScore", size="Amount", color="IsFraud", hover_data=["TransactionType", "Location"], title="Amount vs Risk Score", template=TEMPLATE, color_discrete_map={0: "#2563eb", 1: "#ef4444"})
    fig_scatter.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Fraud Share by Location")
    fraud_by_location = (
        filtered.groupby("Location")["IsFraud"].mean().reset_index().sort_values("IsFraud", ascending=False).head(10)
    )
    fraud_by_location.columns = ["Location", "FraudRate"]
    fig_loc = px.bar(fraud_by_location, x="Location", y="FraudRate", title="Top Fraud Rate Locations", color="FraudRate", template=TEMPLATE, color_continuous_scale=px.colors.sequential.OrRd)
    fig_loc.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
    st.plotly_chart(fig_loc, use_container_width=True)

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
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale=px.colors.diverging.RdBu, template=TEMPLATE)
    fig.update_layout(height=600, plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Evaluation Dashboard")
    if model_registry.get("models"):
        versions = [f"v{item.get('version')} - {item.get('model_name')}" for item in model_registry["models"]]
        selected_version = st.selectbox("Select registry model", options=versions, index=len(versions) - 1)
        selected_index = versions.index(selected_version)
        selected_model = model_registry["models"][selected_index]
        st.markdown(f"**Registry version:** {selected_model.get('version')} — **{selected_model.get('model_name')}**")
        metrics = selected_model.get("metrics", {})
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
        cols[3].metric("F1", f"{metrics.get('f1', 0):.4f}")
        cols[4].metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        st.markdown("---")
        registry_df = pd.DataFrame(model_registry["models"]).sort_values("version")
        st.subheader("Model Registry History")
        st.dataframe(registry_df[["version", "timestamp", "model_name", "model_file"]])
    else:
        st.warning("No model registry found. Train the model to generate a registry entry.")

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
            fig_sup = px.scatter(supervised_melt, x="Model", y="Value", color="Metric", size="Value", title="Supervised Metrics Comparison", template=TEMPLATE, color_discrete_sequence=PALETTE)
            fig_sup.update_traces(mode="markers+lines")
            fig_sup.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
            st.plotly_chart(fig_sup, use_container_width=True)

        if "unsupervised" in model_metrics:
            unsupervised = model_metrics["unsupervised"]
            unsup_df = pd.DataFrame(unsupervised).T.reset_index().rename(columns={"index": "Model"})
            st.subheader("Unsupervised Anomaly Detection")
            st.dataframe(unsup_df[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]].sort_values("roc_auc", ascending=False))
            unsupervised_melt = unsup_df.melt(id_vars="Model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="Metric", value_name="Value")
            fig_unsup = px.scatter(unsupervised_melt, x="Model", y="Value", color="Metric", size="Value", title="Unsupervised Metrics Comparison", template=TEMPLATE, color_discrete_sequence=PALETTE)
            fig_unsup.update_traces(mode="markers+lines")
            fig_unsup.update_layout(plot_bgcolor="rgba(255,255,255,0.95)", paper_bgcolor="rgba(255,255,255,0.95)", title_font_color="#1e3a8a")
            st.plotly_chart(fig_unsup, use_container_width=True)

        if best_details:
            st.markdown(f"**Best supervised model:** {best_details.get('name', 'N/A')} with ROC AUC = {best_details.get('metrics', {}).get('roc_auc', 0):.4f}")
    elif not model_registry.get("models"):
        st.warning("No model metrics found. Train the model to visualize algorithm performance.")

with tab3:
    st.header("Anomaly & Risk Monitoring")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(filtered, x="RiskScore", nbins=40, title="Risk Score Distribution", color="IsFraud", template=TEMPLATE), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(filtered[filtered["IsFraud"] == 1], x="Amount", nbins=40, title="Fraudulent Transaction Amounts", color="TransactionType", template=TEMPLATE), use_container_width=True)

    st.plotly_chart(px.line(filtered.groupby("TransactionMonth")["IsFraud"].mean().reset_index(), x="TransactionMonth", y="IsFraud", title="Monthly Fraud Incidence", markers=True, template=TEMPLATE), use_container_width=True)
    st.plotly_chart(px.box(filtered, x="TransactionType", y="RiskScore", title="Risk Score by Transaction Type", template=TEMPLATE), use_container_width=True)

with tab4:
    st.header("Graph-Based Features")
    if all(col in filtered.columns for col in ["merchant_degree", "location_degree", "type_degree"]):
        top_merchants = filtered.groupby("MerchantID")["merchant_degree"].mean().reset_index().sort_values("merchant_degree", ascending=False).head(10)
        st.subheader("Top merchant degree")
        st.plotly_chart(px.bar(top_merchants, x="MerchantID", y="merchant_degree", title="Merchant Graph Degree", template=TEMPLATE, color_discrete_sequence=PALETTE), use_container_width=True)

        st.subheader("Risk vs Graph Centrality")
        st.plotly_chart(px.scatter(filtered, x="merchant_betweenness", y="merchant_fraud_rate", color="IsFraud", title="Merchant Betweenness vs Fraud Rate", template=TEMPLATE, color_discrete_map={0: "#2563eb", 1: "#ef4444"}), use_container_width=True)
    else:
        st.info("Graph features are not available in the current dataset. Run preprocessing with graph feature extraction enabled.")
