import pandas as pd

from ml_pipeline.predict import FEATURE_COLUMNS, build_input_dataframe, compute_high_amount_threshold


def test_feature_columns_include_graph_features():
    assert "merchant_degree" in FEATURE_COLUMNS
    assert "location_degree" in FEATURE_COLUMNS
    assert "type_degree" in FEATURE_COLUMNS
    assert "merchant_betweenness" in FEATURE_COLUMNS
    assert "location_betweenness" in FEATURE_COLUMNS
    assert "type_betweenness" in FEATURE_COLUMNS


def test_build_input_dataframe_with_empty_history():
    data = {
        "TransactionID": 1,
        "TransactionDate": "23-06-2026",
        "Amount": 100.0,
        "MerchantID": "M1",
        "TransactionType": "purchase",
        "Location": "New York",
    }
    history_df = pd.DataFrame(columns=["MerchantID", "TransactionType", "Location", "Amount", "IsFraud"])
    X = build_input_dataframe(data, history_df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "HighAmount"] == 0
    assert X.loc[0, "merchant_transaction_count"] == 0
    assert X.loc[0, "location_transaction_count"] == 0
    assert X.loc[0, "type_transaction_count"] == 0


def test_compute_high_amount_threshold_no_history():
    empty_history = pd.DataFrame()
    threshold = compute_high_amount_threshold(empty_history)
    assert threshold == 0.0
