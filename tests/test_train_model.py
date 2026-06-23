from ml_pipeline.train_model import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, build_supervised_models, build_unsupervised_models


def test_supervised_models_keys():
    models = build_supervised_models()
    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    if XGBOOST_AVAILABLE:
        assert "xgboost" in models
    else:
        assert "xgboost" not in models
    if LIGHTGBM_AVAILABLE:
        assert "lightgbm" in models


def test_unsupervised_models_keys():
    models = build_unsupervised_models()
    assert "isolation_forest" in models
    assert "one_class_svm" in models
