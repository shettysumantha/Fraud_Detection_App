import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ml_pipeline.predict import predict_transaction
from ml_pipeline.train_model import MODEL_PATH, MODEL_METRICS_PATH, MODEL_REGISTRY_PATH, train_model

router = APIRouter()

METRICS_PATH = MODEL_METRICS_PATH
REGISTRY_PATH = MODEL_REGISTRY_PATH
MODEL_FILE_PATH = MODEL_PATH


class TransactionRequest(BaseModel):
    TransactionID: int = Field(..., example=12345)
    TransactionDate: str = Field(..., example="23-06-2026")
    Amount: float = Field(..., example=1299.75)
    MerchantID: str = Field(..., example="500")
    TransactionType: str = Field(..., example="purchase")
    Location: str = Field(..., example="San Francisco")


class PredictionResponse(BaseModel):
    fraud_probability: float
    label: int
    input: dict


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def api_predict(transaction: TransactionRequest) -> PredictionResponse:
    try:
        result = predict_transaction(transaction.dict())
        return PredictionResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/metrics")
def api_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/registry")
def api_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise HTTPException(status_code=404, detail="Registry file not found")
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/artifacts")
def api_artifacts() -> dict:
    """Return presence and sizes of model artifacts for debugging."""
    info = {}
    for key, path in (("model", MODEL_FILE_PATH), ("metrics", METRICS_PATH), ("registry", REGISTRY_PATH)):
        try:
            exists = path.exists()
            size = path.stat().st_size if exists else 0
        except Exception:
            exists = False
            size = 0
        info[key] = {"path": str(path), "exists": exists, "size": size}
    return info


def retrain_model(rf_estimators: int, n_jobs: int) -> dict:
    # Spawn a background process to run the training CLI to avoid blocking the server
    import subprocess
    import sys

    MODEL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_path = MODEL_FILE_PATH.parent / "retrain.log"
    pid_path = MODEL_FILE_PATH.parent / "retrain.pid"

    cmd = [
        sys.executable,
        "-m",
        "ml_pipeline.train_model",
        "--input-csv",
        "data/processed_transactions.csv",
        "--output-model",
        str(MODEL_FILE_PATH),
        "--output-metrics",
        str(METRICS_PATH),
        "--rf-estimators",
        str(rf_estimators),
        "--n-jobs",
        str(n_jobs),
    ]

    with open(log_path, "ab") as lf:
        process = subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=str(Path(".")))
        # write pid for status checks
        pid_path.write_text(str(process.pid), encoding="utf-8")

    return {"status": "retrain_started", "pid": process.pid, "log": str(log_path)}


@router.post("/retrain")
def api_retrain(
    rf_estimators: int = Query(50, ge=1, description="Number of trees for RandomForest during retraining."),
    n_jobs: int = Query(1, ge=1, description="Parallel jobs for retraining."),
    background_tasks: BackgroundTasks = None,
) -> dict:
    try:
        if background_tasks is not None:
            background_tasks.add_task(retrain_model, rf_estimators, n_jobs)
            return {"status": "retrain_scheduled"}
        # fallback: run synchronously if no background tasks provided
        return retrain_model(rf_estimators, n_jobs)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/retrain")
def api_retrain_get(
    rf_estimators: int = Query(50, ge=1, description="Number of trees for RandomForest during retraining."),
    n_jobs: int = Query(1, ge=1, description="Parallel jobs for retraining."),
) -> dict:
    try:
        # schedule background retrain when called via browser/GET
        return retrain_model(rf_estimators, n_jobs)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
