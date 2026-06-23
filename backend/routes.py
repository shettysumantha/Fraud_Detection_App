import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml_pipeline.predict import predict_transaction

router = APIRouter()

METRICS_PATH = Path("models/model_metrics.json")


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
