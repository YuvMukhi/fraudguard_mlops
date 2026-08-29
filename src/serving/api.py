"""
FastAPI Serving Layer for Real-Time Fraud Detection Inference.
Provides /predict, /predict/async, /predict/batch, /health, /models, and /queue/stats endpoints.
Integrated with DynamicBatcher for sub-millisecond request queuing and adaptive micro-batching.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.serving.model_loader import ModelRegistryLoader
from src.queue.async_queue import DynamicBatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.time()
loader: Optional[ModelRegistryLoader] = None
batcher: Optional[DynamicBatcher] = None


def run_inference_batch(features_list: List[Dict[str, float]], model_version: str) -> Dict[str, Any]:
    """Helper function to invoke model predict for batcher worker thread."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    return loader.predict(features=features_list, version_or_alias=model_version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler for FastAPI app and Async Queue."""
    global loader, batcher
    logger.info("Initializing FastAPI Serving Layer & Pre-loading Model Registry...")
    loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    
    # Pre-warm models
    try:
        loader.get_model("champion")
        loader.get_model("baseline")
        logger.info("Models pre-warmed successfully!")
    except Exception as e:
        logger.warning(f"Model pre-warm warning: {e}")

    # Initialize and start Dynamic Batcher
    batcher = DynamicBatcher(
        inference_fn=run_inference_batch,
        max_batch_size=64,
        max_latency_ms=10.0,
        model_version="champion"
    )
    await batcher.start()
    logger.info("Dynamic Batching Engine active!")

    yield

    logger.info("Shutting down Dynamic Batcher & FastAPI Service...")
    if batcher:
        await batcher.stop()


app = FastAPI(
    title="ML Fraud Detection Inference Platform",
    description="Production-Grade ML Inference Service with Async Queues & Dynamic Batching",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# Pydantic Schemas
# ==========================================

class TransactionItem(BaseModel):
    Time: float = Field(default=0.0, description="Seconds elapsed since first transaction")
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)
    Amount: float = Field(default=0.0, description="Transaction amount")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 412.0,
                "V1": -1.35, "V2": 1.2, "V3": -0.8, "V4": 2.5, "V5": -0.5,
                "V6": -0.2, "V7": -1.1, "V8": 0.4, "V9": -0.6, "V10": 3.0,
                "V11": 2.1, "V12": -1.5, "V13": 0.1, "V14": -3.5, "V15": 0.2,
                "V16": 0.8, "V17": -2.0, "V18": -0.3, "V19": 0.5, "V20": 0.1,
                "V21": 0.25, "V22": -0.1, "V23": 0.05, "V24": 0.3, "V25": -0.2,
                "V26": 0.15, "V27": 0.08, "V28": -0.02,
                "Amount": 149.99
            }
        }


class SinglePredictionRequest(BaseModel):
    transaction: TransactionItem
    model_version: Optional[str] = Field(default="champion", description="Model version or alias")
    threshold: Optional[float] = Field(default=None, description="Custom decision threshold override")


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionItem]
    model_version: Optional[str] = Field(default="champion", description="Model version or alias")
    threshold: Optional[float] = Field(default=None, description="Custom decision threshold override")


class PredictionResult(BaseModel):
    is_fraud: int = Field(description="Binary classification (1=fraud, 0=genuine)")
    fraud_probability: float = Field(description="Model probability score between 0.0 and 1.0")


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model_version: str
    threshold_used: float
    latency_ms: float


class AsyncPredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    model_version: str
    threshold_used: float
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    loaded_models: List[str]
    registry_connected: bool


# ==========================================
# Endpoints
# ==========================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health and readiness check endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    loaded_keys = list(loader.loaded_models.keys())
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="ok",
        uptime_seconds=round(uptime, 2),
        loaded_models=loaded_keys,
        registry_connected=True
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(request: SinglePredictionRequest):
    """Sync single transaction real-time fraud prediction endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    try:
        data_dict = request.transaction.model_dump()
        res = loader.predict(
            features=data_dict,
            version_or_alias=request.model_version or "champion",
            custom_threshold=request.threshold
        )

        results = [
            PredictionResult(is_fraud=pred, fraud_probability=prob)
            for pred, prob in zip(res["predictions"], res["probabilities"])
        ]

        return PredictionResponse(
            predictions=results,
            model_version=res["model_version"],
            threshold_used=res["threshold_used"],
            latency_ms=round(res["latency_ms"], 3)
        )
    except Exception as e:
        logger.error(f"Inference error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/async", response_model=AsyncPredictionResponse, tags=["Inference"])
async def predict_async(request: SinglePredictionRequest):
    """
    Decoupled Async Single Transaction Fraud Prediction.
    Enqueues incoming item into Dynamic Batcher micro-batch queue.
    """
    global batcher
    if batcher is None:
        batcher = DynamicBatcher(
            inference_fn=run_inference_batch,
            max_batch_size=64,
            max_latency_ms=10.0,
            model_version=request.model_version or "champion"
        )
        await batcher.start()

    try:
        data_dict = request.transaction.model_dump()
        result = await batcher.enqueue(data_dict, model_version=request.model_version or "champion")
        return AsyncPredictionResponse(
            is_fraud=result["is_fraud"],
            fraud_probability=round(result["fraud_probability"], 4),
            model_version=result["model_version"],
            threshold_used=result["threshold_used"],
            batch_size=result["batch_size"]
        )
    except Exception as e:
        logger.error(f"Async inference error in /predict/async: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """Batch transactions fraud prediction endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    if not request.transactions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Transactions list cannot be empty.")

    try:
        data_list = [t.model_dump() for t in request.transactions]
        res = loader.predict(
            features=data_list,
            version_or_alias=request.model_version or "champion",
            custom_threshold=request.threshold
        )

        results = [
            PredictionResult(is_fraud=pred, fraud_probability=prob)
            for pred, prob in zip(res["predictions"], res["probabilities"])
        ]

        return PredictionResponse(
            predictions=results,
            model_version=res["model_version"],
            threshold_used=res["threshold_used"],
            latency_ms=round(res["latency_ms"], 3)
        )
    except Exception as e:
        logger.error(f"Inference error in /predict/batch: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/queue/stats", tags=["Monitoring"])
def get_queue_stats():
    """Return live Dynamic Batcher and queue statistics."""
    global batcher
    if batcher is None:
        return {"status": "inactive"}
    return batcher.get_stats()


@app.get("/models", tags=["Registry"])
def list_models():
    """List available active model aliases and pinned versions."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    loader.get_model("champion")
    loader.get_model("baseline")

    models_info = {}
    for alias, (model_obj, version_str, thresh) in loader.loaded_models.items():
        models_info[alias] = {
            "resolved_version": version_str,
            "default_threshold": thresh,
            "type": type(model_obj).__name__
        }

    return {"active_models": models_info}
