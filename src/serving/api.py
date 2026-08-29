"""
FastAPI Serving Layer for Real-Time Fraud Detection Inference.
Provides /predict, /predict/async, /predict/batch, /predict/rollout, /health, /models, /queue/stats, and /rollout/stats.
Integrates Dynamic Batcher and RolloutRouter for A/B testing and Shadow deployments.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.serving.model_loader import ModelRegistryLoader
from src.queue.async_queue import DynamicBatcher
from src.rollout.router import RolloutRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.time()
loader: Optional[ModelRegistryLoader] = None
batcher: Optional[DynamicBatcher] = None
router: Optional[RolloutRouter] = None


def run_inference_batch(features_list: List[Dict[str, float]], model_version: str) -> Dict[str, Any]:
    """Helper function to invoke model predict for batcher/router worker thread."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    return loader.predict(features=features_list, version_or_alias=model_version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    global loader, batcher, router
    logger.info("Initializing FastAPI Serving Layer & Pre-loading Model Registry...")
    loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    
    # Pre-warm models
    try:
        loader.get_model("champion")
        loader.get_model("baseline")
        logger.info("Models pre-warmed successfully!")
    except Exception as e:
        logger.warning(f"Model pre-warm warning: {e}")

    # Initialize Dynamic Batcher
    batcher = DynamicBatcher(
        inference_fn=run_inference_batch,
        max_batch_size=64,
        max_latency_ms=10.0,
        model_version="champion"
    )
    await batcher.start()

    # Initialize Rollout Router
    router = RolloutRouter(
        predict_fn=run_inference_batch,
        champion_version="champion",
        challenger_version="baseline"
    )
    logger.info("Rollout Router & Dynamic Batcher active!")

    yield

    logger.info("Shutting down service...")
    if batcher:
        await batcher.stop()


app = FastAPI(
    title="ML Fraud Detection Inference Platform",
    description="Production-Grade ML Inference Service with A/B Testing & Shadow Deployments",
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


class SinglePredictionRequest(BaseModel):
    transaction: TransactionItem
    model_version: Optional[str] = Field(default="champion", description="Model version or alias")
    threshold: Optional[float] = Field(default=None, description="Custom decision threshold override")


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionItem]
    model_version: Optional[str] = Field(default="champion", description="Model version or alias")
    threshold: Optional[float] = Field(default=None, description="Custom decision threshold override")


class RolloutPredictionRequest(BaseModel):
    transactions: List[TransactionItem]
    mode: str = Field(default="ab_test", description="Rollout mode: 'ab_test' or 'shadow'")
    challenger_ratio: float = Field(default=0.20, description="Percentage of traffic routed to challenger in A/B test (0.0 to 1.0)")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for deterministic hash routing")


class PredictionResult(BaseModel):
    is_fraud: int = Field(description="Binary classification (1=fraud, 0=genuine)")
    fraud_probability: float = Field(description="Model probability score between 0.0 and 1.0")


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model_version: str
    threshold_used: float
    latency_ms: float


class RolloutPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model_version: str
    routing_mode: str
    selected_variant: Optional[str]
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
    """Health check endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - START_TIME, 2),
        loaded_models=list(loader.loaded_models.keys()),
        registry_connected=True
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(request: SinglePredictionRequest):
    """Sync single transaction prediction endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

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


@app.post("/predict/async", response_model=AsyncPredictionResponse, tags=["Inference"])
async def predict_async(request: SinglePredictionRequest):
    """Decoupled async prediction endpoint."""
    global batcher
    if batcher is None:
        batcher = DynamicBatcher(inference_fn=run_inference_batch, max_batch_size=64, max_latency_ms=10.0)
        await batcher.start()

    data_dict = request.transaction.model_dump()
    result = await batcher.enqueue(data_dict, model_version=request.model_version or "champion")
    return AsyncPredictionResponse(
        is_fraud=result["is_fraud"],
        fraud_probability=round(result["fraud_probability"], 4),
        model_version=result["model_version"],
        threshold_used=result["threshold_used"],
        batch_size=result["batch_size"]
    )


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """Batch transactions prediction endpoint."""
    global loader
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    if not request.transactions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Transactions list cannot be empty.")

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


@app.post("/predict/rollout", response_model=RolloutPredictionResponse, tags=["Rollout"])
def predict_rollout(request: RolloutPredictionRequest):
    """
    Safe Rollout Endpoint for A/B Testing & Shadow Deployment.
    Supports 'ab_test' (traffic split) and 'shadow' (silent secondary prediction divergence tracking).
    """
    global router
    if router is None:
        router = RolloutRouter(predict_fn=run_inference_batch)

    if not request.transactions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Transactions list cannot be empty.")

    t0 = time.time()
    data_list = [t.model_dump() for t in request.transactions]

    if request.mode.lower() == "shadow":
        res = router.route_shadow(data_list)
        selected_variant = "champion_primary_with_shadow_challenger"
    else:
        res = router.route_ab_test(
            data_list,
            challenger_ratio=request.challenger_ratio,
            request_id=request.request_id
        )
        selected_variant = res.get("selected_variant", "champion")

    results = [
        PredictionResult(is_fraud=pred, fraud_probability=prob)
        for pred, prob in zip(res["predictions"], res["probabilities"])
    ]

    latency_ms = (time.time() - t0) * 1000.0

    return RolloutPredictionResponse(
        predictions=results,
        model_version=res.get("model_version", "champion"),
        routing_mode=request.mode,
        selected_variant=selected_variant,
        threshold_used=res.get("threshold_used", 0.5),
        latency_ms=round(latency_ms, 3)
    )


@app.get("/rollout/stats", tags=["Rollout"])
def get_rollout_stats():
    """Return live A/B traffic split ratio and shadow divergence metrics."""
    global router
    if router is None:
        router = RolloutRouter(predict_fn=run_inference_batch)
    return router.get_rollout_stats()


@app.get("/queue/stats", tags=["Monitoring"])
def get_queue_stats():
    """Return live Dynamic Batcher statistics."""
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
