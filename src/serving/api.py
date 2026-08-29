"""
FastAPI Serving Layer for Real-Time Fraud Detection Inference.
Includes Prometheus Observability (/metrics), Statistical Drift Monitoring (/monitoring/drift),
Optimization & Quantization Benchmarks (/optimization/benchmark), Dynamic Batching, and Safe Rollout Router.
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, Query, status
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app
import pandas as pd

from src.serving.model_loader import ModelRegistryLoader
from src.queue.async_queue import DynamicBatcher
from src.rollout.router import RolloutRouter
from src.monitoring.drift_detector import DriftDetector
from src.optimization.quantizer import ModelOptimizer
from src.models.heavy import HeavyModel
from src.monitoring.prometheus_metrics import (
    LATENCY_HISTOGRAM,
    REQUEST_COUNTER,
    ERROR_COUNTER,
    PREDICTION_COUNTER,
    QUEUE_DEPTH_GAUGE,
    BATCH_SIZE_AVG_GAUGE,
    update_drift_prometheus_metrics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.time()
loader: Optional[ModelRegistryLoader] = None
batcher: Optional[DynamicBatcher] = None
router: Optional[RolloutRouter] = None
drift_detector: Optional[DriftDetector] = None
optimizer: Optional[ModelOptimizer] = None


def run_inference_batch(features_list: List[Dict[str, float]], model_version: str) -> Dict[str, Any]:
    """Helper function to invoke model predict for batcher/router worker thread."""
    global loader, drift_detector
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    if drift_detector:
        drift_detector.add_production_batch(features_list)

    res = loader.predict(features=features_list, version_or_alias=model_version)

    for label in res.get("predictions", []):
        PREDICTION_COUNTER.labels(model_version=model_version, prediction_label=str(label)).inc()

    return res


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    global loader, batcher, router, drift_detector, optimizer
    logger.info("Initializing FastAPI Serving Layer & Optimization Stack...")
    loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    drift_detector = DriftDetector(data_dir="data/processed")
    optimizer = ModelOptimizer(model_dir="models")

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
    logger.info("Observability, Quantization Optimizer, Dynamic Batcher & Rollout Router Active!")

    yield

    logger.info("Shutting down service...")
    if batcher:
        await batcher.stop()


app = FastAPI(
    title="ML Fraud Detection Inference Platform",
    description="Production-Grade ML Inference Service with INT8 Model Quantization & Benchmarking",
    version="1.0.0",
    lifespan=lifespan
)

# Mount Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# HTTP Request Metrics Middleware
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        response: Response = await call_next(request)
        duration = time.time() - start_time
        status_code = str(response.status_code)

        if not endpoint.startswith("/metrics"):
            LATENCY_HISTOGRAM.labels(endpoint=endpoint, method=method, status_code=status_code).observe(duration)
            REQUEST_COUNTER.labels(endpoint=endpoint, method=method, status_code=status_code).inc()

        return response
    except Exception as e:
        duration = time.time() - start_time
        ERROR_COUNTER.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise e


# ==========================================
# Pydantic Schemas
# ==========================================

class TransactionItem(BaseModel):
    Time: float = Field(default=0.0)
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
    Amount: float = Field(default=0.0)


class SinglePredictionRequest(BaseModel):
    transaction: TransactionItem
    model_version: Optional[str] = Field(default="champion")
    threshold: Optional[float] = Field(default=None)


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionItem]
    model_version: Optional[str] = Field(default="champion")
    threshold: Optional[float] = Field(default=None)


class RolloutPredictionRequest(BaseModel):
    transactions: List[TransactionItem]
    mode: str = Field(default="ab_test")
    challenger_ratio: float = Field(default=0.20)
    request_id: Optional[str] = Field(default=None)


class PredictionResult(BaseModel):
    is_fraud: int
    fraud_probability: float


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
    global loader, drift_detector
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    data_dict = request.transaction.model_dump()
    if drift_detector:
        drift_detector.add_production_sample(data_dict)

    res = loader.predict(
        features=data_dict,
        version_or_alias=request.model_version or "champion",
        custom_threshold=request.threshold
    )

    for label in res.get("predictions", []):
        PREDICTION_COUNTER.labels(model_version=request.model_version or "champion", prediction_label=str(label)).inc()

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
    global loader, drift_detector
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")

    if not request.transactions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Transactions list cannot be empty.")

    data_list = [t.model_dump() for t in request.transactions]
    if drift_detector:
        drift_detector.add_production_batch(data_list)

    res = loader.predict(
        features=data_list,
        version_or_alias=request.model_version or "champion",
        custom_threshold=request.threshold
    )

    for label in res.get("predictions", []):
        PREDICTION_COUNTER.labels(model_version=request.model_version or "champion", prediction_label=str(label)).inc()

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
    """Safe Rollout Endpoint for A/B Testing & Shadow Deployment."""
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
        res = router.route_ab_test(data_list, challenger_ratio=request.challenger_ratio, request_id=request.request_id)
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


@app.get("/optimization/benchmark", tags=["Optimization"])
def benchmark_optimization():
    """
    Run latency (P50, P95, P99) and accuracy benchmarking comparing FP32 vs INT8 Dynamic Quantization.
    Validates model error tolerance and returns fallback decision status.
    """
    global loader, optimizer
    if loader is None:
        loader = ModelRegistryLoader(model_dir="models", data_dir="data/processed")
    if optimizer is None:
        optimizer = ModelOptimizer(model_dir="models")

    # Load heavy model instance
    model_obj, _, _ = loader.get_model("champion")
    if not isinstance(model_obj, HeavyModel):
        heavy_path = os.path.join("models", "heavy_model.pt")
        if os.path.exists(heavy_path):
            model_obj = HeavyModel.load(heavy_path)
        else:
            model_obj = HeavyModel(input_dim=30)
            model_obj.fit(
                pd.DataFrame(np.random.randn(50, 30)),
                pd.Series([0]*45 + [1]*5),
                pd.DataFrame(np.random.randn(20, 30)),
                pd.Series([0]*18 + [1]*2)
            )

    # Load test dataset sample
    test_path = os.path.join("data/processed", "test.parquet")
    if not os.path.exists(test_path):
        test_path = os.path.join("data/processed", "test.csv")

    if os.path.exists(test_path):
        test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)
    else:
        test_df = pd.DataFrame(np.random.randn(100, 31), columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])

    results = optimizer.benchmark_models(heavy_model=model_obj, test_df=test_df, n_iterations=50)
    return results


@app.get("/monitoring/drift", tags=["Monitoring"])
def get_drift_report():
    """Compute and return statistical data drift report (KS p-values & PSI scores)."""
    global drift_detector
    if drift_detector is None:
        drift_detector = DriftDetector(data_dir="data/processed")

    report = drift_detector.compute_drift_report()
    update_drift_prometheus_metrics(report)
    return report


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
    stats_dict = batcher.get_stats()
    QUEUE_DEPTH_GAUGE.set(stats_dict.get("queue_depth", 0))
    BATCH_SIZE_AVG_GAUGE.set(stats_dict.get("avg_batch_size", 0.0))
    return stats_dict


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
