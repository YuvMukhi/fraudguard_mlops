"""
Prometheus Observability & Metrics Registry for ML Inference Platform.
Exposes prediction latency percentiles, error rates, model drift scores, and request throughput.
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# 1. Latency Histograms
LATENCY_HISTOGRAM = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in Seconds",
    ["endpoint", "method", "status_code"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# 2. Request Counters
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP Requests Count",
    ["endpoint", "method", "status_code"]
)

ERROR_COUNTER = Counter(
    "http_request_errors_total",
    "Total HTTP Request Errors Count",
    ["endpoint", "error_type"]
)

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total Model Inferences Executed",
    ["model_version", "prediction_label"]
)

# 3. Model Drift Gauges
FEATURE_PSI_GAUGE = Gauge(
    "model_feature_drift_psi",
    "Feature Population Stability Index (PSI)",
    ["feature_name"]
)

FEATURE_KS_PVALUE_GAUGE = Gauge(
    "model_feature_drift_ks_pvalue",
    "Feature Kolmogorov-Smirnov Test p-value",
    ["feature_name"]
)

MAX_MODEL_PSI_GAUGE = Gauge(
    "model_drift_max_psi",
    "Maximum PSI across all monitored features"
)

# 4. Queue Telemetry Gauges
QUEUE_DEPTH_GAUGE = Gauge(
    "async_queue_depth_current",
    "Current number of pending items in dynamic batch queue"
)

BATCH_SIZE_AVG_GAUGE = Gauge(
    "async_queue_batch_size_avg",
    "Average executed micro-batch size"
)


def update_drift_prometheus_metrics(drift_report: dict):
    """Update Prometheus Gauges with latest statistical drift detection results."""
    features = drift_report.get("features", {})
    for f_name, metrics in features.items():
        if "psi" in metrics:
            FEATURE_PSI_GAUGE.labels(feature_name=f_name).set(metrics["psi"])
        if "ks_pvalue" in metrics:
            FEATURE_KS_PVALUE_GAUGE.labels(feature_name=f_name).set(metrics["ks_pvalue"])

    if "max_psi" in drift_report:
        MAX_MODEL_PSI_GAUGE.set(drift_report["max_psi"])
