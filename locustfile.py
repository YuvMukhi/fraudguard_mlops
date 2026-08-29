"""
Locust Load Testing Suite for ML Inference Platform.
Simulates real-time bursty transaction traffic across /predict, /predict/async, /predict/batch, and /predict/rollout endpoints.
"""
import random
from locust import HttpUser, task, between


class FraudInferenceUser(HttpUser):
    """Simulates realistic client traffic against the ML inference service."""

    host = "http://localhost:8000"
    wait_time = between(0.01, 0.1)  # High throughput request rate

    def on_start(self):
        """Prepare sample transaction payload."""
        self.sample_transaction = {
            "Time": float(random.randint(0, 100000)),
            "V1": random.uniform(-2.0, 2.0),
            "V2": random.uniform(-2.0, 2.0),
            "V3": random.uniform(-2.0, 2.0),
            "V4": random.uniform(-2.0, 2.0),
            "V5": random.uniform(-2.0, 2.0),
            "V6": random.uniform(-2.0, 2.0),
            "V7": random.uniform(-2.0, 2.0),
            "V8": random.uniform(-2.0, 2.0),
            "V9": random.uniform(-2.0, 2.0),
            "V10": random.uniform(-2.0, 2.0),
            "V11": random.uniform(-2.0, 2.0),
            "V12": random.uniform(-2.0, 2.0),
            "V13": random.uniform(-2.0, 2.0),
            "V14": random.uniform(-2.0, 2.0),
            "V15": random.uniform(-2.0, 2.0),
            "V16": random.uniform(-2.0, 2.0),
            "V17": random.uniform(-2.0, 2.0),
            "V18": random.uniform(-2.0, 2.0),
            "V19": random.uniform(-2.0, 2.0),
            "V20": random.uniform(-2.0, 2.0),
            "V21": random.uniform(-2.0, 2.0),
            "V22": random.uniform(-2.0, 2.0),
            "V23": random.uniform(-2.0, 2.0),
            "V24": random.uniform(-2.0, 2.0),
            "V25": random.uniform(-2.0, 2.0),
            "V26": random.uniform(-2.0, 2.0),
            "V27": random.uniform(-2.0, 2.0),
            "V28": random.uniform(-2.0, 2.0),
            "Amount": round(random.uniform(1.0, 1500.0), 2)
        }

    @task(50)
    def predict_single(self):
        """Single real-time transaction inference endpoint."""
        payload = {
            "transaction": self.sample_transaction,
            "model_version": "champion"
        }
        self.client.post("/predict", json=payload, name="/predict (single)")

    @task(25)
    def predict_async(self):
        """Decoupled async dynamic batching endpoint."""
        payload = {
            "transaction": self.sample_transaction,
            "model_version": "champion"
        }
        self.client.post("/predict/async", json=payload, name="/predict/async (dynamic batch)")

    @task(15)
    def predict_batch(self):
        """Batch transactions inference endpoint."""
        payload = {
            "transactions": [self.sample_transaction] * 10,
            "model_version": "champion"
        }
        self.client.post("/predict/batch", json=payload, name="/predict/batch (x10)")

    @task(10)
    def predict_rollout(self):
        """A/B and Shadow rollout inference endpoint."""
        mode = "shadow" if random.random() < 0.3 else "ab_test"
        payload = {
            "transactions": [self.sample_transaction],
            "mode": mode,
            "challenger_ratio": 0.20
        }
        self.client.post("/predict/rollout", json=payload, name=f"/predict/rollout ({mode})")

    @task(5)
    def health_and_monitoring(self):
        """Health check and drift monitoring telemetry."""
        self.client.get("/health", name="/health")
        self.client.get("/monitoring/drift", name="/monitoring/drift")
