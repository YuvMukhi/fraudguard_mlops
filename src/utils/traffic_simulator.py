"""
Bursty Traffic Simulator Script.
Generates Poisson and bursty request patterns to test real-time inference throughput,
latency percentiles (P50, P95, P99), and dynamic batching performance.
"""
import time
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import httpx

from src.data.pipeline import FraudDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrafficSimulator:
    """Simulates real-time bursty transaction traffic against serving endpoint or dynamic queue."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        n_requests: int = 500,
        concurrency: int = 20,
        burst_spike_prob: float = 0.1,
        fraud_ratio: float = 0.05
    ):
        self.base_url = base_url
        self.n_requests = n_requests
        self.concurrency = concurrency
        self.burst_spike_prob = burst_spike_prob
        self.fraud_ratio = fraud_ratio
        
        self.generator = FraudDataGenerator(n_samples=max(100, n_requests), positive_rate=fraud_ratio, random_state=42)
        self.sample_data = self.generator.generate()
        self.feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def _get_random_transaction(self) -> Dict[str, float]:
        """Fetch a single random feature dictionary."""
        row = self.sample_data.sample(n=1).iloc[0]
        return {col: float(row[col]) for col in self.feature_cols}

    async def _send_single_request(self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Send a single prediction request over async HTTP client."""
        async with semaphore:
            payload = {
                "transaction": self._get_random_transaction(),
                "model_version": "champion"
            }
            
            # Simulate burst jitter
            if random.random() < self.burst_spike_prob:
                await asyncio.sleep(random.uniform(0.001, 0.005))

            t0 = time.time()
            try:
                response = await client.post(f"{self.base_url}/predict", json=payload, timeout=10.0)
                latency_ms = (time.time() - t0) * 1000.0
                if response.status_code == 200:
                    data = response.json()
                    is_fraud = data["predictions"][0]["is_fraud"]
                    return {"status": 200, "latency_ms": latency_ms, "is_fraud": is_fraud}
                else:
                    return {"status": response.status_code, "latency_ms": latency_ms, "is_fraud": 0}
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": 500, "latency_ms": latency_ms, "is_fraud": 0, "error": str(e)}

    async def run_simulation(self) -> Dict[str, Any]:
        """Run load simulation with concurrent async HTTP workers."""
        logger.info(f"Starting Traffic Simulator: {self.n_requests} requests, concurrency={self.concurrency}, target={self.base_url}")
        semaphore = asyncio.Semaphore(self.concurrency)

        start_time = time.time()
        async with httpx.AsyncClient() as client:
            tasks = [self._send_single_request(client, semaphore) for _ in range(self.n_requests)]
            results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        successful_requests = [r for r in results if r["status"] == 200]
        latencies = [r["latency_ms"] for r in successful_requests]
        fraud_count = sum(r["is_fraud"] for r in successful_requests)

        rps = len(successful_requests) / total_time if total_time > 0 else 0.0
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

        summary = {
            "total_requests_sent": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(results) - len(successful_requests),
            "throughput_rps": round(rps, 2),
            "total_duration_sec": round(total_time, 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "fraud_detected_count": fraud_count
        }

        logger.info("Traffic Simulation Complete Summary:")
        logger.info(f"  Throughput: {summary['throughput_rps']} RPS over {summary['total_duration_sec']}s")
        logger.info(f"  Latency P50: {summary['latency_p50_ms']}ms | P95: {summary['latency_p95_ms']}ms | P99: {summary['latency_p99_ms']}ms")
        logger.info(f"  Fraud detected: {fraud_count}/{len(successful_requests)}")

        return summary


if __name__ == "__main__":
    sim = TrafficSimulator(n_requests=100, concurrency=10)
    # Note: Requires FastAPI server running on localhost:8000
    try:
        asyncio.run(sim.run_simulation())
    except Exception as e:
        logger.warning(f"Simulator ran standalone without active server: {e}")
