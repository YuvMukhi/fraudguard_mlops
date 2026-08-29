"""
Async Request Queue & Dynamic Batching Engine.
Decouples ingestion from inference by micro-batching incoming requests based on max batch size or latency window.
"""
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RequestItem:
    """Represents a single request item enqueued in the dynamic batcher."""

    def __init__(self, features: Dict[str, float], future: asyncio.Future, timestamp: float = None):
        self.features = features
        self.future = future
        self.timestamp = timestamp or time.time()


class DynamicBatcher:
    """
    High-performance Dynamic Batching Engine.
    Collects requests up to `max_batch_size` or until `max_latency_ms` expires.
    """

    def __init__(
        self,
        inference_fn: Callable[[List[Dict[str, float]], str], Dict[str, Any]],
        max_batch_size: int = 64,
        max_latency_ms: float = 10.0,
        model_version: str = "champion"
    ):
        self.inference_fn = inference_fn
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.model_version = model_version
        
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        
        # Statistics counters
        self.total_processed_requests: int = 0
        self.total_batches_executed: int = 0
        self.batch_sizes_history: List[int] = []

    async def start(self):
        """Start the background batch processing loop."""
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._batch_processing_loop())
            logger.info(f"DynamicBatcher started (max_batch_size={self.max_batch_size}, max_latency_ms={self.max_latency_ms}ms)")

    async def stop(self):
        """Stop the batch processing loop cleanly."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("DynamicBatcher stopped.")

    async def enqueue(self, features: Dict[str, float], model_version: Optional[str] = None) -> Dict[str, Any]:
        """Enqueue a request item and await its prediction result future."""
        if not self._is_running:
            await self.start()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = RequestItem(features=features, future=future)
        
        await self.queue.put(item)
        return await future

    async def _batch_processing_loop(self):
        """Background loop that collects items and triggers batch inference."""
        while self._is_running:
            try:
                # Wait for the first item in the batch
                first_item = await self.queue.get()
                batch: List[RequestItem] = [first_item]
                start_time = time.time()

                # Collect additional items until max_batch_size or max_latency_ms window expires
                while len(batch) < self.max_batch_size:
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    remaining_time_sec = max(0.0, (self.max_latency_ms - elapsed_ms) / 1000.0)

                    if remaining_time_sec <= 0.0:
                        break

                    try:
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time_sec)
                        batch.append(next_item)
                    except asyncio.TimeoutError:
                        break

                # Execute batch inference
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamic batching loop: {e}", exc_info=True)
                await asyncio.sleep(0.001)

    async def _process_batch(self, batch: List[RequestItem]):
        """Execute vectorized batch inference and resolve futures for all batch items."""
        batch_features = [item.features for item in batch]
        
        try:
            # Execute inference function (run in executor to avoid blocking event loop)
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                self.inference_fn,
                batch_features,
                self.model_version
            )

            preds = res.get("predictions", [0] * len(batch))
            probs = res.get("probabilities", [0.0] * len(batch))
            model_ver = res.get("model_version", self.model_version)
            thresh = res.get("threshold_used", 0.5)

            # Resolve futures
            for idx, item in enumerate(batch):
                if not item.future.done():
                    item.future.set_result({
                        "is_fraud": preds[idx],
                        "fraud_probability": probs[idx],
                        "model_version": model_ver,
                        "threshold_used": thresh,
                        "batch_size": len(batch)
                    })

            self.total_processed_requests += len(batch)
            self.total_batches_executed += 1
            self.batch_sizes_history.append(len(batch))

        except Exception as e:
            logger.error(f"Failed to process batch of size {len(batch)}: {e}", exc_info=True)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def get_stats(self) -> Dict[str, Any]:
        """Return dynamic batcher statistics."""
        avg_batch_size = float(np.mean(self.batch_sizes_history)) if self.batch_sizes_history else 0.0
        return {
            "queue_depth": self.queue.qsize(),
            "total_processed_requests": self.total_processed_requests,
            "total_batches_executed": self.total_batches_executed,
            "avg_batch_size": round(avg_batch_size, 2),
            "max_batch_size": self.max_batch_size,
            "max_latency_ms": self.max_latency_ms
        }
