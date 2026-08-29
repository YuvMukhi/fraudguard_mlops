"""
Unit tests for Dynamic Batcher and Async Queue.
"""
import pytest
import asyncio
from src.queue.async_queue import DynamicBatcher


def dummy_inference_fn(batch_features, model_version):
    # Dummy inference returning 0 or 1 based on Amount feature
    preds = [1 if f.get("Amount", 0.0) > 500 else 0 for f in batch_features]
    probs = [0.95 if p == 1 else 0.05 for p in preds]
    return {
        "predictions": preds,
        "probabilities": probs,
        "model_version": model_version,
        "threshold_used": 0.5
    }


@pytest.mark.asyncio
async def test_dynamic_batcher_size_trigger():
    batcher = DynamicBatcher(
        inference_fn=dummy_inference_fn,
        max_batch_size=5,
        max_latency_ms=100.0,
        model_version="test_v1"
    )
    await batcher.start()

    try:
        sample_features = {"Amount": 100.0}
        tasks = [batcher.enqueue(sample_features) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for res in results:
            assert res["is_fraud"] == 0
            assert res["batch_size"] == 5
            assert res["model_version"] == "test_v1"
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_dynamic_batcher_latency_trigger():
    batcher = DynamicBatcher(
        inference_fn=dummy_inference_fn,
        max_batch_size=100,  # High batch limit so latency window triggers batch
        max_latency_ms=20.0,
        model_version="test_v1"
    )
    await batcher.start()

    try:
        sample_fraud = {"Amount": 600.0}
        tasks = [batcher.enqueue(sample_fraud) for _ in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for res in results:
            assert res["is_fraud"] == 1
            assert res["batch_size"] == 3
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_dynamic_batcher_stats():
    batcher = DynamicBatcher(
        inference_fn=dummy_inference_fn,
        max_batch_size=2,
        max_latency_ms=50.0
    )
    await batcher.start()

    try:
        await batcher.enqueue({"Amount": 10.0})
        await batcher.enqueue({"Amount": 20.0})

        stats = batcher.get_stats()
        assert stats["total_processed_requests"] >= 2
        assert stats["total_batches_executed"] >= 1
    finally:
        await batcher.stop()
