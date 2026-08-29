"""
Unit tests for Safe Rollout Router (A/B testing & Shadow deployment).
"""
import pytest
from fastapi.testclient import TestClient
from src.rollout.router import RolloutRouter
from src.serving.api import app


def dummy_predict_fn(features_list, model_version):
    # Dummy predict function returning slightly different probabilities for champion vs challenger
    is_champion = "heavy" in model_version.lower() or "champion" in model_version.lower()
    base_prob = 0.85 if is_champion else 0.45
    preds = [1 if base_prob > 0.5 else 0 for _ in features_list]
    probs = [base_prob for _ in features_list]
    return {
        "predictions": preds,
        "probabilities": probs,
        "model_version": model_version,
        "threshold_used": 0.5
    }


def test_ab_testing_traffic_split():
    router = RolloutRouter(predict_fn=dummy_predict_fn, champion_version="champion", challenger_version="baseline")
    
    sample_features = [{"Amount": 100.0}]
    for i in range(100):
        _ = router.route_ab_test(sample_features, challenger_ratio=0.30)

    stats = router.get_rollout_stats()
    ab_stats = stats["ab_testing"]
    assert ab_stats["total_routed"] == 100
    assert ab_stats["champion_traffic"] > 0
    assert ab_stats["challenger_traffic"] > 0
    # Ratio should be approximately 70% champion / 30% challenger
    assert 0.15 <= ab_stats["traffic_split_ratio"]["challenger"] <= 0.45


def test_shadow_deployment_divergence():
    router = RolloutRouter(predict_fn=dummy_predict_fn, champion_version="champion", challenger_version="baseline")
    
    sample_features = [{"Amount": 100.0}]
    for _ in range(10):
        res = router.route_shadow(sample_features)
        assert res["routing_mode"] == "shadow"
        assert res["shadow_active"] is True

    stats = router.get_rollout_stats()
    shadow_stats = stats["shadow_deployment"]
    assert shadow_stats["total_shadow_evaluations"] == 10
    assert shadow_stats["divergence_metrics"]["mean_abs_divergence"] > 0.0
    # Dummy champion prob=0.85 (pred 1), challenger prob=0.45 (pred 0) -> disagreement rate = 1.0
    assert shadow_stats["disagreement_rate"] == 1.0


def test_rollout_api_endpoints():
    with TestClient(app) as client:
        sample_txn = {
            "Time": 10.0, "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
            "V6": 0.1, "V7": 0.2, "V8": 0.3, "V9": 0.4, "V10": 0.5,
            "V11": 0.1, "V12": 0.2, "V13": 0.3, "V14": 0.4, "V15": 0.5,
            "V16": 0.1, "V17": 0.2, "V18": 0.3, "V19": 0.4, "V20": 0.5,
            "V21": 0.1, "V22": 0.2, "V23": 0.3, "V24": 0.4, "V25": 0.5,
            "V26": 0.1, "V27": 0.2, "V28": 0.3, "Amount": 50.0
        }

        # 1. Test A/B test request
        ab_payload = {
            "transactions": [sample_txn],
            "mode": "ab_test",
            "challenger_ratio": 0.50
        }
        res_ab = client.post("/predict/rollout", json=ab_payload)
        assert res_ab.status_code == 200
        data_ab = res_ab.json()
        assert data_ab["routing_mode"] == "ab_test"
        assert data_ab["selected_variant"] in ["champion", "challenger"]

        # 2. Test Shadow request
        shadow_payload = {
            "transactions": [sample_txn],
            "mode": "shadow"
        }
        res_shadow = client.post("/predict/rollout", json=shadow_payload)
        assert res_shadow.status_code == 200
        data_shadow = res_shadow.json()
        assert data_shadow["routing_mode"] == "shadow"

        # 3. Test stats endpoint
        res_stats = client.get("/rollout/stats")
        assert res_stats.status_code == 200
        stats_data = res_stats.json()
        assert "ab_testing" in stats_data
        assert "shadow_deployment" in stats_data
