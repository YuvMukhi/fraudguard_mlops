"""
Safe Rollout Router supporting A/B Testing & Shadow Deployment.
Enables controlled traffic split and live prediction divergence tracking between Champion & Challenger models.
"""
import time
import random
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RolloutRouter:
    """
    Manages safe model deployment strategies: A/B testing and Shadow Deployment.
    """

    def __init__(
        self,
        predict_fn: Callable[[List[Dict[str, float]], str], Dict[str, Any]],
        champion_version: str = "champion",
        challenger_version: str = "baseline"
    ):
        self.predict_fn = predict_fn
        self.champion_version = champion_version
        self.challenger_version = challenger_version

        # Telemetry metrics
        self.ab_champion_count: int = 0
        self.ab_challenger_count: int = 0
        self.shadow_total_count: int = 0
        self.shadow_disagreement_count: int = 0
        self.divergence_history: List[float] = []

    def _hash_request(self, request_id: str) -> float:
        """Hash string to a deterministic float ratio between 0.0 and 1.0."""
        h = hashlib.md5(request_id.encode("utf-8")).hexdigest()
        val = int(h[:8], 16)
        return float(val) / 0xFFFFFFFF

    def route_ab_test(
        self,
        features_list: List[Dict[str, float]],
        challenger_ratio: float = 0.20,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route request to Champion or Challenger based on traffic split ratio.
        """
        # Determine routing
        ratio_val = self._hash_request(request_id) if request_id else random.random()
        use_challenger = ratio_val < challenger_ratio

        if use_challenger:
            selected_model = self.challenger_version
            self.ab_challenger_count += len(features_list)
        else:
            selected_model = self.champion_version
            self.ab_champion_count += len(features_list)

        # Execute prediction on selected model
        primary_res = self.predict_fn(features_list, selected_model)

        # Asynchronously or secondary log prediction on both models for offline audit
        try:
            other_model = self.champion_version if use_challenger else self.challenger_version
            secondary_res = self.predict_fn(features_list, other_model)
            
            p_primary = primary_res.get("probabilities", [])
            p_secondary = secondary_res.get("probabilities", [])
            if p_primary and p_secondary:
                diffs = [abs(p1 - p2) for p1, p2 in zip(p_primary, p_secondary)]
                self.divergence_history.extend(diffs)
        except Exception as e:
            logger.warning(f"Error in A/B background dual prediction: {e}")

        return {
            "predictions": primary_res.get("predictions", []),
            "probabilities": primary_res.get("probabilities", []),
            "model_version": primary_res.get("model_version", selected_model),
            "threshold_used": primary_res.get("threshold_used", 0.5),
            "routing_mode": "ab_test",
            "selected_variant": "challenger" if use_challenger else "champion"
        }

    def route_shadow(self, features_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Synchronously serve Champion model while executing Challenger silently in shadow mode.
        Computes prediction probability divergence and binary label disagreement.
        """
        # 1. Execute Champion (Primary synchronously)
        champ_res = self.predict_fn(features_list, self.champion_version)

        # 2. Execute Challenger (Shadow)
        try:
            chall_res = self.predict_fn(features_list, self.challenger_version)

            champ_probs = champ_res.get("probabilities", [])
            chall_probs = chall_res.get("probabilities", [])
            champ_preds = champ_res.get("predictions", [])
            chall_preds = chall_res.get("predictions", [])

            for p_c, p_ch, y_c, y_ch in zip(champ_probs, chall_probs, champ_preds, chall_preds):
                diff = abs(p_c - p_ch)
                self.divergence_history.append(diff)
                self.shadow_total_count += 1
                if y_c != y_ch:
                    self.shadow_disagreement_count += 1

        except Exception as e:
            logger.error(f"Error executing shadow challenger model: {e}", exc_info=True)

        return {
            "predictions": champ_res.get("predictions", []),
            "probabilities": champ_res.get("probabilities", []),
            "model_version": champ_res.get("model_version", self.champion_version),
            "threshold_used": champ_res.get("threshold_used", 0.5),
            "routing_mode": "shadow",
            "shadow_active": True
        }

    def get_rollout_stats(self) -> Dict[str, Any]:
        """Return rollout statistics and divergence metrics."""
        total_ab = self.ab_champion_count + self.ab_challenger_count
        champ_split = round(self.ab_champion_count / total_ab, 4) if total_ab > 0 else 0.80
        chall_split = round(self.ab_challenger_count / total_ab, 4) if total_ab > 0 else 0.20

        mean_div = float(np.mean(self.divergence_history)) if self.divergence_history else 0.0
        max_div = float(np.max(self.divergence_history)) if self.divergence_history else 0.0
        p95_div = float(np.percentile(self.divergence_history, 95)) if self.divergence_history else 0.0

        disagreement_rate = (
            round(self.shadow_disagreement_count / self.shadow_total_count, 4)
            if self.shadow_total_count > 0 else 0.0
        )

        return {
            "ab_testing": {
                "total_routed": total_ab,
                "champion_traffic": self.ab_champion_count,
                "challenger_traffic": self.ab_challenger_count,
                "traffic_split_ratio": {"champion": champ_split, "challenger": chall_split}
            },
            "shadow_deployment": {
                "total_shadow_evaluations": self.shadow_total_count,
                "label_disagreements": self.shadow_disagreement_count,
                "disagreement_rate": disagreement_rate,
                "divergence_metrics": {
                    "mean_abs_divergence": round(mean_div, 4),
                    "max_abs_divergence": round(max_div, 4),
                    "p95_abs_divergence": round(p95_div, 4)
                }
            }
        }
