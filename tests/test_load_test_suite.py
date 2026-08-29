"""
Unit tests for Locust load testing suite (locustfile.py).
"""
import pytest
from locust.env import Environment
from locustfile import FraudInferenceUser


def test_locust_user_initialization():
    env = Environment()
    user = FraudInferenceUser(environment=env)
    user.on_start()
    assert hasattr(user, "sample_transaction")
    assert "Time" in user.sample_transaction
    assert "Amount" in user.sample_transaction
    assert "V1" in user.sample_transaction
    assert len(user.sample_transaction) == 30
