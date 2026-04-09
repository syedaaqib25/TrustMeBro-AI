"""
Integration tests for the FastAPI endpoints.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]


class TestPredictEndpoint:
    def test_predict_valid_request(self, sample_fake_text):
        response = client.post("/predict", json={
            "text": sample_fake_text,
            "model": "logistic_regression",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ("Fake", "True")
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["probability_fake"] <= 1.0
        assert 0.0 <= data["probability_true"] <= 1.0

    def test_predict_default_model(self, sample_true_text):
        response = client.post("/predict", json={"text": sample_true_text})
        assert response.status_code == 200
        assert "label" in response.json()

    def test_predict_short_text_rejected(self, short_text):
        response = client.post("/predict", json={"text": short_text})
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_text(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_nonexistent_model(self, sample_fake_text):
        response = client.post("/predict", json={
            "text": sample_fake_text,
            "model": "nonexistent_model_xyz",
        })
        assert response.status_code in (404, 500)


class TestModelsEndpoint:
    def test_get_models_returns_200(self):
        response = client.get("/models")
        assert response.status_code == 200

    def test_get_models_returns_list(self):
        response = client.get("/models")
        data = response.json()
        assert "available_models" in data
        assert isinstance(data["available_models"], list)

    def test_get_models_includes_trained(self):
        response = client.get("/models")
        models = response.json()["available_models"]
        assert "logistic_regression" in models


class TestMetricsEndpoint:
    def test_get_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_get_metrics_has_model_data(self):
        response = client.get("/metrics")
        data = response.json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert isinstance(metrics, dict)
        # Should have at least one model's metrics
        assert len(metrics) > 0

    def test_metrics_contain_expected_fields(self):
        response = client.get("/metrics")
        metrics = response.json()["metrics"]
        for model_name, model_metrics in metrics.items():
            assert "accuracy" in model_metrics
            assert "precision" in model_metrics
            assert "recall" in model_metrics
            assert "f1_score" in model_metrics

class TestAnalyzeEndpoint:
    def test_analyze_valid_request(self, sample_fake_text):
        response = client.post("/analyze", json={"text": sample_fake_text})
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "gemini" in data
        assert "groq" in data
        assert "overall_score" in data
        assert "overall_label" in data
        assert len(data["models"]) > 0
        assert data["overall_score"] >= 0.0

    def test_analyze_short_text_rejected(self, short_text):
        response = client.post("/analyze", json={"text": short_text})
        assert response.status_code == 422

    def test_analyze_overall_label_values(self, sample_fake_text):
        response = client.post("/analyze", json={"text": sample_fake_text})
        data = response.json()
        assert data["overall_label"] in ("Likely Credible", "Uncertain", "Likely Fake")


class TestHistoryEndpoint:
    def test_get_history_returns_200(self):
        response = client.get("/history")
        assert response.status_code == 200

    def test_get_history_returns_list(self):
        response = client.get("/history")
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_delete_history_returns_200(self):
        response = client.delete("/history")
        assert response.status_code == 200
        assert "message" in response.json()


class TestFeedbackEndpoint:
    def test_submit_feedback(self, sample_fake_text):
        payload = {
            "article_text": sample_fake_text,
            "model_scores": [],
            "overall_score": 35.0,
            "user_label": "Fake",
            "user_description": "Test feedback submission",
        }
        response = client.post("/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_get_feedback_returns_200(self):
        response = client.get("/feedback")
        assert response.status_code == 200
        data = response.json()
        assert "feedback" in data
        assert isinstance(data["feedback"], list)

    def test_submit_feedback_missing_label_rejected(self, sample_fake_text):
        payload = {
            "article_text": sample_fake_text,
            "model_scores": [],
            "overall_score": 50.0,
        }
        response = client.post("/feedback", json=payload)
        assert response.status_code == 422

    def test_delete_feedback_returns_200(self):
        response = client.delete("/feedback")
        assert response.status_code == 200
        assert "message" in response.json()


class TestFetchUrlEndpoint:
    def test_fetch_invalid_url(self):
        response = client.post("/fetch-url", json={"url": "not-a-url"})
        assert response.status_code == 422

    def test_fetch_missing_url_rejected(self):
        response = client.post("/fetch-url", json={})
        assert response.status_code == 422

