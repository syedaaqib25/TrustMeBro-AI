"""
Unit tests for src/predict.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from predict import predict_text, list_available_models


class TestPredictText:
    """Tests for the predict_text function (requires trained models)."""

    def test_returns_correct_keys(self, sample_fake_text):
        result = predict_text(sample_fake_text)
        assert "label" in result
        assert "confidence" in result
        assert "probability_fake" in result
        assert "probability_true" in result

    def test_label_is_valid(self, sample_fake_text):
        result = predict_text(sample_fake_text)
        assert result["label"] in ("Fake", "True", "Unknown")

    def test_confidence_in_range(self, sample_true_text):
        result = predict_text(sample_true_text)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_roughly_to_one(self, sample_true_text):
        result = predict_text(sample_true_text)
        total = result["probability_fake"] + result["probability_true"]
        assert abs(total - 1.0) < 0.01

    def test_empty_after_preprocessing_returns_unknown(self):
        # Text with only special chars / URLs becomes empty after cleaning
        result = predict_text("!!! ??? ### $$$")
        assert result["label"] == "Unknown"
        assert result["confidence"] == 0.0

    def test_predict_with_svm_model(self, sample_fake_text):
        result = predict_text(sample_fake_text, model_name="svm")
        assert result["label"] in ("Fake", "True")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_with_naive_bayes(self, sample_fake_text):
        result = predict_text(sample_fake_text, model_name="naive_bayes")
        assert result["label"] in ("Fake", "True")

    def test_missing_model_raises(self, sample_fake_text):
        import pytest
        with pytest.raises(FileNotFoundError):
            predict_text(sample_fake_text, model_name="nonexistent_model")


class TestListAvailableModels:
    def test_returns_list(self):
        models = list_available_models()
        assert isinstance(models, list)

    def test_contains_trained_models(self):
        models = list_available_models()
        # We trained classical models, so at least these should exist
        assert "logistic_regression" in models
        assert "svm" in models
        assert "naive_bayes" in models
