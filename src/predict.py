"""
Prediction Module
==================
Loads a trained model and vectorizer, preprocesses input text,
and returns a prediction with confidence scores.
"""

import os
import sys
import json
import pickle
import logging
import argparse

import numpy as np

# Allow sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import preprocess_text, _ensure_nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABEL_MAP = {0: "Fake", 1: "True"}


def _preprocess(text: str) -> str:
    """Apply the same preprocessing used during training."""
    _ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return preprocess_text(text, stop_words=stop_words, lemmatizer=lemmatizer)


def _load_classical_model(model_name: str = "logistic_regression"):
    """Load a classical sklearn model + TF-IDF vectorizer."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_text(text: str, model_name: str = "logistic_regression") -> dict:
    """
    Predict whether a news article is fake or true.

    Returns dict with label, confidence, probability_fake, probability_true.
    """
    cleaned = _preprocess(text)
    if not cleaned.strip():
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probability_fake": 0.0,
            "probability_true": 0.0,
            "error": "Text became empty after preprocessing",
        }

    # Check for BERT model
    if model_name == "bert":
        return _predict_bert(cleaned, text)

    # Classical model path
    model, vectorizer = _load_classical_model(model_name)
    X = vectorizer.transform([cleaned])

    # Predict
    pred = model.predict(X)[0]
    label = LABEL_MAP.get(int(pred), "Unknown")

    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_fake = float(round(probs[0], 4))
        prob_true = float(round(probs[1], 4))
        confidence = float(round(max(probs), 4))
    else:
        prob_fake = float(1 - pred)
        prob_true = float(pred)
        confidence = 1.0

    return {
        "label": label,
        "confidence": confidence,
        "probability_fake": prob_fake,
        "probability_true": prob_true,
    }


def _predict_bert(cleaned_text: str, raw_text: str) -> dict:
    """Predict using the fine-tuned BERT model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    bert_dir = os.path.join(MODELS_DIR, "bert_model")
    if not os.path.isdir(bert_dir):
        raise FileNotFoundError(f"BERT model not found at {bert_dir}. Train it first with --models bert")

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    model.eval()

    encoded = tokenizer(
        raw_text, truncation=True, padding="max_length",
        max_length=128, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    pred = int(probs.argmax())
    return {
        "label": LABEL_MAP.get(pred, "Unknown"),
        "confidence": float(round(max(probs), 4)),
        "probability_fake": float(round(probs[0], 4)),
        "probability_true": float(round(probs[1], 4)),
    }


def list_available_models() -> list[str]:
    """Return names of trained models."""
    models = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pkl") and f != "tfidf_vectorizer.pkl":
                models.append(f.replace(".pkl", ""))
            elif f.endswith(".pt"):
                models.append(f.replace(".pt", ""))
        if os.path.isdir(os.path.join(MODELS_DIR, "bert_model")):
            models.append("bert")
    return sorted(models)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Predict fake/true news from text")
    parser.add_argument("--text", type=str, required=True, help="News article text to classify")
    parser.add_argument(
        "--model", type=str, default="logistic_regression",
        help="Model to use (default: logistic_regression). Use 'bert' for BERT.",
    )
    args = parser.parse_args()

    result = predict_text(args.text, model_name=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
