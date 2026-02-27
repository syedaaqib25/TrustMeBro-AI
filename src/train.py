"""
Model Training & Evaluation Module
====================================
Trains classical ML, deep learning, and transformer models on the cleaned
dataset. Computes evaluation metrics and saves everything to models/.
"""

import os
import sys
import json
import logging
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# Add parent to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_tfidf, extract_word2vec
from data_pipeline import preprocess_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEED = 42

np.random.seed(SEED)


# ===================================================================
# Evaluation helpers
# ===================================================================

def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Compute all required metrics."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def _class_weight_ratio(y):
    """Return class_weight dict for sklearn."""
    from collections import Counter
    counts = Counter(y)
    total = sum(counts.values())
    return {c: total / (len(counts) * v) for c, v in counts.items()}


# ===================================================================
# Classical ML
# ===================================================================

def train_classical(X_train, X_test, y_train, y_test):
    """Train NB, LR, SVM on TF-IDF features. Returns dict of metrics."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}
    cw = _class_weight_ratio(y_train)

    models = {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED, C=1.0,
        ),
        "svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000),
            cv=3,
        ),
        "lightgbm": CalibratedClassifierCV(
            __import__("lightgbm").LGBMClassifier(
                n_estimators=100,
                random_state=SEED,
                class_weight="balanced",
                importance_type="gain",
            ),
            cv=3,
        ),
    }

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("  %s — F1=%.4f  Acc=%.4f  → %s", name, metrics["f1_score"], metrics["accuracy"], model_path)

    return results


# ===================================================================
# Deep Learning (PyTorch) — CNN & LSTM
# ===================================================================

def train_deep_learning(X_train, X_test, y_train, y_test):
    """Train CNN and LSTM on Word2Vec embeddings."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DL device: %s", device)

    input_dim = X_train.shape[1]

    # --- Dataset ---
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # --- Models ---
    class TextCNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Unflatten(1, (1, input_dim)),  # (B, 1, D)
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    class TextLSTM(nn.Module):
        def __init__(self, input_dim, hidden=128):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden * 2, 2)

        def forward(self, x):
            x = x.unsqueeze(1)  # (B, 1, D) → single time step
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(self.drop(out))

    # --- Training loop ---
    def _train_model(model, name, epochs=20, patience=5):
        model = model.to(device)
        # Class weights
        counts = np.bincount(y_train)
        weights = torch.tensor([counts.sum() / (2.0 * c) for c in counts], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_f1, best_state, wait = 0, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            model.eval()
            all_pred, all_true, all_prob = [], [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    all_prob.append(probs[:, 1].cpu().numpy())
                    all_pred.append(logits.argmax(1).cpu().numpy())
                    all_true.append(yb.numpy())

            y_pred = np.concatenate(all_pred)
            y_true = np.concatenate(all_true)
            y_prob = np.concatenate(all_prob)
            f1 = f1_score(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("  %s early stop at epoch %d", name, epoch)
                    break

            if epoch % 5 == 0 or epoch == 1:
                logger.info("  %s epoch %d — loss=%.4f F1=%.4f", name, epoch, total_loss / len(train_loader), f1)

        # Restore best & save
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pt"))
        metrics = evaluate_model(y_true, y_pred, y_prob)
        logger.info("  %s best F1=%.4f  Acc=%.4f", name, metrics["f1_score"], metrics["accuracy"])
        return metrics

    results = {}
    results["cnn"] = _train_model(TextCNN(input_dim), "cnn")
    results["lstm"] = _train_model(TextLSTM(input_dim), "lstm")
    return results


# ===================================================================
# BERT Fine-tuning
# ===================================================================

def train_bert(texts_train, texts_test, y_train, y_test, epochs=3, batch_size=16, max_length=128):
    """Fine-tune bert-base-uncased for binary classification."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("BERT device: %s", device)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx], truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

    train_loader = DataLoader(TextDataset(texts_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(texts_test, y_test), batch_size=batch_size * 2)

    # Class weights
    counts = np.bincount(y_train)
    weights = torch.tensor([counts.sum() / (2.0 * c) for c in counts], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    best_f1, best_state = 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_enc, batch_labels in train_loader:
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(**batch_enc)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_pred, all_true, all_prob = [], [], []
        with torch.no_grad():
            for batch_enc, batch_labels in test_loader:
                batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
                logits = model(**batch_enc).logits
                probs = torch.softmax(logits, dim=1)
                all_prob.append(probs[:, 1].cpu().numpy())
                all_pred.append(logits.argmax(1).cpu().numpy())
                all_true.append(batch_labels.numpy())

        y_pred = np.concatenate(all_pred)
        y_true_arr = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob)
        f1 = f1_score(y_true_arr, y_pred)
        logger.info("  BERT epoch %d — loss=%.4f  F1=%.4f", epoch, total_loss / len(train_loader), f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best & save
    model.load_state_dict(best_state)
    save_dir = os.path.join(MODELS_DIR, "bert_model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("  BERT best F1=%.4f  saved → %s", best_f1, save_dir)

    metrics = evaluate_model(y_true_arr, y_pred, y_prob)
    return {"bert": metrics}


# ===================================================================
# Orchestrator
# ===================================================================

def run_training(mode: str = "classical"):
    """
    Run the full training pipeline.
    mode: 'classical' | 'deep' | 'bert' | 'all'
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load cleaned data
    data_path = os.path.join(DATA_DIR, "cleaned_dataset.csv")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. Run data_pipeline.py first."
        )
    df = pd.read_csv(data_path)
    logger.info("Loaded cleaned data: %d rows", len(df))

    texts = df["text"].values
    labels = df["label"].values

    # Stratified split
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED,
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    all_metrics = {}

    # --- Classical ML (TF-IDF) ---
    if mode in ("classical", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: Classical ML (TF-IDF)")
        logger.info("=" * 60)
        X_train_tfidf, vec = extract_tfidf(list(X_text_train), fit=True)
        X_test_tfidf, _ = extract_tfidf(list(X_text_test), fit=False, vectorizer=vec)
        classical_results = train_classical(X_train_tfidf, X_test_tfidf, y_train, y_test)
        all_metrics.update(classical_results)

    # --- Deep Learning (Word2Vec) ---
    if mode in ("deep", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: Deep Learning (Word2Vec → CNN / LSTM)")
        logger.info("=" * 60)
        X_train_w2v, w2v_model = extract_word2vec(list(X_text_train), fit=True)
        X_test_w2v, _ = extract_word2vec(list(X_text_test), fit=False, model=w2v_model)
        dl_results = train_deep_learning(X_train_w2v, X_test_w2v, y_train, y_test)
        all_metrics.update(dl_results)

    # --- BERT ---
    if mode in ("bert", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: BERT Fine-tuning")
        logger.info("=" * 60)
        bert_results = train_bert(X_text_train, X_text_test, y_train, y_test)
        all_metrics.update(bert_results)

    # --- Save metrics ---
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    # --- Comparison table ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in all_metrics.items():
        roc = m.get("roc_auc", "N/A")
        roc_str = f"{roc:.4f}" if isinstance(roc, (int, float)) else roc
        logger.info(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {roc_str:>9}"
        )

    return all_metrics


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train fake news detection models")
    parser.add_argument(
        "--models",
        choices=["classical", "deep", "bert", "all"],
        default="classical",
        help="Which model group(s) to train",
    )
    args = parser.parse_args()
    run_training(mode=args.models)


if __name__ == "__main__":
    main()
