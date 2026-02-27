"""
Feature Engineering Module
===========================
Provides three feature extraction pipelines: TF-IDF, Word2Vec, and BERT.
Each returns numerical arrays/matrices ready for model consumption.
"""

import os
import logging
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEED = 42


# ===================================================================
# TF-IDF
# ===================================================================

def extract_tfidf(
    texts,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    fit: bool = True,
    vectorizer=None,
):
    """
    Extract TF-IDF features.
    Returns (sparse_matrix, fitted_vectorizer).
    """
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts)
        # persist vectorizer
        os.makedirs(MODELS_DIR, exist_ok=True)
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("TF-IDF fitted: shape %s, saved vectorizer → %s", X.shape, vec_path)
    else:
        if vectorizer is None:
            vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
        X = vectorizer.transform(texts)
    return X, vectorizer


# ===================================================================
# Word2Vec
# ===================================================================

def _train_w2v(tokenized_texts, vector_size=200, window=5, min_count=2, epochs=10):
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=SEED,
        epochs=epochs,
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "word2vec.model")
    model.save(model_path)
    logger.info("Word2Vec trained: vocab=%d, saved → %s", len(model.wv), model_path)
    return model


def _doc_vector(model, tokens, vector_size=200):
    """Average word vectors for a document."""
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(vector_size)


def extract_word2vec(
    texts,
    vector_size: int = 200,
    fit: bool = True,
    model=None,
):
    """
    Extract Word2Vec document embeddings.
    Returns (numpy array of shape [n_docs, vector_size], model).
    """
    tokenized = [text.split() for text in texts]

    if fit:
        model = _train_w2v(tokenized, vector_size=vector_size)
    else:
        if model is None:
            model_path = os.path.join(MODELS_DIR, "word2vec.model")
            model = Word2Vec.load(model_path)

    X = np.array([_doc_vector(model, t, vector_size) for t in tokenized])
    logger.info("Word2Vec embeddings: shape %s", X.shape)
    return X, model


# ===================================================================
# BERT
# ===================================================================

def extract_bert(
    texts,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
):
    """
    Extract BERT [CLS] embeddings using mean pooling.
    Returns numpy array of shape [n_docs, 768].
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
        # Mean pool over token embeddings (ignoring padding)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(embeddings.cpu().numpy())

    X = np.concatenate(all_embeddings, axis=0)
    logger.info("BERT embeddings: shape %s", X.shape)
    return X


# ===================================================================
# Unified entry point
# ===================================================================

def extract_features(texts, feature_type: str = "tfidf", fit: bool = True, **kwargs):
    """
    Unified feature extraction.
    feature_type: 'tfidf' | 'word2vec' | 'bert'
    """
    if feature_type == "tfidf":
        X, _ = extract_tfidf(texts, fit=fit, **kwargs)
        return X
    elif feature_type == "word2vec":
        X, _ = extract_word2vec(texts, fit=fit, **kwargs)
        return X
    elif feature_type == "bert":
        X = extract_bert(list(texts), **kwargs)
        return X
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
