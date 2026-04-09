"""
Data Pipeline Module
====================
Loads raw Fake/True CSV data, applies comprehensive text preprocessing,
and saves a cleaned dataset ready for feature engineering.
"""

import os
import re
import sys
import logging
import argparse

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")

SEED = 42
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK resources (downloaded once)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    """Download required NLTK data if missing and force load it."""
    import socket
    original_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(10)  # 10s timeout to prevent hanging
        for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
            try:
                # Check for both corpora and tokenizers
                if resource == "punkt":
                    nltk.data.find("tokenizers/punkt")
                else:
                    nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning("NLTK download issue (non-fatal): %s", e)
    finally:
        socket.setdefaulttimeout(original_timeout)

    # FORCE LOAD to prevent threading race conditions with LazyLoader
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        _ = stopwords.words("english")
        _ = WordNetLemmatizer().lemmatize("test")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_SPECIAL_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def remove_urls(text: str) -> str:
    return _URL_RE.sub("", text)


def remove_html(text: str) -> str:
    return _HTML_RE.sub("", text)


def remove_special_chars(text: str) -> str:
    return _SPECIAL_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def preprocess_text(
    text: str,
    stop_words: set | None = None,
    lemmatizer: WordNetLemmatizer | None = None,
) -> str:
    """Full preprocessing pipeline for a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)

    tokens = text.split()

    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]

    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Load Fake.csv and True.csv, assign labels, return combined DataFrame."""
    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")

    if not os.path.isfile(fake_path):
        raise FileNotFoundError(f"Missing {fake_path}")
    if not os.path.isfile(true_path):
        raise FileNotFoundError(f"Missing {true_path}")

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    logger.info("Loaded %d fake + %d true = %d total rows", len(fake), len(true), len(df))
    return df


def run_pipeline(sample_frac: float | None = None) -> pd.DataFrame:
    """End-to-end data pipeline: load → combine text → preprocess → save."""
    _ensure_nltk()

    df = load_raw_data()

    # Combine title + text into a single column
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df[["text", "label"]]

    # Drop empties
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Optional sampling for quick dev runs
    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=SEED).reset_index(drop=True)
        logger.info("Sampled to %d rows (%.0f%%)", len(df), sample_frac * 100)

    # Preprocess
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    logger.info("Preprocessing %d rows …", len(df))
    df["text"] = df["text"].apply(
        lambda x: preprocess_text(x, stop_words=stop_words, lemmatizer=lemmatizer)
    )

    # Drop rows that became empty after cleaning
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved cleaned dataset → %s  (%d rows)", OUTPUT_FILE, len(df))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline")
    parser.add_argument(
        "--sample", type=float, default=None,
        help="Fraction of data to sample (e.g. 0.1 for 10%%). Useful for quick testing.",
    )
    args = parser.parse_args()
    run_pipeline(sample_frac=args.sample)


if __name__ == "__main__":
    main()
