"""
FastAPI Backend
================
REST API for the TrustMeBro Fake News Detection System.
Endpoints: /analyze, /predict, /train, /metrics, /models, /history, /feedback, /fetch-url
"""

import os
import sys
import json
import asyncio
import logging
from typing import Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Allow imports from src/
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from predict import predict_text, list_available_models
from train import run_training
from data_pipeline import run_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env file from project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY", "")

# Rate limiting — relaxed to match actual free-tier quotas
# Gemini free tier: 15 RPM, 1500 RPD
GEMINI_MAX_PER_MINUTE = 14
GEMINI_MAX_PER_DAY = 1400
_gemini_calls: list[float] = []

# ChatGPT: generous limits for paid tier
CHATGPT_MAX_PER_MINUTE = 20
CHATGPT_MAX_PER_DAY = 500
_chatgpt_calls: list[float] = []

import time as _time

def _is_key_valid(key: str) -> bool:
    """Check if an API key looks real (not empty or placeholder)."""
    if not key or not key.strip():
        return False
    placeholders = {"your_gemini_api_key_here", "your_openai_api_key_here", ""}
    return key.strip() not in placeholders

def _gemini_rate_ok() -> bool:
    """Check if we're within Gemini free-tier rate limits."""
    now = _time.time()
    _gemini_calls[:] = [t for t in _gemini_calls if now - t < 86400]
    if len(_gemini_calls) >= GEMINI_MAX_PER_DAY:
        return False
    recent = [t for t in _gemini_calls if now - t < 60]
    if len(recent) >= GEMINI_MAX_PER_MINUTE:
        return False
    return True

# Thread pool for running sync ML predictions concurrently
_executor = ThreadPoolExecutor(max_workers=4)


# ===================================================================
# FastAPI App
# ===================================================================

app = FastAPI(
    title="TrustMeBro — Multi-Model News Analyzer",
    description="Analyze news articles using 4 ML models and Gemini AI concurrently.",
    version="3.0.0",
)

# Initialize NLTK on startup to prevent threading race conditions
from data_pipeline import _ensure_nltk
_ensure_nltk()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ===================================================================
# Pydantic Models
# ===================================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article text to classify")

class ModelScore(BaseModel):
    model_name: str
    label: str
    confidence: float
    probability_fake: float
    probability_true: float
    credibility_score: float  # 0-100

class GeminiScore(BaseModel):
    score: Optional[float] = None  # 0-100
    reasoning: str = ""
    error: Optional[str] = None

class ChatGPTScore(BaseModel):
    score: Optional[float] = None  # 0-100
    reasoning: str = ""
    error: Optional[str] = None

class AnalyzeResponse(BaseModel):
    models: list[ModelScore]
    gemini: GeminiScore
    chatgpt: ChatGPTScore
    overall_score: float  # 0-100
    overall_label: str

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article text to classify")
    model: str = Field(default="logistic_regression", description="Model name to use")

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probability_fake: float
    probability_true: float
    error: Optional[str] = None

class TrainRequest(BaseModel):
    models: str = Field(default="classical", description="Model group")
    run_pipeline: bool = Field(default=False)

class TrainResponse(BaseModel):
    message: str

class ModelInfo(BaseModel):
    available_models: list[str]

class MetricsResponse(BaseModel):
    metrics: dict

class FetchUrlRequest(BaseModel):
    url: str = Field(..., description="URL of the news article to fetch")

class FetchUrlResponse(BaseModel):
    text: str
    title: str = ""

class HistoryResponse(BaseModel):
    history: list[dict]

class FeedbackRequest(BaseModel):
    article_text: str
    model_scores: list[dict]
    gemini_score: Optional[float] = None
    chatgpt_score: Optional[float] = None
    overall_score: float
    user_label: str = Field(..., description="'True' or 'Fake'")
    user_description: str = ""

class FeedbackResponse(BaseModel):
    message: str

class FeedbackListResponse(BaseModel):
    feedback: list[dict]


# ===================================================================
# Gemini API
# ===================================================================

def _parse_ai_json(response_text: str) -> dict:
    """Extract JSON from AI response, handling markdown code blocks."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(cleaned)


async def _call_gemini(text: str) -> dict:
    """Call Gemini API for credibility scoring. Returns {score, reasoning, error}."""
    # Skip if no valid API key
    if not _is_key_valid(GEMINI_API_KEY):
        return {"score": None, "reasoning": "", "error": "No Gemini API key configured. Add GEMINI_API_KEY to .env file."}

    # Rate limit check
    if not _gemini_rate_ok():
        return {"score": None, "reasoning": "", "error": "Gemini rate limit reached. Try again in a minute."}

    from google import genai

    prompt = f"""Rate this news article's credibility from 0 (fake) to 100 (verified).
Return ONLY JSON: {{"score": <0-100>, "reasoning": "<one sentence>"}}

Article:
{text[:2000]}"""

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            _gemini_calls.append(_time.time())
            logger.info("Gemini API call (attempt %d/%d)", attempt + 1, max_retries)

            client = genai.Client(api_key=GEMINI_API_KEY)
            loop = asyncio.get_running_loop()

            def _generate():
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                return response.text

            response_text = await loop.run_in_executor(_executor, _generate)
            result = _parse_ai_json(response_text)
            score = max(0, min(100, float(result.get("score", 50))))
            reasoning = str(result.get("reasoning", ""))

            logger.info("Gemini API SUCCESS: score=%s", score)
            return {"score": score, "reasoning": reasoning, "error": None}

        except Exception as e:
            last_error = str(e)
            is_rate_error = ("429" in last_error or "quota" in last_error.lower()
                            or "resource_exhausted" in last_error.lower())
            if is_rate_error and attempt < max_retries - 1:
                logger.info("Gemini rate limited, retrying in 2s...")
                await asyncio.sleep(2)
                continue
            break

    # All retries exhausted
    if "429" in last_error or "quota" in last_error.lower() or "resource_exhausted" in last_error.lower():
        last_error = "Gemini quota exceeded. Please wait a minute and try again."
    logger.warning("Gemini API call failed: %s", last_error)
    return {"score": None, "reasoning": "", "error": last_error}


# ===================================================================
# ChatGPT API
# ===================================================================

def _chatgpt_rate_ok() -> bool:
    """Check if we're within ChatGPT rate limits."""
    now = _time.time()
    _chatgpt_calls[:] = [t for t in _chatgpt_calls if now - t < 86400]
    if len(_chatgpt_calls) >= CHATGPT_MAX_PER_DAY:
        return False
    recent = [t for t in _chatgpt_calls if now - t < 60]
    if len(recent) >= CHATGPT_MAX_PER_MINUTE:
        return False
    return True

async def _call_chatgpt(text: str) -> dict:
    """Call ChatGPT API for credibility scoring. Returns {score, reasoning, error}."""
    # Skip if no valid API key
    if not _is_key_valid(CHATGPT_API_KEY):
        return {"score": None, "reasoning": "", "error": "No ChatGPT API key configured. Add CHATGPT_API_KEY to .env file."}

    if not _chatgpt_rate_ok():
        return {"score": None, "reasoning": "", "error": "ChatGPT rate limit reached. Try again later."}

    from openai import OpenAI

    prompt = f"""Rate this news article's credibility from 0 (fake) to 100 (verified).
Return ONLY JSON: {{"score": <0-100>, "reasoning": "<one sentence>"}}

Article:
{text[:2000]}"""

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            _chatgpt_calls.append(_time.time())
            logger.info("ChatGPT API call (attempt %d/%d)", attempt + 1, max_retries)

            client = OpenAI(api_key=CHATGPT_API_KEY, timeout=30)
            loop = asyncio.get_running_loop()

            def _generate():
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150,
                )
                return response.choices[0].message.content

            response_text = await loop.run_in_executor(_executor, _generate)
            result = _parse_ai_json(response_text)
            score = max(0, min(100, float(result.get("score", 50))))
            reasoning = str(result.get("reasoning", ""))

            logger.info("ChatGPT API SUCCESS: score=%s", score)
            return {"score": score, "reasoning": reasoning, "error": None}

        except Exception as e:
            last_error = str(e)
            is_rate_error = ("429" in last_error or "rate_limit" in last_error.lower()
                            or "insufficient_quota" in last_error.lower())
            if is_rate_error and attempt < max_retries - 1:
                logger.info("ChatGPT rate limited, retrying in 2s...")
                await asyncio.sleep(2)
                continue
            break

    # All retries exhausted — friendly error messages
    if "insufficient_quota" in last_error.lower() or "billing" in last_error.lower():
        last_error = "OpenAI quota exceeded. Please add credits at platform.openai.com/settings/organization/billing"
    elif "429" in last_error or "rate_limit" in last_error.lower():
        last_error = "ChatGPT rate limit hit. Please wait a moment and retry."
    elif "auth" in last_error.lower() or "api_key" in last_error.lower() or "invalid" in last_error.lower():
        last_error = "ChatGPT API key is invalid. Please update the key."
    logger.warning("ChatGPT API call failed: %s", last_error)
    return {"score": None, "reasoning": "", "error": last_error}


# ===================================================================
# ML Prediction helpers
# ===================================================================

def _run_model(text: str, model_name: str) -> dict:
    """Run a single ML model prediction (sync, for thread pool)."""
    try:
        result = predict_text(text, model_name=model_name)
        # Convert to credibility score (0–100)
        # probability_true maps to credibility
        credibility = round(result["probability_true"] * 100, 1)
        return {
            "model_name": model_name,
            "label": result["label"],
            "confidence": result["confidence"],
            "probability_fake": result["probability_fake"],
            "probability_true": result["probability_true"],
            "credibility_score": credibility,
        }
    except Exception as e:
        logger.error("Model %s failed: %s", model_name, e)
        return {
            "model_name": model_name,
            "label": "Error",
            "confidence": 0.0,
            "probability_fake": 0.0,
            "probability_true": 0.0,
            "credibility_score": 0.0,
        }


# ===================================================================
# History & Feedback helpers
# ===================================================================

MAX_HISTORY = 50

def _load_json(path: str) -> list:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def _save_json(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _append_json(path: str, entry: dict, max_items: int = 500):
    items = _load_json(path)
    items.insert(0, entry)
    items = items[:max_items]
    _save_json(path, items)


# ===================================================================
# Background training
# ===================================================================

def _background_train(mode: str, should_run_pipeline: bool):
    try:
        if should_run_pipeline:
            logger.info("Running data pipeline first …")
            run_pipeline()
        run_training(mode=mode)
        logger.info("Background training complete.")
    except Exception as e:
        logger.error("Background training failed: %s", e)


# ===================================================================
# Endpoints
# ===================================================================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Run LR, SVM, NB, LightGBM concurrently + Gemini + ChatGPT in parallel.
    Returns all scores + weighted overall credibility.
    """
    loop = asyncio.get_running_loop()
    models_to_run = ["logistic_regression", "svm", "naive_bayes", "lightgbm"]

    # Run all 4 ML models + Gemini + ChatGPT concurrently
    ml_futures = [
        loop.run_in_executor(_executor, _run_model, req.text, m)
        for m in models_to_run
    ]
    gemini_future = _call_gemini(req.text)
    chatgpt_future = _call_chatgpt(req.text)

    # Gather all results simultaneously
    ml_results = await asyncio.gather(*ml_futures)
    gemini_result = await gemini_future
    chatgpt_result = await chatgpt_future

    # Calculate weighted overall score
    valid_ml = [r for r in ml_results if r["label"] != "Error"]
    ml_avg = sum(r["credibility_score"] for r in valid_ml) / len(valid_ml) if valid_ml else 50.0

    gemini_ok = gemini_result["score"] is not None
    chatgpt_ok = chatgpt_result["score"] is not None

    if gemini_ok and chatgpt_ok:
        # Both AI models: ML 50%, Gemini 25%, ChatGPT 25%
        overall = round((ml_avg * 0.50) + (gemini_result["score"] * 0.25) + (chatgpt_result["score"] * 0.25), 1)
    elif gemini_ok:
        overall = round((ml_avg * 0.6) + (gemini_result["score"] * 0.4), 1)
    elif chatgpt_ok:
        overall = round((ml_avg * 0.6) + (chatgpt_result["score"] * 0.4), 1)
    else:
        overall = round(ml_avg, 1)

    # Determine overall label
    if overall >= 70:
        overall_label = "Likely Credible"
    elif overall >= 40:
        overall_label = "Uncertain"
    else:
        overall_label = "Likely Fake"

    # Save to history
    snippet = req.text[:200] + ("…" if len(req.text) > 200 else "")
    _append_json(HISTORY_FILE, {
        "text_snippet": snippet,
        "overall_score": overall,
        "overall_label": overall_label,
        "model_scores": {r["model_name"]: r["credibility_score"] for r in ml_results},
        "gemini_score": gemini_result["score"],
        "chatgpt_score": chatgpt_result["score"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, max_items=MAX_HISTORY)

    return AnalyzeResponse(
        models=[ModelScore(**r) for r in ml_results],
        gemini=GeminiScore(**gemini_result),
        chatgpt=ChatGPTScore(**chatgpt_result),
        overall_score=overall,
        overall_label=overall_label,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Single-model prediction (backward compatible)."""
    try:
        result = predict_text(req.text, model_name=req.model)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_train, req.models, req.run_pipeline)
    return TrainResponse(
        message=f"Training started (mode='{req.models}', pipeline={req.run_pipeline})"
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise HTTPException(status_code=404, detail="No metrics found.")
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    return MetricsResponse(metrics=data)


@app.get("/models", response_model=ModelInfo)
async def get_models():
    return ModelInfo(available_models=list_available_models())


# ===================================================================
# History
# ===================================================================

@app.get("/history", response_model=HistoryResponse)
async def get_history():
    return HistoryResponse(history=_load_json(HISTORY_FILE))

@app.delete("/history")
async def clear_history():
    _save_json(HISTORY_FILE, [])
    return {"message": "History cleared"}


# ===================================================================
# Feedback
# ===================================================================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Store user feedback for retraining dataset."""
    entry = {
        "article_snippet": req.article_text[:300] + ("…" if len(req.article_text) > 300 else ""),
        "article_full": req.article_text,
        "model_scores": req.model_scores,
        "gemini_score": req.gemini_score,
        "chatgpt_score": req.chatgpt_score,
        "overall_score": req.overall_score,
        "user_label": req.user_label,
        "user_description": req.user_description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_json(FEEDBACK_FILE, entry, max_items=1000)
    return FeedbackResponse(message="Feedback saved. Thank you!")


@app.get("/feedback", response_model=FeedbackListResponse)
async def get_feedback():
    """Return all feedback for admin dashboard."""
    return FeedbackListResponse(feedback=_load_json(FEEDBACK_FILE))

@app.delete("/feedback")
async def clear_feedback():
    _save_json(FEEDBACK_FILE, [])
    return {"message": "Feedback cleared"}


# ===================================================================
# URL Fetching
# ===================================================================

@app.post("/fetch-url", response_model=FetchUrlResponse)
async def fetch_url(req: FetchUrlRequest):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise HTTPException(status_code=500, detail="Install requests and beautifulsoup4")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(req.url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        article = soup.find("article")
        paragraphs = (article or soup).find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        if not text:
            raise HTTPException(status_code=422, detail="Could not extract article text.")

        return FetchUrlResponse(text=text, title=title)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {str(e)}")


# ===================================================================
# Headlines (NewsAPI)
# ===================================================================

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
_headlines_cache: dict = {"data": None, "timestamp": 0}
HEADLINES_CACHE_SECONDS = 600  # 10 minutes

@app.get("/headlines")
async def get_headlines(category: str = "general", country: str = "us"):
    """Fetch top headlines from NewsAPI with in-memory caching."""
    if not _is_key_valid(NEWS_API_KEY):
        raise HTTPException(status_code=503, detail="No NEWS_API_KEY configured in .env")

    now = _time.time()
    cache_key = f"{country}_{category}"

    # Return cached data if fresh
    if (_headlines_cache.get("key") == cache_key
            and _headlines_cache["data"] is not None
            and now - _headlines_cache["timestamp"] < HEADLINES_CACHE_SECONDS):
        return _headlines_cache["data"]

    try:
        import requests as req_lib
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "country": country,
            "category": category,
            "pageSize": 12,
            "apiKey": NEWS_API_KEY,
        }
        resp = req_lib.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        articles = []
        for a in raw.get("articles", []):
            if a.get("title") and "[Removed]" not in a["title"]:
                articles.append({
                    "title": a["title"],
                    "description": (a.get("description") or "")[:200],
                    "url": a.get("url", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "image": a.get("urlToImage", ""),
                    "publishedAt": a.get("publishedAt", ""),
                })

        result = {"articles": articles, "totalResults": len(articles)}
        _headlines_cache.update({"key": cache_key, "data": result, "timestamp": now})
        return result

    except Exception as e:
        logger.warning("NewsAPI call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to fetch headlines: {str(e)}")


# ===================================================================
# Root
# ===================================================================

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
