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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Rate limiting — relaxed to match actual free-tier quotas
# Gemini free tier: 15 RPM, 1500 RPD
GEMINI_MAX_PER_MINUTE = 14
GEMINI_MAX_PER_DAY = 1400
_gemini_calls: list[float] = []

# Groq (Fast Inference): generous limits
GROQ_MAX_PER_MINUTE = 30
GROQ_MAX_PER_DAY = 14400
_groq_calls: list[float] = []

import time as _time

def _is_key_valid(key: str) -> bool:
    """Check if an API key looks real (not empty or placeholder)."""
    if not key or not key.strip():
        return False
    placeholders = {"your_gemini_api_key_here", "your_groq_api_key_here", "your-groq-key-here", ""}
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
try:
    _ensure_nltk()
except Exception as _nltk_err:
    logger.warning("NLTK initialization failed (non-fatal): %s", _nltk_err)

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

class GroqScore(BaseModel):
    score: Optional[float] = None  # 0-100
    reasoning: str = ""
    error: Optional[str] = None

class AnalyzeResponse(BaseModel):
    models: list[ModelScore]
    gemini: GeminiScore
    groq: GroqScore
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
    groq_score: Optional[float] = None
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
    """Extract JSON from AI response, handling markdown code blocks and extra text."""
    import re
    cleaned = response_text.strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Strip markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    # Regex fallback: find first {...} block
    match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in AI response: {response_text[:200]}")


async def _call_gemini(text: str) -> dict:
    """Call Gemini API for credibility scoring. Returns {score, reasoning, error}."""
    import traceback

    # Skip if no valid API key
    if not _is_key_valid(GEMINI_API_KEY):
        return {"score": None, "reasoning": "", "error": "No Gemini API key configured. Add GEMINI_API_KEY to .env file."}

    # Rate limit check
    if not _gemini_rate_ok():
        return {"score": None, "reasoning": "", "error": "Gemini rate limit reached. Try again in a minute."}

    from google import genai as genai_new

    prompt = f"""Rate this news article's credibility from 0 (fake) to 100 (verified).
Return ONLY JSON: {{"score": <0-100>, "reasoning": "<one sentence>"}}

Article:
{text[:2000]}"""

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            _gemini_calls.append(_time.time())
            logger.info("Gemini API call (attempt %d/%d)", attempt + 1, max_retries)

            loop = asyncio.get_running_loop()

            def _generate():
                client = genai_new.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
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
            traceback.print_exc()
            last_error = str(e)
            is_retryable = ("429" in last_error or "503" in last_error
                            or "quota" in last_error.lower()
                            or "resource_exhausted" in last_error.lower()
                            or "unavailable" in last_error.lower()
                            or "overloaded" in last_error.lower())
            if is_retryable and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s exponential backoff
                logger.info("Gemini temporarily unavailable, retrying in %ds...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            break

    # Show a user-friendly error for 503/overload errors
    logger.warning("Gemini API call failed (REAL ERROR): %s", last_error)
    if last_error and ("503" in last_error or "unavailable" in last_error.lower()):
        return {"score": None, "reasoning": "", "error": "Gemini is currently overloaded with high demand. Try again in a few minutes."}
    return {"score": None, "reasoning": "", "error": last_error}


# ===================================================================
# Groq API (Fast Inference)
# ===================================================================

def _groq_rate_ok() -> bool:
    """Check if we're within Groq rate limits."""
    now = _time.time()
    _groq_calls[:] = [t for t in _groq_calls if now - t < 86400]
    if len(_groq_calls) >= GROQ_MAX_PER_DAY:
        return False
    recent = [t for t in _groq_calls if now - t < 60]
    if len(recent) >= GROQ_MAX_PER_MINUTE:
        return False
    return True

async def _call_groq(text: str) -> dict:
    """Call Groq API for credibility scoring. Returns {score, reasoning, error}."""
    # Skip if no valid API key
    if not _is_key_valid(GROQ_API_KEY):
        return {"score": None, "reasoning": "", "error": "No Groq API key configured. Add GROQ_API_KEY to .env file."}

    if not _groq_rate_ok():
        return {"score": None, "reasoning": "", "error": "Groq rate limit reached. Try again later."}

    from openai import OpenAI

    prompt = f"""Rate this news article's credibility from 0 (fake) to 100 (verified).
Return ONLY JSON: {{"score": <0-100>, "reasoning": "<one sentence>"}}

Article:
{text[:2000]}"""

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            _groq_calls.append(_time.time())
            logger.info("Groq API call (attempt %d/%d)", attempt + 1, max_retries)

            # Groq is OpenAI-compatible
            client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", timeout=30)
            loop = asyncio.get_running_loop()

            def _generate():
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150,
                )
                return response.choices[0].message.content

            response_text = await loop.run_in_executor(_executor, _generate)
            result = _parse_ai_json(response_text)
            score = max(0, min(100, float(result.get("score", 50))))
            reasoning = str(result.get("reasoning", ""))

            logger.info("Groq API SUCCESS: score=%s", score)
            return {"score": score, "reasoning": reasoning, "error": None}

        except Exception as e:
            import traceback
            traceback.print_exc()
            last_error = str(e)
            is_rate_error = ("429" in last_error or "rate_limit" in last_error.lower())
            if is_rate_error and attempt < max_retries - 1:
                logger.info("Groq rate limited, retrying in 2s...")
                await asyncio.sleep(2)
                continue
            break

    # Show the REAL error — don't mask it
    logger.warning("Groq API call failed (REAL ERROR): %s", last_error)
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
        credibility = round(result.get("probability_true", 0.0) * 100, 1)
        return {
            "model_name": model_name,
            "label": result.get("label", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "probability_fake": result.get("probability_fake", 0.0),
            "probability_true": result.get("probability_true", 0.0),
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
    groq_future = _call_groq(req.text)

    # Gather all results simultaneously
    ml_results = await asyncio.gather(*ml_futures)
    gemini_result = await gemini_future
    groq_result = await groq_future

    # Calculate weighted overall score
    valid_ml = [r for r in ml_results if r["label"] != "Error"]
    ml_avg = sum(r["credibility_score"] for r in valid_ml) / len(valid_ml) if valid_ml else 50.0

    # Calculate weighted overall score safely (handle None scores)
    score_sum = ml_avg * 0.5
    weight_sum = 0.5

    if gemini_result.get("score") is not None:
        score_sum += gemini_result["score"] * 0.25
        weight_sum += 0.25

    if groq_result.get("score") is not None:
        score_sum += groq_result["score"] * 0.25
        weight_sum += 0.25

    overall = round(score_sum / weight_sum, 1)

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
        "groq_score": groq_result["score"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, max_items=MAX_HISTORY)

    return AnalyzeResponse(
        models=[ModelScore(**r) for r in ml_results],
        gemini=GeminiScore(**gemini_result),
        groq=GroqScore(**groq_result),
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
        "groq_score": req.groq_score,
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

        # Handle specific HTTP errors with user-friendly messages
        if response.status_code == 402:
            raise HTTPException(
                status_code=422,
                detail="This article is behind a paywall and cannot be accessed. Please paste the article text directly instead."
            )
        if response.status_code == 403:
            raise HTTPException(
                status_code=422,
                detail="Access to this article was denied (403 Forbidden). The site may be blocking automated requests. Please paste the article text directly."
            )
        if response.status_code == 401:
            raise HTTPException(
                status_code=422,
                detail="This article requires authentication to access. Please paste the article text directly."
            )
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
            raise HTTPException(status_code=422, detail="Could not extract article text from this page. The site may use JavaScript rendering. Please paste the article text directly.")

        return FetchUrlResponse(text=text, title=title)

    except HTTPException:
        raise  # Re-raise our custom HTTPExceptions
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=422, detail="Request timed out. The site took too long to respond. Please try again or paste the article text directly.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=422, detail="Could not connect to the website. Please check the URL and try again.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {str(e)}")


# ===================================================================
# Headlines (NewsAPI)
# ===================================================================

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
_headlines_cache: dict = {"data": None, "timestamp": 0}
HEADLINES_CACHE_SECONDS = 600  # 10 minutes

@app.get("/newsfeed")
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
        resp = req_lib.get(url, params=params, timeout=20)
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
        logger.warning("NewsAPI (newsfeed) call failed: %s", e)
        # Return empty list instead of 502 to avoid UI disruption
        return {"articles": [], "totalResults": 0}


# ===================================================================
# Root
# ===================================================================

@app.get("/debug-ai")
async def debug_ai():
    """Diagnostic endpoint to verify AI key loading."""
    return {
        "gemini_key_loaded": bool(GEMINI_API_KEY) and len(GEMINI_API_KEY) > 10,
        "gemini_key_prefix": GEMINI_API_KEY[:10] + "..." if GEMINI_API_KEY else "EMPTY",
        "groq_key_loaded": bool(GROQ_API_KEY) and len(GROQ_API_KEY) > 10,
        "groq_key_prefix": GROQ_API_KEY[:10] + "..." if GROQ_API_KEY else "EMPTY",
        "news_key_loaded": bool(NEWS_API_KEY),
    }


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
