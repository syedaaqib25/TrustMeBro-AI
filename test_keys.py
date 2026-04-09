"""Quick test for Gemini and Groq (Fast Inference) API keys."""
import os, traceback
from dotenv import load_dotenv
load_dotenv()

# Test 1: Gemini
print("=" * 50)
print("TESTING GEMINI (gemini-2.5-flash)...")
print("=" * 50)
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in 5 words"
    )
    print("GEMINI SUCCESS:", r.text.strip())
except Exception as e:
    traceback.print_exc()
    print(f"\nGEMINI FAILED: {type(e).__name__}: {str(e)[:200]}")

# Test 2: Groq
print()
print("=" * 50)
print("TESTING GROQ (Fast Inference)...")
print("=" * 50)
try:
    from openai import OpenAI
    # Groq uses OpenAI-compatible SDK
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        timeout=15
    )
    
    # List models first to verify key
    print("Listing models to verify key...")
    models = client.models.list()
    model_names = [m.id for m in models.data]
    print(f"Key works! Available models: {model_names}")
    
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
        max_tokens=20,
    )
    print("GROQ SUCCESS:", r.choices[0].message.content.strip())
except Exception as e:
    traceback.print_exc()
    print(f"\nGROQ FAILED: {type(e).__name__}: {str(e)[:300]}")
