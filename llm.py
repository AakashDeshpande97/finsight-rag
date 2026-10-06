"""Shared Gemini helper: one place for retry + fallback logic."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]


def generate(contents, config):
    """Call Gemini with retry on rate limits (429) and fallback on overload (503).
    Returns (response, model_name)."""
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                return response, model
            except errors.ClientError as e:
                if e.code != 429:
                    raise                      # bad key, bad model name... fail loudly
                print(f"  ({model}: rate limit, waiting 30s)")
                time.sleep(30)
            except errors.ServerError:
                print(f"  ({model}: busy, waiting 5s)")
                time.sleep(5)
    raise RuntimeError("All Gemini models unavailable - try again later")
