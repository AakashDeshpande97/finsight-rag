"""Shared Gemini helper: one place for retry + fallback logic, traced as an LLM generation."""
import os
import time

from google import genai
from google.genai import errors

from tracing import langfuse, observe

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]


@observe(name="gemini", as_type="generation")
def generate(contents, config):
    """Call Gemini with retry on rate limits (429) and fallback on overload (503).
    Returns (response, model_name)."""
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                usage = response.usage_metadata
                langfuse.update_current_generation(
                    model=model,
                    usage_details={
                        "input": usage.prompt_token_count or 0,
                        "output": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
                    },
                    metadata={"attempt": attempt + 1},
                )
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