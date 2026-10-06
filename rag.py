import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from search import search

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]

SYSTEM_PROMPT = """You answer questions about company annual reports.
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- After every fact, cite its source like [1] or [2].
- If the sources do not contain the answer, say exactly:
  "The provided sources do not contain this information." Do not guess.
- Quote numbers exactly as written, including units (e.g. "in thousands")."""


def build_prompt(question, results):
    sources = "\n\n".join(
        f"[{i}] ({meta['company']}, page {meta['page']})\n{doc}"
        for i, (doc, meta, _) in enumerate(results, 1)
    )
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}"


def generate(prompt):
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)
    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                return response.text, model
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                print(f"  ({model}: rate limit, waiting 30s)")
                time.sleep(30)
            except errors.ServerError:
                print(f"  ({model}: busy, waiting 5s)")
                time.sleep(5)
    return "LLM unavailable, try again later.", None


def ask(question, company, k=5):
    results = search(question, company=company, k=k)          # R: retrieve
    answer, model = generate(build_prompt(question, results))  # A + G: augment + generate
    print(f"\n### Q: {question} ({company})")
    print(answer)
    print("\nSources:")
    for i, (_, meta, dist) in enumerate(results, 1):
        print(f"  [{i}] {meta['company']}, page {meta['page']} (distance {dist:.3f})")
    print(f"(answered by {model})")


if __name__ == "__main__":
    ask("How much notice must the CEO give to resign?", "Downer EDI Limited")
    ask("What risks does the company face from clinical trials?", "Aptevo Therapeutics Inc.")
    ask("What was the net income for the year?", "ACRES Commercial Realty Corp.")