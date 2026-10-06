"""Create a synthetic evaluation set: Gemini writes one question per randomly chosen chunk.
Because we know which chunk each question came from, we can later check retrieval AND answers."""
import json
import random
import time
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from ingest import build_chunks, find
from llm import generate

N_COMPANIES = 5
PER_COMPANY = 4          # 5 x 4 = 20 questions
MAX_TRIES = 8            # per company, so we never burn the whole free quota
SLEEP_SECONDS = 13       # stay under the free tier's requests-per-minute limit
random.seed(42)          # same random chunks every run -> reproducible


class QA(BaseModel):
    usable: bool = Field(description="False if the passage has no specific fact worth asking about "
                                     "(e.g. table of contents, only headings, boilerplate)")
    question: str
    answer: str
    key_fact: str = Field(description="Shortest exact phrase or number from the passage that a correct "
                                      "answer must contain. Max 6 words, copied verbatim.")


PROMPT = """Below is a passage from the annual report of {company}.
Write ONE question a financial analyst might ask that can be answered from this passage alone.
- The question must be specific and self-contained. Never say "in this passage" or "in the table".
- Prefer concrete facts: numbers, dates, names, durations, policies.
- key_fact must be copied exactly from the passage.
- If the passage contains no meaningful fact, set usable to false.

PASSAGE:
{text}"""

CONFIG = types.GenerateContentConfig(
    temperature=0.7,                      # some variety in question style
    response_mime_type="application/json",
    response_schema=QA,
)

companies = json.loads(find("subset.json").read_text(encoding="utf-8"))[:N_COMPANIES]
eval_set = []

for c in companies:
    chunks = [ch for ch in build_chunks(c) if len(ch["text"]) > 400]   # skip tiny chunks
    random.shuffle(chunks)
    found = 0
    for ch in chunks[:MAX_TRIES]:
        if found == PER_COMPANY:
            break
        body = ch["text"].split("\n\n", 1)[-1]          # passage without our "Company | Page" header
        response, _ = generate(PROMPT.format(company=c["company_name"], text=body), CONFIG)
        qa = response.parsed
        time.sleep(SLEEP_SECONDS)

        if qa is None or not qa.usable:
            print(f"  skipped: not usable (p.{ch['metadata']['page']})")
            continue
        if qa.key_fact.lower() not in body.lower():     # quality gate: key fact must really be there
            print(f"  skipped: key_fact not found in passage ({qa.key_fact!r})")
            continue

        eval_set.append({
            "company": c["company_name"],
            "question": qa.question,
            "expected_answer": qa.answer,
            "key_fact": qa.key_fact,
            "source_page": ch["metadata"]["page"],
            "source_id": ch["id"],
            "source_text": ch["text"],
        })
        found += 1
        print(f"  [{c['company_name'][:22]}] {qa.question}  ->  key fact: {qa.key_fact}")

Path("eval_set.json").write_text(json.dumps(eval_set, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved {len(eval_set)} questions to eval_set.json")
