"""Run an evaluation file through the RAG pipeline and score it.

Usage:
    python evaluate.py                    # uses eval_set.json (synthetic)
    python evaluate.py hard_eval.json     # any other eval file

Retrieval metrics (only for questions that have a known source):
  - chunk hit: the exact source chunk was retrieved (strict)
  - page hit:  any chunk from the source page was retrieved (forgiving)

Answer metric:
  - key fact found in the answer  -> correct (cheap, no LLM call)
  - otherwise an LLM judge decides (only runs when needed)
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from llm import generate
from rag import answer_question

SLEEP_SECONDS = 13
EVAL_FILE = sys.argv[1] if len(sys.argv) > 1 else "eval_set.json"
RESULTS_FILE = EVAL_FILE.replace(".json", "_results.json")


class Verdict(BaseModel):
    correct: bool
    reason: str


JUDGE_PROMPT = """You are grading answers from a question-answering system about annual reports.

QUESTION: {question}
EXPECTED ANSWER: {expected}
SYSTEM ANSWER: {actual}

Is the system answer correct?
- It is correct if it contains the same key information as the expected answer.
  Different wording and extra correct details are fine.
- It is incorrect if it contradicts the expected answer or gives a different number.
- If the expected answer says the information is NOT in the report, the system answer is
  correct only if it also says the information is not available, without inventing an answer.
- If the expected answer contains information, a system answer saying the information is
  missing is incorrect."""

JUDGE_CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    response_schema=Verdict,
)


def normalise(text):
    """Lowercase and drop spaces, commas and $ so '$10,426' matches '10426'."""
    return re.sub(r"[\s,$]", "", text.lower())


def judge(item, answer):
    prompt = JUDGE_PROMPT.format(question=item["question"],
                                 expected=item["expected_answer"], actual=answer)
    response, _ = generate(prompt, JUDGE_CONFIG)
    time.sleep(SLEEP_SECONDS)
    return response.parsed


def mark(value):
    return "n/a " if value is None else ("OK  " if value else "MISS")


eval_set = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
print(f"Evaluating {len(eval_set)} questions from {EVAL_FILE}\n")
results = []

for i, item in enumerate(eval_set, 1):
    out = answer_question(item["question"], item["company"])
    time.sleep(SLEEP_SECONDS)

    retrieved_texts = [doc for doc, _, _ in out["results"]]
    retrieved_pages = [meta["page"] for _, meta, _ in out["results"]]
    chunk_hit = item["source_text"] in retrieved_texts if "source_text" in item else None
    page_hit = item["source_page"] in retrieved_pages if "source_page" in item else None

    key_fact = item.get("key_fact", "")
    if key_fact and normalise(key_fact) in normalise(out["answer"]):
        answer_ok, method, reason = True, "key fact", ""
    else:
        verdict = judge(item, out["answer"])
        answer_ok, method, reason = verdict.correct, "judge", verdict.reason

    qtype = item.get("type", "")
    print(f"{i:2}. chunk {mark(chunk_hit)} | page {mark(page_hit)} | "
          f"answer {mark(answer_ok)} ({method:8}) | {qtype[:12]:12} | {item['question'][:55]}")
    if method == "judge":
        print(f"      judge: {reason}")
    if not answer_ok:
        print(f"      system answer: {out['answer'][:300]}")

    results.append({
        "question": item["question"],
        "company": item["company"],
        "type": qtype,
        "expected_answer": item["expected_answer"],
        "rag_answer": out["answer"],
        "retrieved_pages": retrieved_pages,
        "chunk_hit": chunk_hit,
        "page_hit": page_hit,
        "answer_ok": answer_ok,
        "answer_method": method,
        "judge_reason": reason,
    })

# ---------- Summary ----------
n = len(results)
with_source = [x for x in results if x["page_hit"] is not None]
if with_source:
    m = len(with_source)
    chunk = sum(bool(x["chunk_hit"]) for x in with_source)
    page = sum(x["page_hit"] for x in with_source)
    print(f"\nRetrieval (exact chunk): {chunk}/{m} = {100 * chunk / m:.0f}%")
    print(f"Retrieval (source page): {page}/{m} = {100 * page / m:.0f}%")

ans = sum(x["answer_ok"] for x in results)
judged = sum(x["answer_method"] == "judge" for x in results)
print(f"\nAnswer accuracy: {ans}/{n} = {100 * ans / n:.0f}%  ({judged} decided by LLM judge)")

by_type = defaultdict(list)
for x in results:
    if x["type"]:
        by_type[x["type"]].append(x["answer_ok"])
for t, oks in sorted(by_type.items()):
    print(f"  {t:14} {sum(oks)}/{len(oks)}")

Path(RESULTS_FILE).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nDetails saved to {RESULTS_FILE}")