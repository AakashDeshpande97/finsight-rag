"""Run every eval question through the RAG pipeline and score it.
Two separate scores:
  - retrieval hit: was the source chunk among the retrieved chunks?
  - answer hit:    does the final answer contain the key fact?"""
import json
import re
import time
from pathlib import Path

from rag import answer_question

SLEEP_SECONDS = 13


def normalise(text):
    """Lowercase and drop spaces, commas and $ so '$10,426' matches '10426'."""
    return re.sub(r"[\s,$]", "", text.lower())


eval_set = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))
results = []

for i, item in enumerate(eval_set, 1):
    out = answer_question(item["question"], item["company"])
    retrieved_texts = [doc for doc, _, _ in out["results"]]
    retrieval_hit = item["source_text"] in retrieved_texts
    answer_hit = normalise(item["key_fact"]) in normalise(out["answer"])

    print(f"{i:2}. retrieval {'OK  ' if retrieval_hit else 'MISS'} | "
          f"answer {'OK  ' if answer_hit else 'MISS'} | {item['question'][:75]}")
    results.append({
        "question": item["question"],
        "company": item["company"],
        "key_fact": item["key_fact"],
        "expected_answer": item["expected_answer"],
        "source_page": item["source_page"],
        "rag_answer": out["answer"],
        "retrieved_pages": [meta["page"] for _, meta, _ in out["results"]],
        "retrieval_hit": retrieval_hit,
        "answer_hit": answer_hit,
    })
    time.sleep(SLEEP_SECONDS)

n = len(results)
r = sum(x["retrieval_hit"] for x in results)
a = sum(x["answer_hit"] for x in results)
print(f"\nRetrieval hit rate: {r}/{n} = {100 * r / n:.0f}%")
print(f"Answer accuracy:    {a}/{n} = {100 * a / n:.0f}%")

Path("eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("Details saved to eval_results.json")
