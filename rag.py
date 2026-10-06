"""RAG pipeline: (query rewriting) + hybrid retrieval + Gemini generation with citations."""
from google.genai import types

from hybrid_search import hybrid_search
from llm import generate
from query_rewrite import multi_query_search

SYSTEM_PROMPT = """You answer questions about company annual reports.
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- After every fact, cite its source like [1] or [2].
- If the sources do not contain the answer, say exactly:
  "The provided sources do not contain this information." Do not guess.
- Quote numbers exactly as written, including units (e.g. "in thousands")."""

CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)


def build_prompt(question, results):
    sources = "\n\n".join(
        f"[{i}] ({meta['company']}, PDF page {meta['page']})\n{doc}"
        for i, (doc, meta, _) in enumerate(results, 1)
    )
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}"


def answer_question(question, company, k=5, rewrite=True):
    """Run the full RAG pipeline and RETURN the result (used by the API and evaluation)."""
    if rewrite:
        results, queries = multi_query_search(question, company, k=k)   # R (with rewriting)
    else:
        results, queries = hybrid_search(question, company, k=k), [question]
    response, model = generate(build_prompt(question, results), CONFIG)   # A + G
    return {"answer": response.text, "results": results, "queries": queries, "model": model}


def ask(question, company, k=5, rewrite=True):
    """Run the pipeline and PRINT the result (for manual testing)."""
    out = answer_question(question, company, k, rewrite)
    print(f"\n### Q: {question} ({company})")
    print(f"Search queries used: {out['queries']}")
    print(out["answer"])
    print("\nSources:")
    for i, (_, meta, score) in enumerate(out["results"], 1):
        print(f"  [{i}] {meta['company']}, PDF page {meta['page']} (score {score:.4f})")
    print(f"(answered by {out['model']})")


if __name__ == "__main__":
    ask("How much profit did the company make in 2022?", "ACRES Commercial Realty Corp.")
    ask("Did the company make a profit for its common shareholders in 2022?", "ACRES Commercial Realty Corp.")