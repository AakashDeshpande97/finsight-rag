"""RAG pipeline: hybrid retrieval + Gemini generation with citations."""
from google.genai import types

from hybrid_search import hybrid_search
from llm import generate

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


def answer_question(question, company, k=5):
    """Run the full RAG pipeline and RETURN the result (used by the API and evaluation)."""
    results = hybrid_search(question, company, k=k)                  # R: retrieve
    response, model = generate(build_prompt(question, results), CONFIG)  # A + G
    return {"answer": response.text, "results": results, "model": model}


def ask(question, company, k=5):
    """Run the pipeline and PRINT the result (for manual testing)."""
    out = answer_question(question, company, k)
    print(f"\n### Q: {question} ({company})")
    print(out["answer"])
    print("\nSources:")
    for i, (_, meta, score) in enumerate(out["results"], 1):
        print(f"  [{i}] {meta['company']}, PDF page {meta['page']} (score {score:.4f})")
    print(f"(answered by {out['model']})")


if __name__ == "__main__":
    ask("How much notice must the CEO give to resign?", "Downer EDI Limited")
    ask("What risks does the company face from clinical trials?", "Aptevo Therapeutics Inc.")
    ask("What was the net income for the year?", "ACRES Commercial Realty Corp.")
