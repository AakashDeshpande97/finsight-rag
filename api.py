"""Annual Report Q&A API
Ask questions about 100 company annual reports and get answers with cited sources.

Run locally:   uvicorn api:app
Docs:          http://127.0.0.1:8000/docs
"""
import difflib
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingest import find
from rag import answer_question

# ---------- Company list (loaded once at startup) ----------
COMPANIES = sorted(c["company_name"] for c in json.loads(find("subset.json").read_text(encoding="utf-8")))
BY_LOWER = {name.lower(): name for name in COMPANIES}   # case-insensitive lookup

app = FastAPI(
    title="Annual Report Q&A",
    description="RAG over 100 company annual reports: hybrid search (vector + BM25), "
                "query rewriting, and Gemini answers with page-level citations.",
    version="1.0",
)


# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, description="Your question about the report")
    company: str = Field(description="Exact company name (see GET /companies)")
    rewrite: bool = Field(default=True, description="Use LLM query rewriting (better recall, one extra LLM call)")


class Source(BaseModel):
    number: int
    company: str
    pdf_page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    search_queries: list[str]
    model: str | None


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "companies": len(COMPANIES)}


@app.get("/companies")
def companies():
    """All company names the API can answer questions about."""
    return {"count": len(COMPANIES), "companies": COMPANIES}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    company = BY_LOWER.get(req.company.strip().lower())
    if company is None:
        suggestions = difflib.get_close_matches(req.company, COMPANIES, n=3, cutoff=0.3)
        raise HTTPException(status_code=404,
                            detail={"error": f"Unknown company: {req.company}", "did_you_mean": suggestions})
    try:
        out = answer_question(req.question, company, rewrite=req.rewrite)
    except RuntimeError as e:                       # all Gemini models unavailable
        raise HTTPException(status_code=503, detail=str(e))

    sources = [
        Source(number=i, company=meta["company"], pdf_page=meta["page"],
               excerpt=doc.split("\n\n", 1)[-1][:300])
        for i, (doc, meta, _) in enumerate(out["results"], 1)
    ]
    return AskResponse(answer=out["answer"], sources=sources,
                       search_queries=out["queries"], model=out["model"])
