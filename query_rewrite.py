"""Query rewriting: translate the user's everyday words into annual-report vocabulary,
search with every version, and fuse the results with Reciprocal Rank Fusion."""
from google.genai import types
from pydantic import BaseModel, Field

from hybrid_search import hybrid_search
from llm import generate


class SearchQueries(BaseModel):
    queries: list[str] = Field(description="2-3 short search queries using annual-report terminology")


PROMPT = """A user asks a question about a company's annual report.
Rewrite it into 2-3 short search queries that use the exact terminology annual reports use.
Examples of translations: "profit" -> "net income"; "profit for shareholders" -> "net income
allocable to common shares"; "sales" -> "revenue"; "staff" -> "employees"; "boss" -> "Chief
Executive Officer"; "fire" -> "terminate employment".
Keep each query short and focused. Do not answer the question.

QUESTION: {question}"""

CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    response_schema=SearchQueries,
)


def rewrite(question):
    response, _ = generate(PROMPT.format(question=question), CONFIG)
    return response.parsed.queries


def multi_query_search(question, company, k=5, per_query=10, rrf_k=60):
    """Search with the original question + rewritten queries, fuse with RRF.
    Returns (results, queries) so we can log which queries were used."""
    queries = [question] + rewrite(question)          # always keep the original question
    scores, metas = {}, {}
    for q in queries:
        for rank, (doc, meta, _) in enumerate(hybrid_search(q, company, k=per_query), 1):
            scores[doc] = scores.get(doc, 0) + 1 / (rrf_k + rank)
            metas[doc] = meta
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(doc, metas[doc], scores[doc]) for doc in best], queries
