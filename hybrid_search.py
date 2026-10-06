import re
from rank_bm25 import BM25Okapi
from search import collection, search

STOPWORDS = {"what", "was", "were", "the", "for", "of", "a", "an", "is", "in", "to",
             "and", "did", "does", "do", "how", "much", "many", "year", "company"}


def tokenize(text):
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams

def keyword_search(question, company, k=30):
    """BM25 keyword search over one company's chunks."""
    data = collection.get(where={"company": company}, include=["documents", "metadatas"])
    bm25 = BM25Okapi([tokenize(doc) for doc in data["documents"]])
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(data["documents"][i], data["metadatas"][i]) for i in top]


def hybrid_search(question, company, k=5, candidates=30, rrf_k=60):
    """Combine vector and keyword results with Reciprocal Rank Fusion."""
    vector_results = search(question, company=company, k=candidates)
    keyword_results = keyword_search(question, company, k=candidates)

    scores, metas = {}, {}
    for rank, (doc, meta, _) in enumerate(vector_results, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (rrf_k + rank)
        metas[doc] = meta
    for rank, (doc, meta) in enumerate(keyword_results, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (rrf_k + rank)
        metas[doc] = meta

    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(doc, metas[doc], scores[doc]) for doc in best]


if __name__ == "__main__":
    q, c = "What was the net income for the year?", "ACRES Commercial Realty Corp."
    for name, results in [("VECTOR ONLY", search(q, company=c, k=5)),
                          ("KEYWORD ONLY", [(d, m, 0) for d, m in keyword_search(q, c, k=5)]),
                          ("HYBRID", hybrid_search(q, c))]:
        print(f"\n=== {name} ===")
        for doc, meta, _ in results:
            hit = "HIT" if "net income" in doc.lower() else "   "
            body = doc.split("\n\n", 1)[-1][:110].replace("\n", " ")
            print(f" {hit} p.{meta['page']:3} | {body}")