import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("annual_reports")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(question, company=None, k=5):
    q_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=k,
                               where={"company": company} if company else None)
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))


def show(question, company=None):
    print(f"\n### Q: {question} | filter: {company or 'none'}")
    for doc, meta, dist in search(question, company):
        snippet = doc.split("\n\n", 1)[-1][:150].replace("\n", " ")
        print(f"  [{dist:.3f}] {meta['company'][:25]} p.{meta['page']}: {snippet}")


if __name__ == "__main__":
    show("What was the net income for the year?")
    show("What was the net income for the year?", company="ACRES Commercial Realty Corp.")
    show("How much notice must the CEO give to resign?", company="Downer EDI Limited")
    show("What risks does the company face from clinical trials?", company="Aptevo Therapeutics Inc.")