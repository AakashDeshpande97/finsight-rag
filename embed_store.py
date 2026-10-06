import json
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import build_chunks, find

N_COMPANIES = 100
MODEL_NAME = "BAAI/bge-small-en-v1.5"

companies = json.loads(find("subset.json").read_text(encoding="utf-8"))[:N_COMPANIES]
chunks = [ch for c in companies for ch in build_chunks(c)]
print(f"{len(chunks)} chunks from {len(companies)} companies")

model = SentenceTransformer(MODEL_NAME)
print(f"Model reads at most {model.max_seq_length} tokens per chunk")

too_long = sum(len(model.tokenizer.encode(ch["text"])) > model.max_seq_length for ch in chunks)
print(f"Chunks longer than that (will be truncated): {too_long}")

embeddings = model.encode([ch["text"] for ch in chunks], batch_size=32,
                          show_progress_bar=True, normalize_embeddings=True)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("annual_reports")

BATCH = 1000
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    collection.upsert(
        ids=[ch["id"] for ch in batch],
        documents=[ch["text"] for ch in batch],
        metadatas=[ch["metadata"] for ch in batch],
        embeddings=embeddings[i:i + BATCH].tolist(),
    )
print(f"Stored {collection.count()} chunks in ChromaDB")