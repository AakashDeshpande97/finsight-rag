FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/hf_cache

# 1. CPU-only PyTorch (the default Linux build includes huge GPU libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Bake the embedding model into the image, then never contact Hugging Face again
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# 4. Data: vector database + company list
COPY chroma_db ./chroma_db
COPY data/archive/subset.json ./data/subset.json

# 5. Code last (changes most often -> fast rebuilds)
COPY *.py ./

ENV PORT=8080
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
