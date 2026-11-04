# FinSight RAG
[![CI](https://github.com/AakashDeshpande97/finsight-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AakashDeshpande97/finsight-rag/actions/workflows/ci.yml)


**Question answering over 100 corporate annual reports, with page-level citations.**

Ask a question like *"How much profit did ACRES Commercial Realty make in 2022?"* and get a grounded answer with the exact PDF pages it came from, or an honest "the sources do not contain this information" when the answer isn't there.

The service was deployed on Google Cloud Run (europe-west1) and torn down afterwards to avoid costs. Screenshots of the live deployment are below; the [deploy commands](#run-it) recreate it in about 15 minutes.

### Screenshots

| | |
|---|---|
| ![API docs](docs/screenshots/01_docs.png) | ![Answer with citations](docs/screenshots/02_answer.png) |
| Interactive API docs on Cloud Run | Grounded answer citing the income statement (PDF page 95) |
| ![Unknown company](docs/screenshots/03_did_you_mean.png) | ![Cloud Run console](docs/screenshots/04_cloud_run.png) |
| Helpful 404 with `did_you_mean` suggestions | Cloud Run service in the GCP console |

**Tech:** Python, Gemini API, sentence-transformers (`bge-small-en-v1.5`), ChromaDB, BM25, FastAPI, Pydantic, Docker, Google Cloud Run

---

## How it works

```mermaid
flowchart LR
    subgraph Ingestion [Ingestion - run once]
        A[100 annual reports<br/>Markdown] --> B[Clean + split by page]
        B --> C[Table-aware chunking<br/>+ metadata]
        C --> D[Embed<br/>bge-small, local]
        D --> E[(ChromaDB<br/>58,670 chunks)]
    end

    subgraph Serving [Serving - per question]
        Q[Question + company] --> R[LLM query rewriting]
        R --> S1[Vector search]
        R --> S2[BM25 keyword search]
        E -.-> S1
        E -.-> S2
        S1 --> F[Reciprocal Rank Fusion]
        S2 --> F
        F --> G[Gemini: answer only<br/>from sources, cite pages]
        G --> H[Answer + sources]
    end
```

1. **Ingestion.** Reports are cleaned (image links, HTML tags, escaped characters, table padding), split by page so every chunk knows its PDF page, then chunked into ~1,200-character pieces. Tables are never cut in half; large tables are split by rows with the header repeated. Each chunk gets a `Company | Page` header and metadata (company, industry, page).
2. **Embedding.** Chunks are embedded locally with `BAAI/bge-small-en-v1.5` (512-token input limit, so long chunks are not silently truncated as with 256-token models) and stored in ChromaDB.
3. **Retrieval.** Every search is filtered to the requested company. The question is rewritten by an LLM into 2 to 3 queries in annual-report vocabulary ("profit" becomes "net income"). Each query runs both vector search and BM25 keyword search, and all ranked lists are merged with Reciprocal Rank Fusion.
4. **Generation.** Gemini answers using only the top 5 chunks, cites every fact as `[1]`, `[2]`, keeps units such as "in thousands", and refuses when the sources don't contain the answer.

---

## Design decisions and what drove them

Each improvement was made because a measurement showed a specific failure.

| Problem observed | Cause | Fix |
|---|---|---|
| "What was the net income?" returned taxable income, cash, interest income, but never the income statement | Financial tables (mostly numbers and `\|` characters) produce weak embeddings, so they rank far below prose | **Hybrid search**: added BM25 keyword search |
| BM25 alone still ranked the wrong chunks | It scored "net" and "income" separately, and numbers made tables look "long", which BM25 penalises | **Custom tokenizer**: drop numbers and stopwords, add bigrams (`net_income`) |
| Numbers attributed to the wrong company | Search without filtering mixes reports | **Metadata filtering** by company before search |
| "How much profit…?" failed, though the answer was in the report | Users say "profit", reports say "net income": both keyword and vector search miss | **LLM query rewriting** with multi-query RRF fusion |
| Answers marked wrong although correct | Exact key-fact matching is too strict ("stockholder approval is not required" vs "without stockholder approval") | **LLM-as-judge** for non-exact matches, plus page-level retrieval metric |

---

## Evaluation

Two test sets, scored separately for retrieval and answers:

| Test set | Questions | Result |
|---|---|---|
| Synthetic (LLM-generated from random chunks, 5 reports, quality-gated) | 20 | Retrieval: 100% at page level (95% exact chunk). Answers: 100% (2 decided by LLM judge) |
| Hand-written hard set (paraphrases, a sign-flip trap, unanswerable questions) | 8 | 75% with hybrid search, **100% with query rewriting and 8 chunks of context** |

Honest caveats:
- These are small sets. Synthetic questions are easier than real user questions, so treat the scores as upper bounds.
- Results vary between runs (model updates, non-deterministic rewriting). With 5 chunks of context, failures moved between questions from run to run; raising the context to 8 chunks made the hard set pass consistently.
- Every answer marked by the LLM judge was spot-checked by hand.

Run it yourself:

```bash
python evaluate.py                  # synthetic set
python evaluate.py hard_eval.json   # hard set
```

---

## Observability and CI

- **Langfuse tracing:** every question is traced step by step (query rewriting, each hybrid search, each Gemini call) with latency and token usage. A typical question takes about 2 seconds, of which the two LLM calls take about 70%.
- **Evaluation scores in Langfuse:** each evaluation run attaches `answer_correct`, `page_hit` and `chunk_hit` scores to the question's trace, so a failure can be diagnosed directly from the trace. Example: a failing question about common share *distributions* showed that query rewriting had replaced the report's term with *dividends* in every query.
- **CI (GitHub Actions):** unit tests for cleaning, table-aware chunking, the BM25 tokenizer and company-name suggestions run on every push to `main` and every pull request. The LLM evaluations need API keys and the vector index, so they run manually.

---

## v3: Agent over structured and unstructured data

A LangGraph agent decides per question which tool to use:

| Tool | Data | Used for |
|---|---|---|
| `query_financials` | SQLite table of key figures (revenue, net income, total assets, employees, with source pages) extracted from all 100 reports | Rankings, comparisons, counts across companies |
| `search_reports` | The RAG pipeline above | Explanations, risks, policies, details |

- **Extraction:** one structured-output call per report (Pydantic schema), unit conversion done in code, resumable with checkpointing. Coverage: net income for 91/100 reports, revenue and total assets for 66/100; spot-checked by hand for 3 companies.
- **Safety:** the SQL tool uses a read-only connection and accepts only SELECT queries; the agent loop is capped.
- **Memory:** multi-turn conversations via a LangGraph checkpointer and `thread_id` (`chat.py`, `POST /agent`).
- **Evaluation:** 7 questions covering SQL, document, combined and unanswerable cases, scored for tool choice and answer correctness (7/7 on both), with scores in Langfuse.

Example: *"Which USD company had the biggest loss, and what caused it?"* → SQL finds Aurora Innovation (net loss $1.72B); when the first document search fails, the agent reformulates and finds the $1,114M goodwill impairment.

---

## Known limitations

- **Broad questions** ("What are the main risks?") get narrow answers: 8 chunks cannot summarise a whole risk section, and near-duplicate chunks sometimes waste context. Next: deduplication and adaptive `k`.
- **~2% of chunks** exceed the embedding model's 512-token limit and are truncated.
- **Index baked into the container image** (~920 MB compressed). Fine for a demo; in production, ingestion and serving would be separated, with a managed vector database (e.g. PostgreSQL + pgvector) so new reports can be added without redeploying.
- **Dataset labels are imperfect** (e.g. a gold explorer labelled "Pharmaceuticals"); answers come from report text, not labels.
- **The demo endpoint has no authentication**; cost was capped with `--max-instances 2`. Production would add API keys or IAM and Secret Manager for the Gemini key.
- **The agent can carry a wrong assumption across turns**, e.g. guessing a fiscal year it never retrieved.
- **Agent memory is in-process** (`InMemorySaver`), so conversations are lost on restart; production would need a database-backed checkpointer.
- **Amounts in 10 currencies are not converted**, so cross-currency comparisons are refused rather than computed.
- **The deployed Cloud Run version is v1**; the agent and Langfuse tracing have only been run locally so far.

## Project structure

| File | Purpose |
|---|---|
| `explore.py` | Inspect the dataset |
| `ingest.py` | Load, clean and chunk reports |
| `embed_store.py` | Embed chunks and store them in ChromaDB |
| `search.py` | Vector search with company filtering |
| `hybrid_search.py` | BM25 + vector search fused with RRF |
| `query_rewrite.py` | LLM query rewriting and multi-query fusion |
| `llm.py` | Gemini calls with retry (429) and model fallback (503) |
| `rag.py` | Full pipeline: retrieve, build prompt, generate |
| `api.py` | FastAPI service: `/ask`, `/companies`, `/health` |
| `make_eval.py` | Generate the synthetic evaluation set |
| `evaluate.py` | Score retrieval and answers (key fact + LLM judge) |
| `eval_set.json`, `hard_eval.json` | Evaluation sets |

---

## Run it

**Data:** [Enterprise RAG Challenge dataset (Kaggle)](https://www.kaggle.com/datasets/rrr3try/enterprise-rag-markdown). Download and unzip into `data/`. It is not included in this repo.

**Locally:**

```bash
pip install -r requirements.txt
echo GEMINI_API_KEY=your_key > .env
python embed_store.py          # one-off, about 1.5 hours on CPU for all 100 reports
uvicorn api:app                # then open http://127.0.0.1:8000/docs
```

**Docker:**

```bash
docker build -t rag-api .
docker run -p 8080:8080 --env-file .env rag-api
```

**Google Cloud Run:**

```bash
docker tag rag-api europe-west1-docker.pkg.dev/PROJECT_ID/rag-repo/rag-api:v1
docker push europe-west1-docker.pkg.dev/PROJECT_ID/rag-repo/rag-api:v1
gcloud run deploy rag-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/rag-repo/rag-api:v1 \
  --region europe-west1 --memory 2Gi --cpu 2 --timeout 120 \
  --min-instances 0 --max-instances 2 --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

**Tear down** (to stop all costs):

```bash
gcloud run services delete rag-api --region europe-west1
gcloud artifacts repositories delete rag-repo --location europe-west1
```

---

## Roadmap

- [x] v1: RAG pipeline, hybrid search, query rewriting, evaluation, Cloud Run deployment
- [x] v2: Langfuse tracing and evaluation scores, unit tests and CI (GitHub Actions)
- [x] v3: LangGraph agent combining document search with SQL over extracted financial figures
- [ ] v4: Second deployment on Azure Container Apps
