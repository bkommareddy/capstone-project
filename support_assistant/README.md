# Module 3 — Support Assistant

corpus, a LangGraph-orchestrated intent router, a structured JSON output
guarantee, and a FastAPI wrapper — served locally via `uvicorn` or Docker.

(or set to `1`), the service is fully deterministic and offline — no signup,
no API key, no network call to any LLM provider. This is the default and the
graded configuration.** Setting `MOCK_LLM=0` switches on the optional, ungraded
real-LLM extension (Groq's free tier).

## Setup

```bash
pip install -r requirements.txt
python ingest.py
```

`ingest.py` needs internet access the first time only, to download the
`all-MiniLM-L6-v2` model weights from Hugging Face (cached locally after
that — same pattern as Module 2's dataset caching). No ChromaDB account or
API key is required; it's a local, on-disk vector store.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How much does standard delivery cost for a small order?"}'
```

### Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The Dockerfile runs `ingest.py` at build time, so the image serves `/ask`
immediately on `docker run` with `MOCK_LLM` at its default (fully offline —
no key/secret needed for the required baseline).

## Architecture (ingestion → embedding → retrieval → generation)

1. **Ingestion** (`ingest.py`, `load_documents`): reads the 8 fixed policy
   documents from `docs/*.txt`. Each document is short and single-topic, so
   the chunking granularity is one chunk per document — no further splitting.

2. **Embedding** (`ingest.py`, `build_index`): each chunk is embedded locally
   with `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` — no
   API key, runs on-device. Embeddings are stored, along with the raw text and
   a `source` metadata field, in a persistent ChromaDB collection named
   `zepto_policies` on disk at `chroma_db/` (`chromadb.PersistentClient`).

3. **Retrieval** (`retrieval.py`, `retrieve_top_k`): embeds the incoming query
   with the same model and queries the ChromaDB collection for the top-3
   chunks by cosine similarity (`metadata={"hnsw:space": "cosine"}` set at
   collection creation). This step **always runs for real in both `MOCK_LLM`
   modes** — no key or network call needed once the model is cached.

4. **Generation** (`graph.py`): a LangGraph `StateGraph` with a `TypedDict`
   state (`AssistantState`) and 3 nodes:
   - `classify_intent` — keyword heuristic (mock) or LLM call (`MOCK_LLM=0`)
     to route the query as `policy_question` or `general_question`.
   - `retrieve_and_answer` — calls `retrieve_top_k`, then either returns the
     canned `"Based on the retrieved context: {snippet}"` template (mock) or
     prompts a real LLM using the structured template in `prompt_template.py`
     (`MOCK_LLM=0`).
   - `direct_answer` — returns a fixed canned string (mock) or prompts the LLM
     directly with no retrieval (`MOCK_LLM=0`).

   A conditional edge (`route_by_intent`) wires `classify_intent` to whichever
   of the two answer nodes matches the classification; this routing itself
   never depends on `MOCK_LLM`. `run_assistant()` invokes the compiled graph
   and packages the final state into the validated `AskResponse` Pydantic
   model (`schemas.py`) — `main.py`'s `POST /ask` endpoint returns exactly that.

**What changes with `MOCK_LLM`**: only the final *generation* step inside
`classify_intent`, `retrieve_and_answer`, and `direct_answer` branches on it
(see `_mock_llm_enabled()` in `graph.py`). Ingestion, embedding, and
retrieval are identical in both modes. In mock mode (default, graded), the
`AskResponse` fields are populated deterministically from code — `sources` is
the retrieved chunk IDs (or empty for `general_question`), `confidence` is a
fixed `1.0`. In the `MOCK_LLM=0` extension, `llm_client.py` calls a real LLM
(Groq's free tier by default) and validates/retries the raw output against
`AskResponse` up to 2 additional times before returning a clearly-marked
error response (`call_llm_for_structured_response`).

```
docs/*.txt --(ingest.py: chunk + embed)--> ChromaDB (chroma_db/)
                                                  |
query --(retrieval.py: embed + cosine search)-----+--> top-3 chunks
                                                  |
                          graph.py (LangGraph) ----+
                          classify_intent -> retrieve_and_answer -> AskResponse
                                          \-> direct_answer -------/
                                                  |
                                          main.py POST /ask
```
