# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions
through dialogue, built for the SHL AI Intern take-home assignment.

## Architecture

```
OFFLINE
Scraper (Playwright) -> Clean -> Enrich (tags) -> catalog.json
                                        |
                          Precompute embeddings + BM25 validation

RUNTIME (stateless, full history rebuilt every call)
POST /chat
   -> Guardrail Gate (code, regex)          -- injection/legal/off-topic
   -> Slot Extractor (rule-based + LLM fallback if stalled)
   -> Decision Controller                   -- clarify/recommend/refine/compare/refuse
   -> Hybrid Retrieval (BM25 + embeddings + RRF)
   -> Metadata Filter (with relax-and-retry fallback)
   -> Conditional Cross-Encoder (only if >10 candidates survive filtering)
   -> LLM (Groq, structured JSON output)    -- grounded generation only
   -> Validation (catalog membership, schema, dedup)
   -> FastAPI response
```

Stack: FastAPI, Pydantic, Groq (`openai/gpt-oss-120b`), sentence-transformers
(`BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2`), rank-bm25.
No database — catalog is a JSON file loaded into memory (~hundreds of items,
no scale justification for one).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your GROQ_API_KEY
```

## Build pipeline (run once, before first deploy)

```bash
# 1. Fetch SHL's catalog (direct JSON feed, no browser needed)
cd scripts/scraper
python scrape_catalog.py
cd ../..
# -> data/raw/shl_catalog_raw.json

# 2. Run the full indexing pipeline (clean -> enrich -> embed -> validate)
python scripts/indexing/build_indexes.py
# -> data/processed/catalog.json, data/embeddings/*
```

**After step 2, check `data/processed/excluded_as_job_solutions.json`** — the
cleaning step filters out items that look like "Pre-packaged Job Solutions"
(out of scope per the assignment) using a name-pattern heuristic since the
feed doesn't explicitly flag this distinction. Spot-check that list.

## Run locally

```bash
uvicorn app.main:app --reload
```

## Evaluate before submission

```bash
# Latency FIRST — an architecturally perfect system that times out scores zero
python scripts/evaluation/latency.py --url http://localhost:8000

# Scripted behavior probes (refusal, no premature recommend, turn cap, etc.)
python scripts/evaluation/behavior_tests.py --url http://localhost:8000

# Full replay against the 10 provided conversation traces + Recall@10
# (adjust scripts/evaluation/replay_harness.py's load_trace() to match
# the actual downloaded trace file format first)
python scripts/evaluation/replay_harness.py --traces-dir data/traces --url http://localhost:8000
```

## Run tests

```bash
pytest tests/ -v
```

## Deploy

Deploy config included for **Render** (`render.yaml`) — connect the repo in the
Render dashboard, it will auto-detect the blueprint. Set `GROQ_API_KEY` manually
in the dashboard's environment variables (never commit real keys). A `Procfile`
is also included for Railway/Heroku-style platforms as an alternative.

Before deploying, ensure:
- `data/processed/catalog.json` and `data/embeddings/*` are included in
  the deployed artifact (not gitignored — see `.gitignore`, these are
  intentionally NOT excluded since the running service needs them)
- The embedding model download (first cold start) doesn't blow the
  2-minute wake-up allowance — verify this on the actual host, not just
  locally, since HuggingFace download speed varies by host network.
- After deploying, immediately run `scripts/evaluation/latency.py --url <your-render-url>`
  against the live URL, not just localhost — free-tier CPU is slower than
  most dev machines and this is where a timeout is most likely to surface.
