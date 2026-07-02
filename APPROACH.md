# Approach Document — SHL Assessment Recommender

## Design choices

The core design goal was a lean, defensible pipeline: every component maps to a
named requirement in the assignment (schema compliance, Recall@10, or a specific
behavior probe), rather than adding architecture for its own sake. Three
technologies considered and deliberately cut: **LangChain** (no unique capability
over the raw Groq SDK + Pydantic for this scope — a handful of fixed prompts and
native JSON-schema structured output don't need a framework layer), a **vector
database** (ChromaDB/similar — the catalog is a few hundred items; an in-memory
NumPy array is faster to load and equally correct at this scale), and an
**always-on cross-encoder** (kept, but made conditional — see Retrieval Setup).

**Stateless design.** Every `/chat` call rebuilds all state from the full
`messages` history — no server-side session store. Slot extraction, retrieval,
and the decision controller all re-derive from scratch each call, which is more
compute per request but the only design compatible with the assignment's
explicit "stores no per-conversation state" requirement.

**Agent orchestration.** Rather than a general-purpose agent framework, the
conversational logic is a small, explicit decision controller
(`app/decision/controller.py`) that routes each turn to one of five behaviors
(clarify/recommend/refine/compare/refuse) based on: a code-level guardrail gate,
rule-based slot extraction over the full conversation, and simple heuristics for
refine/compare detection. This keeps the "when to ask vs. retrieve vs. answer vs.
refuse" decision explicit and inspectable, rather than left implicit inside a
single large LLM call.

## Retrieval setup

**Hybrid retrieval**: BM25 (`rank-bm25`) for lexical/keyword matches (catches
exact product codes and tech terms embeddings can miss) plus dense embeddings
(`BAAI/bge-small-en-v1.5`) for semantic matches, merged via **Reciprocal Rank
Fusion** (k=60) rather than raw score averaging, since BM25 and cosine-similarity
scores live on incomparable scales.

**Catalog-side enrichment, not query expansion.** An earlier design considered
expanding queries at retrieval time (e.g. "Backend Developer" → "Java,
Microservices") but this bakes in unstated assumptions about the specific user's
stack — a real hallucination risk given the behavior-probe scoring on
hallucination rate. Instead, each catalog item is enriched **offline, once**
with LLM-generated tags describing what the assessment itself covers (e.g. "Java
8 Test" → tags: Java, Backend, API, REST, Spring). This improves recall the same
way query expansion would, without ever assuming anything about the user's
query.

**Conditional cross-encoder reranking.** Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
only runs when more than 10 candidates survive metadata filtering, and its input
is hard-capped at 25 candidates regardless of pool size. Below that threshold,
RRF ordering is already close enough to final that reranking would spend latency
reshuffling a short list — a bad trade against the 30-second-per-call budget.
This was the single biggest latency-vs-quality tradeoff in the design: an
always-on cross-encoder would add a full extra model inference pass on every
recommend turn, which is exactly the turn where Recall@10 is measured and where
staying under budget matters most.

**Metadata filtering with a relax-and-retry fallback.** Seniority and test-type
constraints filter the fused candidate pool, but every filter has a
"would this leave too few candidates?" safety check — an over-narrow filter is a
bigger threat to Recall@10 than a slightly broader pool the LLM can still sort
through. If filtering leaves fewer than 3 candidates, constraints are
progressively relaxed (test-type first, then seniority) rather than returning an
empty shortlist.

## Prompt design

All prompts share one grounding rule: the LLM selects only from candidate IDs
explicitly listed in the prompt, never generates a name or URL freely. This is
enforced twice — once implicitly via the prompt instructions, and once
explicitly in code (`app/llm/parser.py` drops any selected ID not present in the
offered candidates; `app/validation/catalog_validator.py` performs a second,
independent check against the full catalog before any response leaves the
service). Structured output uses Groq's `json_schema` mode with `strict: true`
(constrained decoding), which is a stronger guarantee than best-effort JSON mode
for the hard schema-compliance eval.

## Evaluation approach

Three separate evaluation scripts, run in priority order:

1. **Latency first** (`scripts/evaluation/latency.py`) — checked before Recall@10,
   since a system that times out scores zero on hard evals regardless of
   retrieval quality.
2. **Scripted behavior probes** (`scripts/evaluation/behavior_tests.py`) —
   refusal on off-topic/injection, no premature recommendation on vague queries,
   turn-cap forcing, URL shape validity.
3. **Replay harness** (`scripts/evaluation/replay_harness.py`) — runs the 10
   provided conversation traces using an LLM-simulated persona (mirroring how
   SHL's own harness works), computes Mean Recall@10.

*[Fill in after running: actual latency numbers per turn type, behavior probe
pass rate, Mean Recall@10 across the 10 public traces.]*

## What didn't work / was reconsidered

- **Always-on cross-encoder reranking** was the initial design; moved to
  conditional after estimating it would add a full model inference pass to
  every recommend turn on free-tier CPU hosting, threatening the 30s budget for
  no benefit on already-short candidate pools.
- **A separate LLM call for intent classification** was cut in favor of
  rule-based slot extraction with an LLM fallback that only triggers when the
  rule-based pass produces no new information for a full turn — halves LLM
  round-trips per conversation in the common case.
- **Query-time expansion** was replaced with catalog-side enrichment for the
  hallucination-safety reasons described above.

*[Fill in after running: any specific queries or traces where retrieval
under-performed, and what was changed in response.]*

## AI tool usage

This entire codebase was built with Claude (Anthropic) as an AI pair-programmer:
architecture discussion and iteration, code generation for every module, and
test-writing. Each module was tested (unit tests with mocked external
dependencies — the embedding/cross-encoder models and the Groq LLM — since
sandbox network access couldn't reach HuggingFace or a live API key) before
moving to the next, with several real bugs caught and fixed this way (e.g. an
early version of slot extraction was accidentally reading keywords out of the
assistant's own clarifying questions rather than only the user's answers).
