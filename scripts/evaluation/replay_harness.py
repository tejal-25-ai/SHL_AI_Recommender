"""
Replay harness — runs each provided conversation trace against a live
local /chat server, using the LLM (Groq) to simulate the persona's
answers from its fact set, exactly as the doc describes SHL's own
evaluator working. Computes Mean Recall@10 across all traces at the end.

IMPORTANT: this assumes a trace JSON shape of:
    {
      "persona": "...",
      "facts": {"role": "...", "seniority": "...", ...},
      "expected_relevant_urls": ["https://www.shl.com/...", ...]
    }
Adjust `load_trace` below once you've inspected the actual downloaded
trace files — the doc doesn't specify the exact field names, only the
content (persona + facts + labeled expected shortlist).

Usage:
    python scripts/evaluation/replay_harness.py --traces-dir data/traces --url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import httpx
from groq import Groq

from app.core.config import get_settings
from app.core.logging import get_logger
from scripts.evaluation.recall import mean_recall_at_k

logger = get_logger(__name__)

MAX_TURNS = 8

_SIMULATED_USER_SYSTEM_PROMPT = """You are roleplaying as a hiring manager with this persona and \
these facts:

Persona: {persona}
Facts: {facts}

You are talking to an assistant that will ask you questions to recommend SHL assessments. \
Answer truthfully and only from the facts given. If asked about something not in your facts, \
say you have no preference. Once the assistant gives you a shortlist of recommendations, \
respond with exactly "END_CONVERSATION" and nothing else. Keep answers short and natural, \
like a real hiring manager typing in a chat."""


def load_trace(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_user_turn(groq_client: Groq, model: str, trace: dict, history: list[dict]) -> str:
    system_prompt = _SIMULATED_USER_SYSTEM_PROMPT.format(
        persona=trace.get("persona", ""), facts=json.dumps(trace.get("facts", {}))
    )
    # flip roles for the simulated user: what was 'assistant' to our
    # service is 'user' input to this simulator, and vice versa
    flipped = [
        {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
        for m in history
    ]
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + flipped,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


def run_trace(client: httpx.Client, groq_client: Groq, model: str, base_url: str, trace: dict) -> dict:
    history: list[dict] = []
    final_recommendations: list[dict] = []

    # First user turn: use trace's own opening line if provided, else
    # synthesize one from facts.
    opening = trace.get("opening_message") or f"I'm hiring for: {trace.get('facts', {})}"
    history.append({"role": "user", "content": opening})

    for turn in range(MAX_TURNS):
        resp = client.post(f"{base_url}/chat", json={"messages": history})
        resp.raise_for_status()
        data = resp.json()

        history.append({"role": "assistant", "content": data.get("reply", "")})

        if data.get("recommendations"):
            final_recommendations = data["recommendations"]

        if data.get("end_of_conversation") or turn == MAX_TURNS - 1:
            break

        user_reply = simulate_user_turn(groq_client, model, trace, history)
        if "END_CONVERSATION" in user_reply:
            break
        history.append({"role": "user", "content": user_reply})

    return {
        "trace_name": trace.get("name", "unnamed"),
        "retrieved_urls": [r["url"] for r in final_recommendations],
        "relevant_urls": trace.get("expected_relevant_urls", []),
        "turns_used": len(history),
    }


def main(traces_dir: str, base_url: str) -> None:
    settings = get_settings()
    groq_client = Groq(api_key=settings.groq_api_key)

    trace_paths = sorted(Path(traces_dir).glob("*.json"))
    if not trace_paths:
        print(f"No trace files found in {traces_dir}")
        return

    results = []
    with httpx.Client(timeout=35) as client:
        for path in trace_paths:
            trace = load_trace(path)
            print(f"\nRunning trace: {path.name}")
            result = run_trace(client, groq_client, settings.llm_model, base_url, trace)
            print(f"  turns used: {result['turns_used']}, recommendations: {len(result['retrieved_urls'])}")
            results.append(result)

    per_query = [(r["retrieved_urls"], r["relevant_urls"]) for r in results if r["relevant_urls"]]
    if per_query:
        mean_recall = mean_recall_at_k(per_query, k=10)
        print(f"\n=== Mean Recall@10: {mean_recall:.3f} across {len(per_query)} traces ===")
    else:
        print("\nNo traces had expected_relevant_urls — cannot compute Recall@10.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="data/traces")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    main(args.traces_dir, args.url)
