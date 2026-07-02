"""
Measures /chat call latency against the doc's hard 30-second-per-call
budget. Run this BEFORE worrying about Recall@10 — an architecturally
perfect system that times out scores zero on hard evals regardless of
retrieval quality.

Usage:
    python scripts/evaluation/latency.py [--url http://localhost:8000]
"""

import argparse
import time
import httpx

TIMEOUT_BUDGET_SECONDS = 30

_TEST_CONVERSATIONS = [
    [{"role": "user", "content": "I need an assessment"}],  # clarify path
    [
        {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
        {"role": "assistant", "content": "Sure. What is seniority level?"},
        {"role": "user", "content": "Mid-level, around 4 years"},
    ],  # recommend path -- the most latency-sensitive one
    [{"role": "user", "content": "What is the difference between OPQ32r and GSA?"}],  # compare path
]


def measure_latency(base_url: str) -> None:
    with httpx.Client(timeout=TIMEOUT_BUDGET_SECONDS + 5) as client:
        print("Checking /health...")
        health_start = time.perf_counter()
        health_resp = client.get(f"{base_url}/health")
        print(f"  /health: {health_resp.status_code} in {time.perf_counter() - health_start:.2f}s")

        for i, messages in enumerate(_TEST_CONVERSATIONS):
            start = time.perf_counter()
            resp = client.post(f"{base_url}/chat", json={"messages": messages})
            elapsed = time.perf_counter() - start

            status = "OK" if elapsed <= TIMEOUT_BUDGET_SECONDS else "OVER BUDGET"
            print(f"\nConversation {i + 1} ({len(messages)} turns): {elapsed:.2f}s [{status}]")
            if resp.status_code == 200:
                data = resp.json()
                print(f"  reply: {data.get('reply', '')[:100]}")
                print(f"  recommendations: {len(data.get('recommendations', []))} items")
            else:
                print(f"  ERROR: status {resp.status_code} — {resp.text[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    measure_latency(args.url)
