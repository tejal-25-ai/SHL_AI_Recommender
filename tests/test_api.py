"""
API-level tests. Only /health is tested here directly, since /chat's
process-level singletons (RetrievalService, LLMClient) need a real
catalog file + downloaded embedding model + a real Groq key to build —
none of which exist in a clean checkout or CI without setup.

Once you have data/processed/catalog.json + data/embeddings/* built
(see scripts/indexing/build_indexes.py) and a real GROQ_API_KEY, use
scripts/evaluation/behavior_tests.py and replay_harness.py against a
running `uvicorn app.main:app` for full /chat integration testing —
those exercise the real endpoint end-to-end rather than mocking it.
"""

import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import health


def _health_only_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    return app


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        client = TestClient(_health_only_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
