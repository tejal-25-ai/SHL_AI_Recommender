"""
POST /chat — the core conversational endpoint. Stateless: rebuilds
everything from the full message history on every call, per the
assignment's explicit no-server-side-state requirement.

Retrieval indexes and the LLM client are expensive to build (loading
the embedding model, loading precomputed catalog embeddings) so they
are built ONCE as process-level singletons (see get_retrieval_service/
get_llm_client), warmed up at app startup (see app/main.py) rather than
paid for on the first real request.
"""

import json
from functools import lru_cache

from fastapi import APIRouter

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.models.catalog import CatalogItem
from app.decision.controller import decide
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retrieval_service import RetrievalService
from app.llm.client import LLMClient
from app.llm.generator import generate_response
from app.validation.schema_validator import build_validated_response
from app.core.config import get_settings
from app.core.logging import get_logger, log_duration

logger = get_logger(__name__)
router = APIRouter()


def _load_catalog(path: str) -> list[CatalogItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    return [CatalogItem(**item) for item in raw_items]


@lru_cache
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()
    catalog = _load_catalog(settings.catalog_path)

    # BM25 is pure Python, cheap to rebuild fresh at process start —
    # no need to persist/load it from disk like embeddings.
    bm25_index = BM25Index(catalog)

    embedding_index = EmbeddingIndex(model_name=settings.embedding_model)
    embedding_index.load(settings.embeddings_path, settings.embeddings_ids_path)

    reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model)

    return RetrievalService(catalog, bm25_index, embedding_index, reranker)


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        with log_duration(logger, "full_chat_request"):
            decision = decide(request.messages)
            return generate_response(
                request.messages, decision, get_retrieval_service(), get_llm_client()
            )
    except Exception as e:  # noqa: BLE001 — must never let a request 500; schema compliance is a hard eval
        logger.error(f"Unhandled error in /chat: {e}", exc_info=True)
        return build_validated_response(
            reply="I ran into an issue processing that. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )
