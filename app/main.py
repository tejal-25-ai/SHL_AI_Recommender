"""
Entry point for the SHL Assessment Recommender API.
Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.router import router
from app.api.chat import get_retrieval_service, get_llm_client
from app.core.logging import get_logger, log_duration

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting...")
    yield
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    with log_duration(logger, "startup_warmup"):
        retrieval = get_retrieval_service()
        get_llm_client()

        # Preload models so the first request doesn't pay the load cost
        retrieval.embedding_index._load_model()
        retrieval.reranker._load_model()

    logger.info("Startup warmup complete — service ready")
    yield
""" 
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up expensive singletons (embedding model + index load, LLM
    # client) during the doc's allowed startup window, rather than on
    # the first real /chat call — that first call must fit in the 30s
    # per-call budget just like every other turn.
    with log_duration(logger, "startup_warmup"):
        get_retrieval_service()
        get_llm_client()
    logger.info("Startup warmup complete — service ready")
    yield
 """

app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)
app.include_router(router)
