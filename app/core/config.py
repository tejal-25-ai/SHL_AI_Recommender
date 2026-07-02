"""
Centralized configuration, loaded once from environment variables.
Every other module reads thresholds/model names from here — nothing
should hardcode these values elsewhere.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        # --- LLM provider ---
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

        # --- Retrieval models ---
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.cross_encoder_model: str = os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # --- Retrieval tuning ---
        self.retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "30"))
        self.cross_encoder_threshold: int = int(os.getenv("CROSS_ENCODER_THRESHOLD", "10"))
        self.final_top_k: int = int(os.getenv("FINAL_TOP_K", "10"))

        # --- Conversation limits (must match assignment doc exactly) ---
        self.max_turns: int = int(os.getenv("MAX_TURNS", "8"))
        self.force_recommend_after_turn: int = int(
            os.getenv("FORCE_RECOMMEND_AFTER_TURN", "6")
        )

        # --- Paths ---
        self.catalog_path: str = os.getenv("CATALOG_PATH", "data/processed/catalog.json")
        self.embeddings_path: str = os.getenv(
            "EMBEDDINGS_PATH", "data/embeddings/catalog_embeddings.npy"
        )
        self.embeddings_ids_path: str = os.getenv(
            "EMBEDDINGS_IDS_PATH", "data/embeddings/catalog_ids.json"
        )
        self.bm25_index_path: str = os.getenv("BM25_INDEX_PATH", "data/indexes/bm25_index.pkl")

    def validate(self) -> None:
        """Fail loudly at startup rather than deep inside a request."""
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )


@lru_cache
def get_settings() -> Settings:
    """
    Cached singleton accessor. Use `from app.core.config import get_settings`
    everywhere instead of importing Settings directly, so config is loaded
    exactly once per process.
    """
    settings = Settings()
    settings.validate()
    return settings
