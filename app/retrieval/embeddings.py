"""
Semantic embedding search over the catalog using sentence-transformers.

Two responsibilities:
1. OFFLINE: encode the full catalog once (scripts/indexing/build_embeddings.py
   calls build_from_catalog + save). Never re-embed the catalog at
   request time — that would burn the 30s-per-call budget for nothing.
2. RUNTIME: load precomputed catalog embeddings, encode each incoming
   query, and rank by cosine similarity.

Model loading is isolated in _load_model()/_embed_texts() so it can be
swapped for a lightweight/mock embedder in tests without requiring a
real network download of model weights.
"""

import json
import numpy as np

from app.models.catalog import CatalogItem
from app.core.logging import get_logger

logger = get_logger(__name__)


def cosine_similarity_matrix(query_vec: np.ndarray, catalog_matrix: np.ndarray) -> np.ndarray:
    """
    query_vec: shape (d,)
    catalog_matrix: shape (n, d)
    Returns shape (n,) of cosine similarities. Normalizes both inputs
    internally rather than assuming pre-normalized vectors — defensive,
    avoids a subtle bug class where a caller passes raw model output.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    catalog_norms = catalog_matrix / (
        np.linalg.norm(catalog_matrix, axis=1, keepdims=True) + 1e-10
    )
    return catalog_norms @ query_norm


class EmbeddingIndex:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None  # lazy-loaded
        self.catalog_ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        """Lazy import + load — keeps sentence-transformers (and its
        torch dependency) out of modules that don't need it, and lets
        tests substitute a mock encoder by overriding _embed_texts
        without ever importing sentence-transformers."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Isolated so tests/mocks can override this single method
        instead of the real model."""
        model = self._load_model()
        return np.array(model.encode(texts, normalize_embeddings=True))

    def build_from_catalog(self, catalog: list[CatalogItem]) -> None:
        if not catalog:
            raise ValueError("Cannot build EmbeddingIndex from an empty catalog")
        self.catalog_ids = [item.id for item in catalog]
        texts = [item.embedding_text() for item in catalog]
        logger.info(f"Encoding {len(texts)} catalog items...")
        self.embeddings = self._embed_texts(texts)

    def save(self, embeddings_path: str, ids_path: str) -> None:
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save — call build_from_catalog first")
        np.save(embeddings_path, self.embeddings)
        with open(ids_path, "w") as f:
            json.dump(self.catalog_ids, f)
        logger.info(f"Saved embeddings to {embeddings_path}, ids to {ids_path}")

    def load(self, embeddings_path: str, ids_path: str) -> None:
        self.embeddings = np.load(embeddings_path)
        with open(ids_path) as f:
            self.catalog_ids = json.load(f)
        if len(self.catalog_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"catalog_ids length ({len(self.catalog_ids)}) does not match "
                f"embeddings row count ({self.embeddings.shape[0]}) — index files "
                "are out of sync, rebuild via scripts/indexing/build_embeddings.py"
            )
        logger.info(f"Loaded {len(self.catalog_ids)} precomputed catalog embeddings")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.embeddings is None or not self.catalog_ids:
            raise RuntimeError(
                "EmbeddingIndex has no data — call build_from_catalog or load first"
            )

        query_vec = self._embed_texts([query])[0]
        scores = cosine_similarity_matrix(query_vec, self.embeddings)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(self.catalog_ids[i], float(scores[i])) for i in ranked_idx]
