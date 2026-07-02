"""
Conditional cross-encoder reranking.

Per the locked architecture: reranking only runs when the candidate
pool remaining AFTER metadata filtering exceeds a threshold (default
10). Below that, BM25+embedding RRF ordering is already close enough
to final, and running a cross-encoder would spend latency reshuffling
a list that's already short — a bad trade against the 30s call budget.

When it does run, input is hard-capped (default 25) regardless of how
many candidates survived filtering, so a rare wide-match query can't
spike latency unpredictably.
"""

from app.models.catalog import CatalogItem
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RERANK_THRESHOLD = 10
DEFAULT_MAX_RERANK_INPUT = 25


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load_model(self):
        """Lazy import + load — mirrors app/retrieval/embeddings.py's
        pattern so tests can override _score_pairs with a deterministic
        mock instead of downloading real model weights."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Isolated so tests/mocks can override this single method."""
        model = self._load_model()
        return [float(s) for s in model.predict(pairs)]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        catalog_by_id: dict[str, CatalogItem],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        candidates: [(catalog_id, score), ...], any order.
        Returns [(catalog_id, cross_encoder_score), ...] sorted
        descending, length <= top_k. Cross-encoder scores are on a
        different scale than RRF scores — callers should treat this
        output as the final ranking, not something to re-fuse further.
        """
        if not candidates:
            return []

        pairs = [(query, catalog_by_id[cid].embedding_text()) for cid, _ in candidates]
        scores = self._score_pairs(pairs)

        reranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return reranked[:top_k]


def maybe_rerank(
    query: str,
    candidates: list[tuple[str, float]],
    catalog_by_id: dict[str, CatalogItem],
    reranker: CrossEncoderReranker,
    top_k: int,
    threshold: int = DEFAULT_RERANK_THRESHOLD,
    max_rerank_input: int = DEFAULT_MAX_RERANK_INPUT,
) -> list[tuple[str, float]]:
    """
    The conditional gate: skip the cross-encoder entirely when there's
    nothing meaningful left to reorder (<= threshold candidates after
    metadata filtering); otherwise rerank a capped-size input.
    """
    if len(candidates) <= threshold:
        logger.info(
            f"Skipping cross-encoder: {len(candidates)} candidates <= threshold {threshold}"
        )
        return candidates[:top_k]

    capped = candidates[:max_rerank_input]
    logger.info(
        f"Running cross-encoder on {len(capped)} candidates "
        f"(capped from {len(candidates)}, threshold {threshold})"
    )
    return reranker.rerank(query, capped, catalog_by_id, top_k)
