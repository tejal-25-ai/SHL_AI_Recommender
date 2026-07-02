"""
BM25 lexical search over the catalog.

Catches exact keyword matches (product codes, tech names) that
embedding similarity alone can miss — e.g. a query mentioning "Java"
should strongly favor "Java 8 (New)" via keyword overlap, independent
of semantic similarity.
"""

import re
from rank_bm25 import BM25Okapi

from app.models.catalog import CatalogItem
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alnum tokenizer — consistent between index build
    and query time, which is all BM25 needs (no stemming required for
    a catalog this size)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, catalog: list[CatalogItem]):
        if not catalog:
            raise ValueError("Cannot build BM25Index from an empty catalog")

        self.catalog_ids: list[str] = [item.id for item in catalog]
        corpus = [tokenize(item.embedding_text()) for item in catalog]
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25Index built over {len(catalog)} catalog items")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Returns [(catalog_id, score), ...] sorted descending by score,
        length <= top_k. Items with score 0 (no keyword overlap at all)
        are excluded — a zero BM25 score is not a meaningful ranking
        signal and would just add noise ahead of RRF fusion.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.catalog_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(cid, score) for cid, score in ranked if score > 0][:top_k]
