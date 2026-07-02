"""
Retrieval service — the single entry point for turning a query + slots
into a ranked list of CatalogItems. Orchestrates every retrieval-layer
module built so far: BM25, embeddings, RRF fusion, metadata filtering,
and conditional cross-encoder reranking.

Also implements the empty-result fallback from the locked architecture:
if metadata filtering leaves too few candidates, progressively relax
constraints (drop test_type_pref first, then seniority) rather than
returning an empty shortlist.
"""

from app.models.catalog import CatalogItem
from app.models.conversation import Slots, SlotValue
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.fusion import fuse_bm25_and_embeddings
from app.retrieval.metadata_filter import filter_by_metadata
from app.retrieval.reranker import CrossEncoderReranker, maybe_rerank
from app.core.logging import get_logger, log_duration

logger = get_logger(__name__)

MIN_VIABLE_CANDIDATES = 3


class RetrievalService:
    def __init__(
        self,
        catalog: list[CatalogItem],
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
        reranker: CrossEncoderReranker,
    ):
        self.catalog_by_id: dict[str, CatalogItem] = {item.id: item for item in catalog}
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        slots: Slots,
        top_k: int = 10,
        retrieval_top_k: int = 30,
    ) -> list[CatalogItem]:
        """
        Returns a ranked list of CatalogItems, length <= top_k. Never
        raises on a poor query — worst case returns an empty list,
        which the caller (app/llm/generator.py) should treat as "not
        enough to recommend yet, clarify instead" rather than a crash.
        """
        with log_duration(logger, "bm25_search"):
            bm25_results = self.bm25_index.search(query, retrieval_top_k)
        with log_duration(logger, "embedding_search"):
            embedding_results = self.embedding_index.search(query, retrieval_top_k)

        fused = fuse_bm25_and_embeddings(bm25_results, embedding_results, top_k=retrieval_top_k)
        if not fused:
            logger.info("No candidates from BM25 or embeddings — returning empty result")
            return []

        filtered = filter_by_metadata(fused, self.catalog_by_id, slots)

        # --- empty-result fallback: progressively relax constraints ---
        if len(filtered) < MIN_VIABLE_CANDIDATES:
            logger.info(
                f"Only {len(filtered)} candidates after filtering — relaxing "
                "test_type_pref and retrying"
            )
            relaxed = slots.model_copy(update={"test_type_pref": SlotValue()})
            filtered = filter_by_metadata(fused, self.catalog_by_id, relaxed)

        if len(filtered) < MIN_VIABLE_CANDIDATES:
            logger.info(
                f"Still only {len(filtered)} candidates — relaxing seniority too"
            )
            relaxed = slots.model_copy(
                update={"test_type_pref": SlotValue(), "seniority": SlotValue()}
            )
            filtered = filter_by_metadata(fused, self.catalog_by_id, relaxed)

        if len(filtered) < MIN_VIABLE_CANDIDATES:
            logger.info("Giving up on filtering entirely — using raw fused candidates")
            filtered = fused

        with log_duration(logger, "conditional_rerank"):
            final_ranked = maybe_rerank(
                query, filtered, self.catalog_by_id, self.reranker, top_k=top_k
            )

        return [self.catalog_by_id[cid] for cid, _score in final_ranked if cid in self.catalog_by_id]

    def get_by_names_or_ids(self, names_or_codes: list[str]) -> list[CatalogItem]:
        """
        Used by the COMPARE intent to look up specific named assessments
        (e.g. "OPQ32r", "GSA") directly rather than via ranked retrieval.
        Case-insensitive substring match against name or id.
        """
        results: list[CatalogItem] = []
        seen_ids: set[str] = set()
        for query_name in names_or_codes:
            q_lower = query_name.lower()
            for item in self.catalog_by_id.values():
                if item.id in seen_ids:
                    continue
                if q_lower in item.name.lower() or q_lower in item.id.lower():
                    results.append(item)
                    seen_ids.add(item.id)
        return results
