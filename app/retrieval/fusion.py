"""
Reciprocal Rank Fusion (RRF) — merges the BM25 (lexical) and embedding
(semantic) ranked lists into a single combined ranking.

RRF is used instead of raw score averaging because BM25 scores and
cosine similarities live on different, incomparable scales — RRF sums
1/(k + rank) per list, so it only ever needs each list's *ordering*,
never their raw magnitudes.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)

# Standard RRF constant. Larger k flattens the influence of rank
# position (top-1 vs top-5 matters less); smaller k sharpens it. 60 is
# the commonly used default in IR literature and a reasonable choice
# without needing to tune it against held-out data we don't have yet.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """
    ranked_lists: one or more ranked [(id, original_score), ...] lists,
    each already sorted best-first (original_score is ignored — only
    rank position within each list matters for RRF).

    Returns a single fused ranking: [(id, rrf_score), ...] sorted
    descending by rrf_score. An id that appears in multiple input lists
    accumulates a contribution from each; an id in only one list still
    appears, just with a smaller total score.
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused


def fuse_bm25_and_embeddings(
    bm25_results: list[tuple[str, float]],
    embedding_results: list[tuple[str, float]],
    top_k: int | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """
    Convenience wrapper for the specific two-source fusion this project
    uses. Either input list may be empty (e.g. a query with zero
    keyword overlap yields an empty BM25 list) — RRF handles that
    gracefully since it only sums over lists actually containing the id.
    """
    fused = reciprocal_rank_fusion([bm25_results, embedding_results], k=k)
    logger.info(
        f"RRF fused {len(bm25_results)} BM25 + {len(embedding_results)} embedding "
        f"results into {len(fused)} unique candidates"
    )
    return fused[:top_k] if top_k is not None else fused
