"""
Recall@K computation per the assignment's exact definition:
Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments for the query)
Mean Recall@K = (1/N) * sum over queries of Recall@K_i
"""


def recall_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    """
    retrieved_urls: the shortlist URLs the agent actually returned, in
    the order returned (only the first k are considered).
    relevant_urls: the trace's labeled expected-relevant URLs.

    Returns 0.0 if relevant_urls is empty (undefined recall — avoid
    dividing by zero; callers should exclude such traces from the mean
    rather than let them silently count as a perfect or zero score).
    """
    if not relevant_urls:
        raise ValueError("relevant_urls must be non-empty to compute recall")

    top_k = set(retrieved_urls[:k])
    relevant_set = set(relevant_urls)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_recall_at_k(per_query_results: list[tuple[list[str], list[str]]], k: int = 10) -> float:
    """
    per_query_results: list of (retrieved_urls, relevant_urls) pairs,
    one per conversation trace.
    """
    if not per_query_results:
        return 0.0
    scores = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in per_query_results]
    return sum(scores) / len(scores)
