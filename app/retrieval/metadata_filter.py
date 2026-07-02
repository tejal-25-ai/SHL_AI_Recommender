"""
Metadata filtering — narrows the RRF-fused candidate pool using
explicit constraints extracted into Slots (test_type_pref, seniority).

Deliberately conservative: every filter has a "would this wipe out too
many candidates?" safety check before being applied. An over-narrow
filter that leaves too few candidates is a bigger threat to Recall@10
than a slightly-too-broad candidate set that the cross-encoder/LLM can
still sort through — so a filter that would gut the pool is skipped
entirely rather than applied and silently starving recommend of options.
"""

from app.models.catalog import CatalogItem
from app.models.conversation import Slots
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_CANDIDATES = 5

# Maps our extracted seniority buckets (see app/decision/slot_extractor.py)
# to substrings likely to appear in SHL's free-text job_levels field.
# Matching is intentionally soft (substring, case-insensitive) since
# catalog job_levels wording won't exactly match our bucket labels.
_SENIORITY_SYNONYMS: dict[str, list[str]] = {
    "entry": ["entry", "graduate", "trainee", "junior"],
    "junior": ["junior", "entry"],
    "mid": ["mid", "professional", "intermediate"],
    "senior": ["senior", "lead", "expert", "principal"],
    "manager": ["manager", "supervisor", "management"],
    "director": ["director", "executive", "vp", "vice president", "chief", "senior leader"],
}


def _seniority_matches(item: CatalogItem, seniority_bucket: str) -> bool:
    """
    True if the item's job_levels are compatible with the requested
    seniority bucket, OR if the item has no job_levels data at all
    (unknown is treated as compatible, not excluded — we should never
    drop a candidate just because scraping didn't capture this field).
    """
    if not item.job_levels:
        return True

    synonyms = _SENIORITY_SYNONYMS.get(seniority_bucket, [seniority_bucket])
    job_levels_text = " ".join(item.job_levels).lower()
    return any(syn in job_levels_text for syn in synonyms)


def filter_by_metadata(
    candidates: list[tuple[str, float]],
    catalog_by_id: dict[str, CatalogItem],
    slots: Slots,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> list[tuple[str, float]]:
    """
    candidates: fused [(catalog_id, score), ...] from RRF, best-first.
    catalog_by_id: full catalog keyed by id, for metadata lookup.
    slots: accumulated conversation constraints.

    Returns a filtered (still best-first) candidate list. Falls back to
    the pre-filter set for any single filter that would leave fewer
    than min_candidates results, rather than applying it anyway.
    """
    filtered = candidates

    # --- Test type filter (hard signal — test_type is a clean 1:1 field) ---
    if slots.test_type_pref.value:
        narrowed = [
            (cid, score)
            for cid, score in filtered
            if catalog_by_id[cid].test_type == slots.test_type_pref.value
        ]
        if len(narrowed) >= min_candidates:
            logger.info(
                f"test_type_pref={slots.test_type_pref.value} filter: "
                f"{len(filtered)} -> {len(narrowed)} candidates"
            )
            filtered = narrowed
        else:
            logger.info(
                f"test_type_pref={slots.test_type_pref.value} filter skipped "
                f"(would leave only {len(narrowed)} < {min_candidates} candidates)"
            )

    # --- Seniority filter (soft signal — free-text substring match) ---
    if slots.seniority.value:
        narrowed = [
            (cid, score)
            for cid, score in filtered
            if _seniority_matches(catalog_by_id[cid], slots.seniority.value)
        ]
        if len(narrowed) >= min_candidates:
            logger.info(
                f"seniority={slots.seniority.value} filter: "
                f"{len(filtered)} -> {len(narrowed)} candidates"
            )
            filtered = narrowed
        else:
            logger.info(
                f"seniority={slots.seniority.value} filter skipped "
                f"(would leave only {len(narrowed)} < {min_candidates} candidates)"
            )

    return filtered
