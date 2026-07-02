"""
Catalog validator — the authoritative, final check that every
recommended item truly exists in the full scraped catalog. This is a
second, independent check beyond app/llm/parser.py's per-call
candidate check: paranoid by design, since a hallucinated or malformed
URL in the response is a hard-eval failure.
"""

from app.models.catalog import CatalogItem
from app.core.constants import MAX_RECOMMENDATIONS
from app.core.logging import get_logger

logger = get_logger(__name__)


def validate_recommendations(
    items: list[CatalogItem], full_catalog_by_id: dict[str, CatalogItem]
) -> list[CatalogItem]:
    """
    Filters `items` down to only those whose id exists in the full
    catalog, dedups by id (preserving first-seen order), and caps at
    MAX_RECOMMENDATIONS. Anything dropped is logged — dropping should
    be rare in practice since items originate from retrieval against
    this same catalog, but this check must never be skipped.
    """
    seen_ids: set[str] = set()
    validated: list[CatalogItem] = []

    for item in items:
        if item.id not in full_catalog_by_id:
            logger.warning(f"Dropping recommendation '{item.id}' — not found in full catalog")
            continue
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        validated.append(item)

    if len(validated) > MAX_RECOMMENDATIONS:
        logger.info(f"Truncating {len(validated)} validated recommendations to {MAX_RECOMMENDATIONS}")
        validated = validated[:MAX_RECOMMENDATIONS]

    return validated
