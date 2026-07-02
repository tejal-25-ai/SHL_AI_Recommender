"""
Parses the LLM's raw structured-output dict (see app/llm/client.py's
schema: reply/selected_ids/end_of_conversation) into a safe internal
result, filtering selected_ids down to ones that actually exist in the
candidates the LLM was shown. This is a first line of defense against
hallucination — app/validation/catalog_validator.py does the second,
authoritative check against the full catalog before the HTTP response
is ever built.
"""

from dataclasses import dataclass

from app.models.catalog import CatalogItem
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedLLMOutput:
    reply: str
    selected_items: list[CatalogItem]
    end_of_conversation: bool


def parse_llm_output(raw: dict, candidates: list[CatalogItem]) -> ParsedLLMOutput:
    """
    raw: the dict returned by LLMClient.generate_structured().
    candidates: the exact list of CatalogItems the LLM was shown for
    this call — selected_ids not present in this list are dropped
    silently (logged) rather than trusted, since the LLM should never
    be able to select something it wasn't offered.
    """
    candidates_by_id = {item.id: item for item in candidates}

    reply = raw.get("reply", "")
    selected_ids = raw.get("selected_ids", []) or []
    end_of_conversation = bool(raw.get("end_of_conversation", False))

    selected_items: list[CatalogItem] = []
    for cid in selected_ids:
        item = candidates_by_id.get(cid)
        if item is not None:
            if item.id not in {i.id for i in selected_items}:  # dedup
                selected_items.append(item)
        else:
            logger.warning(
                f"LLM selected id '{cid}' not present in offered candidates — dropping"
            )

    return ParsedLLMOutput(
        reply=reply, selected_items=selected_items, end_of_conversation=end_of_conversation
    )
