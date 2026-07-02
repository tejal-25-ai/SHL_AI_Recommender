"""
Response validator — the single call site app/llm/generator.py uses to
turn (reply text, selected CatalogItems, end_of_conversation) into a
guaranteed-valid ChatResponse. Combines catalog_validator (membership,
dedup, cap) and schema_validator (safe construction) in the right order.
"""

from app.models.catalog import CatalogItem
from app.models.response import ChatResponse, Recommendation
from app.validation.catalog_validator import validate_recommendations
from app.validation.schema_validator import build_validated_response


def finalize_response(
    reply: str,
    selected_items: list[CatalogItem],
    end_of_conversation: bool,
    full_catalog_by_id: dict[str, CatalogItem],
) -> ChatResponse:
    validated_items = validate_recommendations(selected_items, full_catalog_by_id)
    recommendations = [
        Recommendation(name=item.name, url=item.url, test_type=item.test_type)
        for item in validated_items
    ]
    return build_validated_response(reply, recommendations, end_of_conversation)
