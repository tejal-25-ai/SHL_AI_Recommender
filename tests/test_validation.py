import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

import pytest
from pydantic import ValidationError

from app.models.response import ChatResponse, Recommendation
from app.models.catalog import CatalogItem
from app.validation.catalog_validator import validate_recommendations
from app.validation.schema_validator import build_validated_response
from app.validation.response_validator import finalize_response


class TestResponseModel:
    def test_valid_response(self):
        r = ChatResponse(
            reply="Here are 5 assessments.",
            recommendations=[Recommendation(name="Java 8", url="https://x.com", test_type="K")],
            end_of_conversation=True,
        )
        assert len(r.recommendations) == 1

    def test_rejects_bad_test_type(self):
        with pytest.raises(ValidationError):
            Recommendation(name="Bad", url="https://x.com", test_type="Z")

    def test_rejects_over_10_recommendations(self):
        with pytest.raises(ValidationError):
            ChatResponse(
                reply="x",
                recommendations=[
                    Recommendation(name=f"T{i}", url=f"https://x.com/{i}", test_type="K") for i in range(11)
                ],
            )

    def test_empty_recommendations_valid(self):
        r = ChatResponse(reply="Can you tell me more?", recommendations=[], end_of_conversation=False)
        assert r.recommendations == []


class TestCatalogValidator:
    def test_drops_items_not_in_full_catalog(self):
        full_catalog = {"a": CatalogItem(id="a", name="A", url="https://x.com/a", test_type="K")}
        fake_item = CatalogItem(id="b", name="B", url="https://x.com/b", test_type="K")
        result = validate_recommendations([fake_item], full_catalog)
        assert result == []

    def test_dedups_by_id(self):
        item = CatalogItem(id="a", name="A", url="https://x.com/a", test_type="K")
        full_catalog = {"a": item}
        result = validate_recommendations([item, item], full_catalog)
        assert len(result) == 1

    def test_caps_at_max_recommendations(self):
        items = [CatalogItem(id=str(i), name=f"item-{i}", url=f"https://x.com/{i}", test_type="K") for i in range(15)]
        full_catalog = {i.id: i for i in items}
        result = validate_recommendations(items, full_catalog)
        assert len(result) == 10


class TestSchemaValidator:
    def test_builds_valid_response(self):
        r = build_validated_response("Here you go", [Recommendation(name="A", url="https://x.com", test_type="K")], True)
        assert isinstance(r, ChatResponse)

    def test_truncates_over_10(self):
        recs = [Recommendation(name=f"T{i}", url=f"https://x.com/{i}", test_type="K") for i in range(15)]
        r = build_validated_response("x", recs, True)
        assert len(r.recommendations) == 10


class TestResponseValidatorIntegration:
    def test_finalize_response_end_to_end(self):
        item = CatalogItem(id="a", name="A", url="https://x.com/a", test_type="K")
        full_catalog = {"a": item}
        r = finalize_response("Here's my pick", [item], True, full_catalog)
        assert isinstance(r, ChatResponse)
        assert len(r.recommendations) == 1
        assert r.recommendations[0].name == "A"
