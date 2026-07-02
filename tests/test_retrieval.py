import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

from app.models.catalog import CatalogItem
from app.models.conversation import Slots, SlotValue
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import fuse_bm25_and_embeddings, reciprocal_rank_fusion
from app.retrieval.metadata_filter import filter_by_metadata
from app.retrieval.reranker import CrossEncoderReranker, maybe_rerank


def _mock_catalog():
    return [
        CatalogItem(id="java-8", name="Java 8", url="https://x.com/1", test_type="K",
                    description="Java developer knowledge test", job_levels=["Professional"]),
        CatalogItem(id="opq32r", name="OPQ32r", url="https://x.com/2", test_type="P",
                    description="Personality and leadership questionnaire", job_levels=["Manager", "Senior Leader"]),
        CatalogItem(id="gsa", name="GSA", url="https://x.com/3", test_type="A",
                    description="Cognitive reasoning aptitude assessment", job_levels=["Entry-Level"]),
    ]


class TestBM25:
    def test_ranks_keyword_match_first(self):
        index = BM25Index(_mock_catalog())
        results = index.search("Java developer", top_k=3)
        assert results[0][0] == "java-8"

    def test_no_overlap_returns_empty(self):
        index = BM25Index(_mock_catalog())
        assert index.search("underwater basket weaving", top_k=3) == []


class TestFusion:
    def test_item_in_both_lists_ranks_highest(self):
        bm25 = [("java-8", 5.0), ("gsa", 1.0)]
        emb = [("java-8", 0.9), ("opq32r", 0.7)]
        fused = fuse_bm25_and_embeddings(bm25, emb)
        assert fused[0][0] == "java-8"

    def test_empty_lists_no_crash(self):
        assert fuse_bm25_and_embeddings([], []) == []

    def test_rrf_score_matches_manual_calculation(self):
        result = reciprocal_rank_fusion([[("only-item", 99.0)]], k=60)
        assert abs(result[0][1] - (1.0 / 61)) < 1e-9


class TestMetadataFilter:
    def test_test_type_hard_filter(self):
        catalog = _mock_catalog()
        catalog_by_id = {i.id: i for i in catalog}
        candidates = [(i.id, 1.0) for i in catalog]
        slots = Slots(test_type_pref=SlotValue(value="K"))
        result = filter_by_metadata(candidates, catalog_by_id, slots, min_candidates=1)
        assert {cid for cid, _ in result} == {"java-8"}

    def test_safety_net_skips_over_narrow_filter(self):
        catalog = _mock_catalog()
        catalog_by_id = {i.id: i for i in catalog}
        candidates = [(i.id, 1.0) for i in catalog]
        slots = Slots(test_type_pref=SlotValue(value="B"))  # no item has type B
        result = filter_by_metadata(candidates, catalog_by_id, slots, min_candidates=3)
        assert len(result) == 3  # filter skipped, full set returned

    def test_no_filter_when_slots_empty(self):
        catalog = _mock_catalog()
        catalog_by_id = {i.id: i for i in catalog}
        candidates = [(i.id, 1.0) for i in catalog]
        result = filter_by_metadata(candidates, catalog_by_id, Slots())
        assert len(result) == 3


class TestReranker:
    def test_skips_below_threshold(self):
        class MockReranker(CrossEncoderReranker):
            def _score_pairs(self, pairs):
                raise AssertionError("Should not be called below threshold")

        candidates = [(f"item-{i}", 1.0 - i * 0.1) for i in range(5)]
        catalog_by_id = {f"item-{i}": CatalogItem(id=f"item-{i}", name="x", url="https://x.com", test_type="K")
                          for i in range(5)}
        result = maybe_rerank("query", candidates, catalog_by_id, MockReranker(), top_k=3, threshold=10)
        assert result == candidates[:3]

    def test_reranks_above_threshold(self):
        class MockReranker(CrossEncoderReranker):
            def _score_pairs(self, pairs):
                return [float(len(set(q.split()) & set(d.split()))) for q, d in pairs]

        catalog_by_id = {
            f"item-{i}": CatalogItem(
                id=f"item-{i}", name="x", url="https://x.com", test_type="K",
                description="java backend" if i == 15 else "unrelated content",
            )
            for i in range(20)
        }
        candidates = [(f"item-{i}", 1.0) for i in range(20)]
        result = maybe_rerank("java backend", candidates, catalog_by_id, MockReranker(), top_k=3, threshold=10)
        assert result[0][0] == "item-15"
