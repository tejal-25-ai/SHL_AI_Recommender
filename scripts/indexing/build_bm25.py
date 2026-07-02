"""
BM25 is pure Python and cheap enough to rebuild fresh at process
startup (see app/api/chat.py's get_retrieval_service) — it does not
need to be persisted to disk like embeddings do. This script exists to
validate, offline, that the final catalog builds a working BM25 index
and to sanity-check a few queries before deploying.

Run after build_embeddings.py:
    python scripts/indexing/build_bm25.py
"""

import json
import sys

sys.path.insert(0, ".")

from app.models.catalog import CatalogItem
from app.retrieval.bm25 import BM25Index
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SMOKE_TEST_QUERIES = [
    "Java developer",
    "personality assessment for managers",
    "cognitive ability test",
    "sales situational judgement",
]


def main():
    settings = get_settings()

    with open(settings.catalog_path, "r", encoding="utf-8") as f:
        raw_catalog = json.load(f)
    catalog = [CatalogItem(**item) for item in raw_catalog]

    index = BM25Index(catalog)
    catalog_by_id = {item.id: item for item in catalog}

    logger.info("Running smoke-test queries against BM25 index:")
    for query in _SMOKE_TEST_QUERIES:
        results = index.search(query, top_k=3)
        logger.info(f"  '{query}':")
        for cid, score in results:
            name = catalog_by_id[cid].name if cid in catalog_by_id else "?"
            logger.info(f"    {name} (score={score:.3f})")

    logger.info("BM25 validation complete.")


if __name__ == "__main__":
    main()
