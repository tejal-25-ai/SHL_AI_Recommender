"""
Builds precomputed catalog embeddings from data/processed/catalog_enriched.json
and saves them to disk for fast runtime loading (see app/retrieval/embeddings.py's
EmbeddingIndex.load, used by app/api/chat.py at startup).

Run after enrichment:
    python scripts/indexing/build_embeddings.py
"""

import json
import sys
import os

sys.path.insert(0, ".")

from app.models.catalog import CatalogItem
from app.retrieval.embeddings import EmbeddingIndex
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def main():
    settings = get_settings()

    with open("data/processed/catalog_enriched.json", "r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    catalog = [CatalogItem(**item) for item in raw_catalog]
    logger.info(f"Building embeddings for {len(catalog)} catalog items using {settings.embedding_model}...")

    index = EmbeddingIndex(model_name=settings.embedding_model)
    index.build_from_catalog(catalog)

    os.makedirs(os.path.dirname(settings.embeddings_path), exist_ok=True)
    index.save(settings.embeddings_path, settings.embeddings_ids_path)

    # Also copy the final catalog into the exact path app/api/chat.py
    # expects at runtime, so this script is the single source of truth
    # for "what does the running service load".
    os.makedirs(os.path.dirname(settings.catalog_path), exist_ok=True)
    with open(settings.catalog_path, "w", encoding="utf-8") as f:
        json.dump(raw_catalog, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Embeddings saved to {settings.embeddings_path}")
    logger.info(f"Runtime catalog saved to {settings.catalog_path}")


if __name__ == "__main__":
    main()
