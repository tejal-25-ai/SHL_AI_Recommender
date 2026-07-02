"""
Internal representation of a single SHL assessment, as stored in
data/processed/catalog.json after scraping + enrichment.

This is NOT the API response model (see response.py's Recommendation,
which is the trimmed public-facing shape). This model carries the
extra fields retrieval/ranking/validation need internally.
"""

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    id: str  # stable slug/id, derived from URL — used as the join key
    #          across catalog.json, embeddings.npy, and bm25 index
    name: str
    url: str
    test_type: str  # single-letter code, e.g. "K", "P"
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    duration_minutes: int | None = None
    remote_testing: bool = False
    adaptive_irt: bool = False

    # --- enrichment fields (added offline by scripts/enrichment/) ---
    tags: list[str] = Field(default_factory=list)  # e.g. ["Backend", "API", "REST"]
    skills: list[str] = Field(default_factory=list)
    seniority: list[str] = Field(default_factory=list)

    def embedding_text(self) -> str:
        """
        Canonical text used to compute this item's embedding.
        Centralized here so indexing and any re-embedding step
        always produce it identically.
        """
        parts = [self.name, self.description]
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.skills:
            parts.append(" ".join(self.skills))
        return " | ".join(p for p in parts if p)
