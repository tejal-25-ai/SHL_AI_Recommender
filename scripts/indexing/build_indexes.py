"""
Runs the full offline indexing pipeline in order:
enrich_catalog -> generate_tags -> build_embeddings -> build_bm25 (validation).

Run once after scraping, before deploying:
    python scripts/indexing/build_indexes.py
"""

import subprocess
import sys


_STEPS = [
    ("Cleaning raw catalog", ["python", "scripts/enrichment/enrich_catalog.py"]),
    ("Generating enrichment tags", ["python", "scripts/enrichment/generate_tags.py"]),
    ("Building embeddings", ["python", "scripts/indexing/build_embeddings.py"]),
    ("Validating BM25", ["python", "scripts/indexing/build_bm25.py"]),
]


def main():
    for label, cmd in _STEPS:
        print(f"\n=== {label} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED at step: {label}")
            sys.exit(1)
    print("\nAll indexing steps completed successfully.")


if __name__ == "__main__":
    main()
