"""
Generates enrichment tags for each catalog item — the offline,
catalog-side alternative to risky query-time expansion (see
app/models/catalog.py's CatalogItem.tags field).

For each item, asks the LLM to list concrete skills/technologies/
domains genuinely implied by the item's own name + description. This
is safe because it describes what the assessment already IS — not an
assumption about any particular user's query.

Run once, offline: python scripts/enrichment/generate_tags.py
"""

import json
import sys
import time

sys.path.insert(0, ".")

from groq import Groq
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TAG_PROMPT = """Given this SHL assessment, list 3-8 concrete skills, technologies, job \
functions, or domains this assessment is relevant to. Base this ONLY on the name and \
description provided — do not guess beyond what's reasonably implied.

Name: {name}
Description: {description}

Respond with ONLY a JSON array of short strings, e.g. ["Java", "Backend Development", "API Design"]"""


def generate_tags_for_item(client: Groq, model: str, name: str, description: str) -> list[str]:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": _TAG_PROMPT.format(name=name, description=description)}
            ],
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        # Model may wrap in markdown fences despite instructions -- strip defensively.
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        tags = json.loads(content)
        if isinstance(tags, list):
            return [str(t) for t in tags][:8]
        return []
    except Exception as e:
        logger.warning(f"Tag generation failed for '{name}': {e}")
        return []


def main():
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    with open("data/processed/catalog_cleaned.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    logger.info(f"Generating tags for {len(catalog)} catalog items...")
    for i, item in enumerate(catalog):
        tags = generate_tags_for_item(client, settings.llm_model, item["name"], item.get("description", ""))
        item["tags"] = tags
        logger.info(f"  [{i+1}/{len(catalog)}] {item['name']} -> {tags}")
        time.sleep(0.3)  # stay well under free-tier rate limits

    with open("data/processed/catalog_enriched.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info("Done. Saved to data/processed/catalog_enriched.json")


if __name__ == "__main__":
    main()
