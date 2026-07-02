"""
Cleans SHL's real JSON catalog feed (see scripts/scraper/scrape_catalog.py)
into the shape app/models/catalog.py's CatalogItem expects.

Two non-trivial decisions made here, both worth spot-checking against
the actual data once you have it:

1. MULTI-CATEGORY ITEMS: the feed's "keys" field is a list (an item can
   be both "Simulations" and "Knowledge & Skills"), but the assignment's
   API schema requires exactly one `test_type` per recommendation. We
   take keys[0] as the primary type. If this produces obviously wrong
   primary types for specific items, adjust _CATEGORY_PRIORITY below to
   reorder which category wins when multiple are present.

2. JOB SOLUTIONS FILTER: the feed appears to mix Individual Test
   Solutions with Pre-packaged Job Solutions (which the assignment
   explicitly puts out of scope), with no explicit type flag to
   separate them. We filter by a name-pattern heuristic (ends with
   "Solution"/"Solutions"). This is an approximation -- after running,
   check data/processed/excluded_as_job_solutions.json and manually
   verify nothing wrongly excluded/included.

Run: python scripts/enrichment/enrich_catalog.py
"""

import json
import re
import sys

sys.path.insert(0, ".")

from app.core.logging import get_logger

logger = get_logger(__name__)

# Maps SHL's full category names to the single-letter codes the
# assignment's API schema requires (see app/core/constants.py).
_CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",  # British spelling variant, just in case
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}

_JOB_SOLUTION_NAME_RE = re.compile(r"\bsolutions?\b\s*$", re.IGNORECASE)


def _is_likely_job_solution(name: str) -> bool:
    return bool(_JOB_SOLUTION_NAME_RE.search(name.strip()))


def _primary_test_type(keys: list[str]) -> str:
    for key in keys:
        code = _CATEGORY_TO_CODE.get(key)
        if code:
            return code
    logger.warning(f"No recognized category in {keys} -- defaulting to 'K'")
    return "K"


def _parse_duration_minutes(duration_raw: str) -> int | None:
    match = re.search(r"(\d+)", duration_raw or "")
    return int(match.group(1)) if match else None


def clean_item(raw_item: dict) -> dict:
    return {
        "id": str(raw_item["entity_id"]),
        "name": raw_item["name"].strip(),
        "url": raw_item["link"],
        "test_type": _primary_test_type(raw_item.get("keys", [])),
        "description": (raw_item.get("description") or "").strip(),
        "job_levels": raw_item.get("job_levels", []) or [],
        "duration_minutes": _parse_duration_minutes(raw_item.get("duration_raw", "")),
        "remote_testing": (raw_item.get("remote") or "").lower() == "yes",
        "adaptive_irt": (raw_item.get("adaptive") or "").lower() == "yes",
        "tags": [],   # populated later by generate_tags.py
        "skills": [],
        "seniority": [],
    }


def main():
    with open("data/raw/shl_catalog_raw.json", "r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    logger.info(f"Loaded {len(raw_catalog)} raw entries.")

    individual_solutions = []
    excluded_job_solutions = []
    for raw_item in raw_catalog:
        if raw_item.get("status") != "ok":
            continue
        if _is_likely_job_solution(raw_item.get("name", "")):
            excluded_job_solutions.append(raw_item["name"])
            continue
        individual_solutions.append(clean_item(raw_item))

    logger.info(f"Kept {len(individual_solutions)} as Individual Test Solutions.")
    logger.info(
        f"Excluded {len(excluded_job_solutions)} items matching the 'Solution(s)' "
        "name pattern -- SPOT-CHECK these, see data/processed/excluded_as_job_solutions.json"
    )

    # dedup by id defensively
    seen_ids = set()
    deduped = []
    for item in individual_solutions:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            deduped.append(item)
        else:
            logger.warning(f"Duplicate id '{item['id']}' skipped")

    with open("data/processed/catalog_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    with open("data/processed/excluded_as_job_solutions.json", "w", encoding="utf-8") as f:
        json.dump(excluded_job_solutions, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. {len(deduped)} unique items saved to data/processed/catalog_cleaned.json")


if __name__ == "__main__":
    main()
