"""
Fetches SHL's own pre-scraped JSON catalog feed directly. Much simpler
and more reliable than HTML scraping — no Playwright/browser needed.

Run: python scrape_catalog.py
Output: ../../data/raw/shl_catalog_raw.json
"""

import json
import os
import requests

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = "../../data/raw/shl_catalog_raw.json"


def main():
    print(f"Fetching {CATALOG_URL} ...")
    resp = requests.get(CATALOG_URL, timeout=30)
    resp.raise_for_status()

    # SHL's feed contains raw control characters inside some string
    # fields (e.g. stray newlines in a description), which strict JSON
    # parsing rejects. strict=False tolerates them.
    data = json.loads(resp.text, strict=False)
    print(f"Fetched {len(data)} total catalog entries.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()