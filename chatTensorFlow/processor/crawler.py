# ===================================================================================
# Project: ChatTensorFlow
# File: processor/crawler.py
# Description: This file is used to scrape the TensorFlow documentation page and extract the URLs.
# Author: LALAN KUMAR
# Created: [13-05-2025]
# Updated: [13-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================
# CAUTION: Please run this file only when there's a update to the documentation page.

import asyncio
import os
import sys
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Set
from urllib.parse import urlparse

import aiohttp
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from logger import logging

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

BASE_URL         = "https://www.tensorflow.org"
SITEMAP_URL      = f"{BASE_URL}/sitemap.xml"
OUTPUT_DIR       = "tensorflow_docs"
EXCLUDED_PATTERNS = [
    '/blog/', '/versions/', '/ecosystem/', '/resources/',
    '/community/', '/about/', '/responsible_ai',
    '/install/', '/learn/', '/js/', '/swift/', '/lite/',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── SITEMAP FETCHING ────────────────────────────────────────────────────────────

async def fetch_sitemap(url: str) -> List[str]:
    """
    Fetch all valid URLs from the sitemap (and any sub-sitemaps) in a single aiohttp session.
    Uses XML namespace 'http://www.sitemaps.org/schemas/sitemap/0.9' to locate <loc> entries. :contentReference[oaicite:3]{index=3}
    """
    urls: List[str] = []
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    async with aiohttp.ClientSession() as session:
        logging.info(f"Fetching sitemap: {url}")
        async with session.get(url) as resp:
            if resp.status != 200:
                logging.error(f"Sitemap fetch failed [{resp.status}]: {url}")
                return []
            sitemap_xml = await resp.text()

        root = ET.fromstring(sitemap_xml)
        index_entries = root.findall('.//sm:sitemap/sm:loc', ns)

        if index_entries:
            logging.info(f"Found {len(index_entries)} sub-sitemaps")
            for loc in index_entries:
                sub_url = loc.text or ""
                if not sub_url.startswith(BASE_URL) or any(p in sub_url for p in EXCLUDED_PATTERNS):
                    continue
                logging.info(f"  → Sub-sitemap: {sub_url}")
                async with session.get(sub_url) as sub_resp:
                    if sub_resp.status != 200:
                        continue
                    sub_xml = await sub_resp.text()
                sub_root = ET.fromstring(sub_xml)
                for url_loc in sub_root.findall('.//sm:url/sm:loc', ns):
                    link = url_loc.text or ""
                    if link.startswith(BASE_URL) and not any(p in link for p in EXCLUDED_PATTERNS):
                        urls.append(link)
        else:
            # Single sitemap case
            for url_loc in root.findall('.//sm:url/sm:loc', ns):
                link = url_loc.text or ""
                if link.startswith(BASE_URL) and not any(p in link for p in EXCLUDED_PATTERNS):
                    urls.append(link)

    logging.info(f"Total URLs to crawl: {len(urls)}")
    return urls

# ─── URL FILTER ─────────────────────────────────────────────────────────────────

def should_process_url(url: str, seen: Set[str]) -> bool:
    """
    Return True if URL is on-domain, not excluded, and not already processed.
    Ensures we only crawl TensorFlow docs pages. :contentReference[oaicite:4]{index=4}
    """
    if url in seen:
        return False
    parsed = urlparse(url)
    if not parsed.netloc.endswith("tensorflow.org"):
        return False
    if any(p in url for p in EXCLUDED_PATTERNS):
        return False
    return True

# ─── MAIN CRAWLER ────────────────────────────────────────────────────────────────

async def crawl_tensorflow_docs():
    # 1. Load URLs from sitemap
    urls = await fetch_sitemap(SITEMAP_URL)

    # 2. Configure Crawl4AI
    browser_cfg = BrowserConfig(headless=True)  # Chromium in headless mode
    run_cfg     = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=".devsite-article-body",    # Extract main article container :contentReference[oaicite:5]{index=5}
        word_count_threshold=50,
        screenshot=False
    )

    seen: Set[str]      = set()
    all_docs: List[dict] = []

    # 3. Crawl each page under a single AsyncWebCrawler context :contentReference[oaicite:6]{index=6}
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            if not should_process_url(url, seen):
                continue
            seen.add(url)

            logging.info(f"Crawling: {url}")
            try:
                result = await crawler.arun(url=url, config=run_cfg)
            except Exception as e:
                logging.error(f"  ❌ Error fetching {url}: {e}")
                continue

            if not result.success:
                logging.warning(f"  ⚠️ Failed: {result.error_message}")
                continue

            # 4. Extract title from metadata (if present) :contentReference[oaicite:7]{index=7}
            metadata = result.metadata or {}
            title = metadata.get("title", "").strip()

            # 5. Fallback to first line of raw_markdown if no metadata title :contentReference[oaicite:8]{index=8}
            if not title and result.markdown:
                raw_md = (result.markdown.raw_markdown
                          if hasattr(result.markdown, "raw_markdown")
                          else result.markdown)
                title = raw_md.split("\n", 1)[0].strip()

            # 6. Determine content markdown
            content_md = (result.markdown.raw_markdown
                          if hasattr(result.markdown, "raw_markdown")
                          else (result.markdown or ""))

            # 7. Record crawl time explicitly
            crawled_at = datetime.utcnow().isoformat() + "Z"

            doc = {
                "url":        url,
                "title":      title,
                "content":    content_md,
                "crawled_at": crawled_at,
            }

            # 8. Save individual page JSON
            fname = url.replace(BASE_URL, "").strip("/").replace("/", "_") or "index"
            path  = os.path.join(OUTPUT_DIR, f"{fname}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            logging.info(f"  ✅ Saved: {fname}.json")

            all_docs.append(doc)

    # 9. Write combined RAG JSON
    combined_path = os.path.join(OUTPUT_DIR, "tensorflow_docs_rag.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)
    logging.info(f"Combined RAG file saved: {combined_path}")

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(crawl_tensorflow_docs())


