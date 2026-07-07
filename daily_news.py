#!/usr/bin/env python3
"""
daily_news.py

Pulls RSS feeds from retail/fashion trade publications, filters for
AI-related articles published in the last 24 hours, asks Claude to
summarize each one with a Value Retail / Bicester Collection lens,
and logs the useful ones to a CSV (skipping generic AI hype and
duplicates).

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python daily_news.py
"""

import os
import sys
import csv
import time
from datetime import datetime, timedelta, timezone

import feedparser
from anthropic import Anthropic, APIError, APIStatusError, APIConnectionError

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RSS_FEEDS = [
    "https://www.retaildive.com/feeds/news/",
    "https://www.voguebusiness.com/rss",
    "https://risnews.com/rss.xml",
]

KEYWORDS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "generative ai",
    "chatgpt",
    "computer vision",
]

CSV_FILE = "AI_Retail_News_Log.csv"
CSV_HEADERS = ["Date", "Article Summary", "Link", "Source"]

CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
LOOKBACK_HOURS = 24
SKIP_TOKEN = "SKIP"

PROMPT_TEMPLATE = (
    "You are an AI analyst for Value Retail (owner of the Bicester "
    "Collection). Given this article title and summary, write a "
    "1-sentence summary focused on what matters for luxury retail, "
    "leasing, or fashion. If this is generic AI hype with no concrete "
    "retail example, respond with 'SKIP'. Title: {title}. Summary: {summary}"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_existing_links(csv_path):
    """Return a set of links already present in the CSV log (if any)."""
    existing_links = set()
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    link = row.get("Link")
                    if link:
                        existing_links.add(link.strip())
        except Exception as e:
            print(f"Warning: could not fully read existing CSV ({e}). "
                  f"Proceeding with what was loaded.")
    return existing_links


def ensure_csv_has_headers(csv_path):
    """Create the CSV with headers if it doesn't already exist."""
    if not os.path.isfile(csv_path):
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
        except Exception as e:
            print(f"Error: could not create CSV file '{csv_path}': {e}")
            sys.exit(1)


def get_entry_datetime(entry):
    """
    Try to extract a timezone-aware publish datetime from a feedparser entry.
    Falls back to None if no usable date is found.
    """
    for field in ("published_parsed", "updated_parsed"):
        struct_time = getattr(entry, field, None)
        if struct_time:
            try:
                return datetime(*struct_time[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def is_within_lookback(entry_dt, hours=LOOKBACK_HOURS):
    if entry_dt is None:
        # If we can't determine a date, don't silently include it —
        # skip it to avoid processing stale/undated articles repeatedly.
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return entry_dt >= cutoff


def matches_keywords(title, summary, keywords=KEYWORDS):
    haystack = f"{title or ''} {summary or ''}".lower()
    return any(kw in haystack for kw in keywords)


def get_entry_summary(entry):
    return getattr(entry, "summary", "") or getattr(entry, "description", "") or ""


def get_entry_link(entry):
    return getattr(entry, "link", "") or ""


def fetch_feed_entries(feed_url):
    """Fetch and parse a single RSS feed. Returns a list of entries."""
    try:
        parsed = feedparser.parse(feed_url)
        if parsed.bozo and not parsed.entries:
            print(f"Warning: feed '{feed_url}' could not be parsed cleanly "
                  f"({parsed.bozo_exception}). Skipping.")
            return []
        return parsed.entries
    except Exception as e:
        print(f"Warning: failed to fetch feed '{feed_url}': {e}")
        return []


def call_claude(client, title, summary, retries=2, backoff=2.0):
    """
    Call the Claude API to get a retail-focused 1-sentence summary,
    or 'SKIP'. Returns the stripped text response, or None on repeated
    failure.
    """
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)

    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            return "".join(text_parts).strip()
        except (APIStatusError, APIConnectionError, APIError) as e:
            print(f"  Claude API error on attempt {attempt}: {e}")
            if attempt <= retries:
                time.sleep(backoff * attempt)
            else:
                return None
        except Exception as e:
            print(f"  Unexpected error calling Claude API: {e}")
            return None
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    ensure_csv_has_headers(CSV_FILE)
    existing_links = load_existing_links(CSV_FILE)

    total_found = 0          # matched keyword + within 24h
    total_skipped_dupe = 0   # already in CSV
    total_skipped_claude = 0 # Claude said SKIP
    total_errors = 0         # API/parse errors
    total_added = 0          # actually written to CSV

    rows_to_write = []

    for feed_url in RSS_FEEDS:
        print(f"Fetching feed: {feed_url}")
        entries = fetch_feed_entries(feed_url)
        print(f"  {len(entries)} entries retrieved.")

        for entry in entries:
            title = getattr(entry, "title", "") or ""
            summary = get_entry_summary(entry)
            link = get_entry_link(entry)

            if not link:
                continue

            entry_dt = get_entry_datetime(entry)
            if not is_within_lookback(entry_dt):
                continue

            if not matches_keywords(title, summary):
                continue

            total_found += 1

            if link in existing_links:
                total_skipped_dupe += 1
                continue

            print(f"  -> Analyzing: {title[:80]}")
            claude_response = call_claude(client, title, summary)

            if claude_response is None:
                total_errors += 1
                print("     Skipped due to API error.")
                continue

            if claude_response.strip().upper() == SKIP_TOKEN:
                total_skipped_claude += 1
                print("     Claude judged this generic AI hype. Skipped.")
                continue

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rows_to_write.append([date_str, claude_response, link, feed_url])
            existing_links.add(link)  # prevent intra-run duplicates too
            total_added += 1

    if rows_to_write:
        try:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_write)
        except Exception as e:
            print(f"Error: failed to write to CSV file '{CSV_FILE}': {e}")
            sys.exit(1)

    print("\n--- Summary ---")
    print(f"Articles matching keywords in last {LOOKBACK_HOURS}h: {total_found}")
    print(f"Skipped as duplicates (already logged): {total_skipped_dupe}")
    print(f"Skipped by Claude (generic AI hype): {total_skipped_claude}")
    print(f"Skipped due to errors: {total_errors}")
    print(f"New articles added to {CSV_FILE}: {total_added}")


if __name__ == "__main__":
    main()
