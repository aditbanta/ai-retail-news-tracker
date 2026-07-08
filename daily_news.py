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
import json
import time
import smtplib
from email.mime.text import MIMEText
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
    "https://wwd.com/feed/",
    "https://www.drapersonline.com/feed",
    "https://www.theguardian.com/fashion/rss",
    "https://www.retaildesignblog.net/feed",
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

# Some feeds reject feedparser's default user-agent and return an HTML
# error page instead of XML (causing "not well-formed" parse errors).
# A browser-like UA avoids that in most cases.
FEED_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

CLAUDE_MODEL = "claude-sonnet-5"
ANALYSIS_MODEL = "claude-sonnet-5"
LOOKBACK_HOURS = 24
SKIP_TOKEN = "SKIP"

PROMPT_TEMPLATE = (
    "You are an AI analyst for Value Retail (owner of the Bicester "
    "Collection). Given this article title and summary, write a "
    "1-sentence summary focused on what matters for luxury retail, "
    "leasing, or fashion. If this is generic AI hype with no concrete "
    "retail example, respond with 'SKIP'. Title: {title}. Summary: {summary}"
)

# Web search stage: casts a wider net than the fixed RSS feeds by letting
# Claude search the open web directly for AI-related retail/fashion/leasing
# news from the last 24 hours.
WEB_SEARCH_MAX_USES = 6
WEB_SEARCH_MAX_TOKENS = 2000
WEB_SEARCH_PROMPT = (
    "Search the web for news articles published in the last 24 hours about "
    "artificial intelligence, machine learning, generative AI, computer "
    "vision, or ChatGPT as they relate to retail, fashion, or commercial "
    "leasing. Run several distinct searches to cover different angles "
    "(e.g. AI in luxury retail, AI-driven leasing/pricing, AI in fashion "
    "brands, AI in-store technology). "
    "After searching, respond with ONLY a JSON array (no markdown code "
    "fences, no commentary before or after) of the genuinely relevant, "
    "substantive articles you found. Each element must be an object with "
    "these exact fields: \"title\", \"summary\" (1-2 sentences), \"link\" "
    "(the article URL), and \"source\" (the publication name). Exclude "
    "generic AI hype pieces with no concrete retail example, and exclude "
    "anything not published in roughly the last 24 hours. If you find "
    "nothing that qualifies, respond with an empty JSON array: []"
)

ANALYSIS_PROMPT_TEMPLATE = (
    "You are an AI analyst for Value Retail, owner of the Bicester "
    "Collection (luxury outlet shopping villages across Europe). "
    "Analyse the following AI-related retail/fashion/leasing news "
    "articles from the last 24 hours.\n"
    "Provide:\n"
    "1. THEME CLUSTERS - Group articles by theme. Name each cluster "
    "by the claim or shift (e.g. 'AI-driven dynamic leasing pricing "
    "is going mainstream'), not by source. For each cluster: "
    "one-paragraph synthesis and a 'So What for Value Retail' line "
    "covering leasing strategy, tenant mix, in-store experience, "
    "or brand partnerships.\n"
    "2. WHO TO WATCH - Companies or executives whose AI moves in "
    "retail/leasing/fashion drove the most discussion.\n"
    "3. SIGNAL VS NOISE - Flag which items are genuine signal vs "
    "generic trend pieces.\n"
    "Articles:\n{articles}"
)

ANALYSIS_MAX_TOKENS = 2000
ANALYSIS_ARCHIVE_DIR = "analysis"

# Email configuration
EMAIL_RECIPIENT = "abanta@valueretail.com"
NO_NEWS_EMAIL_BODY = "No significant AI retail news found in the last 24 hours."


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
        parsed = feedparser.parse(feed_url, request_headers=FEED_REQUEST_HEADERS)
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


def call_claude_web_search(client, retries=2, backoff=2.0):
    """
    Use Claude's built-in web search tool to find AI-related retail/
    fashion/leasing news from across the web, beyond the fixed RSS feeds.
    Returns a list of dicts with keys title/summary/link/source, or an
    empty list if nothing qualifies or the call fails after retries.
    """
    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=WEB_SEARCH_MAX_TOKENS,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                }],
                messages=[{"role": "user", "content": WEB_SEARCH_PROMPT}],
            )
            # Only the final text blocks contain Claude's answer; search
            # activity shows up as separate server_tool_use /
            # web_search_tool_result blocks, which we ignore here.
            text_parts = [
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            raw_text = "".join(text_parts).strip()

            # Defensively strip markdown code fences in case Claude adds
            # them despite instructions not to.
            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            try:
                articles = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"  Warning: web search response was not valid JSON "
                      f"(first 200 chars): {raw_text[:200]!r}")
                return []

            if not isinstance(articles, list):
                print("  Warning: web search response JSON was not a list. Ignoring.")
                return []

            # Keep only well-formed entries with the fields we need.
            valid_articles = [
                a for a in articles
                if isinstance(a, dict) and a.get("title") and a.get("link")
            ]
            return valid_articles

        except (APIStatusError, APIConnectionError, APIError) as e:
            print(f"  Claude API error on web search attempt {attempt}: {e}")
            if attempt <= retries:
                time.sleep(backoff * attempt)
            else:
                return []
        except Exception as e:
            print(f"  Unexpected error during web search: {e}")
            return []
    return []


def process_candidate_article(client, title, summary, link, source, existing_links):
    """
    Shared pipeline for a single candidate article, regardless of whether
    it came from an RSS feed or the web search stage: checks for a
    duplicate link, then calls Claude for a retail-focused summary and
    applies the SKIP filter.

    Returns a tuple (status, row, article) where status is one of
    'duplicate', 'error', 'skipped_hype', or 'added'. row and article are
    only populated when status == 'added'.
    """
    if link in existing_links:
        return ("duplicate", None, None)

    claude_response = call_claude(client, title, summary)

    if claude_response is None:
        return ("error", None, None)

    if claude_response.strip().upper() == SKIP_TOKEN:
        return ("skipped_hype", None, None)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = [date_str, claude_response, link, source]
    article = {"title": title, "summary": summary, "source": source, "link": link}
    return ("added", row, article)


def format_articles_for_analysis(articles):
    """
    Format a list of article dicts (title, summary, source, link) into a
    numbered plain-text block suitable for insertion into the analysis
    prompt.
    """
    lines = []
    for i, art in enumerate(articles, start=1):
        lines.append(
            f"{i}. Title: {art['title']}\n"
            f"   Summary: {art['summary']}\n"
            f"   Source: {art['source']}\n"
            f"   Link: {art['link']}"
        )
    return "\n\n".join(lines)


def call_claude_analysis(client, articles, retries=2, backoff=2.0):
    """
    Call the Claude API with the full list of today's articles to produce
    a theme-cluster analysis. Returns the analysis text, or None on
    repeated failure.
    """
    formatted = format_articles_for_analysis(articles)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(articles=formatted)

    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=ANALYSIS_MODEL,
                max_tokens=ANALYSIS_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            return "".join(text_parts).strip()
        except (APIStatusError, APIConnectionError, APIError) as e:
            print(f"  Claude API error on analysis attempt {attempt}: {e}")
            if attempt <= retries:
                time.sleep(backoff * attempt)
            else:
                return None
        except Exception as e:
            print(f"  Unexpected error calling Claude API for analysis: {e}")
            return None
    return None


def save_analysis_files(analysis_text, date_str):
    """
    Save the analysis text to a dated markdown file under ANALYSIS_ARCHIVE_DIR/
    and overwrite LATEST_ANALYSIS.md at the repo root. Returns the dated
    filename (including its subfolder path), or None on failure.
    """
    try:
        os.makedirs(ANALYSIS_ARCHIVE_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error: failed to create '{ANALYSIS_ARCHIVE_DIR}/' directory: {e}")
        return None

    dated_filename = os.path.join(ANALYSIS_ARCHIVE_DIR, f"daily_analysis_{date_str}.md")
    try:
        with open(dated_filename, "w", encoding="utf-8") as f:
            f.write(analysis_text)
    except Exception as e:
        print(f"Error: failed to write '{dated_filename}': {e}")
        dated_filename = None

    try:
        with open("LATEST_ANALYSIS.md", "w", encoding="utf-8") as f:
            f.write(analysis_text)
    except Exception as e:
        print(f"Error: failed to write 'LATEST_ANALYSIS.md': {e}")

    return dated_filename


def send_email(subject, body):
    """
    Send a plain-text email using SMTP credentials from environment
    variables. Returns True on success, False on failure.
    """
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    missing = [
        name for name, val in [
            ("SMTP_SERVER", smtp_server),
            ("SMTP_PORT", smtp_port),
            ("SMTP_USER", smtp_user),
            ("SMTP_PASSWORD", smtp_password),
        ] if not val
    ]
    if missing:
        print(f"Error: missing SMTP environment variable(s): "
              f"{', '.join(missing)}. Skipping email send.")
        return False

    try:
        smtp_port = int(smtp_port)
    except ValueError:
        print(f"Error: SMTP_PORT '{smtp_port}' is not a valid integer.")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = EMAIL_RECIPIENT

    try:
        # Port 465 conventionally means implicit SSL; otherwise use
        # STARTTLS on the given port (e.g. 587).
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [EMAIL_RECIPIENT], msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [EMAIL_RECIPIENT], msg.as_string())
        print(f"Email sent to {EMAIL_RECIPIENT}.")
        return True
    except Exception as e:
        print(f"Error: failed to send email: {e}")
        return False


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
    todays_articles = []  # dicts with title, summary, source, link — for analysis stage

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

            print(f"  -> Analyzing: {title[:80]}")
            status, row, article = process_candidate_article(
                client, title, summary, link, feed_url, existing_links
            )

            if status == "duplicate":
                total_skipped_dupe += 1
            elif status == "error":
                total_errors += 1
                print("     Skipped due to API error.")
            elif status == "skipped_hype":
                total_skipped_claude += 1
                print("     Claude judged this generic AI hype. Skipped.")
            elif status == "added":
                rows_to_write.append(row)
                todays_articles.append(article)
                existing_links.add(link)  # prevent intra-run duplicates too
                total_added += 1

    # ------------------------------------------------------------------
    # Web search stage: broaden coverage beyond the fixed RSS feeds by
    # letting Claude search the open web directly for AI retail news.
    # ------------------------------------------------------------------
    print("\nRunning supplemental web search for AI retail news...")
    web_articles = call_claude_web_search(client)
    print(f"  {len(web_articles)} candidate article(s) found via web search.")

    for art in web_articles:
        title = art.get("title", "") or ""
        summary = art.get("summary", "") or ""
        link = art.get("link", "") or ""
        source = art.get("source", "") or "Web Search"

        if not link:
            continue

        total_found += 1

        print(f"  -> Analyzing: {title[:80]}")
        status, row, article = process_candidate_article(
            client, title, summary, link, source, existing_links
        )

        if status == "duplicate":
            total_skipped_dupe += 1
        elif status == "error":
            total_errors += 1
            print("     Skipped due to API error.")
        elif status == "skipped_hype":
            total_skipped_claude += 1
            print("     Claude judged this generic AI hype. Skipped.")
        elif status == "added":
            rows_to_write.append(row)
            todays_articles.append(article)
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

    # ----------------------------------------------------------------------
    # Stage 2: theme-cluster analysis + email digest
    # ----------------------------------------------------------------------
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    email_subject = f"\U0001F916 AI Retail Intel - {today_str}"

    if not todays_articles:
        print("\nNo new articles today — sending 'no news' email.")
        send_email(email_subject, NO_NEWS_EMAIL_BODY)
        return

    print(f"\nRunning theme-cluster analysis on {len(todays_articles)} article(s)...")
    analysis_text = call_claude_analysis(client, todays_articles)

    if analysis_text is None:
        print("Error: analysis generation failed after retries. "
              "Skipping file save and email.")
        return

    dated_filename = save_analysis_files(analysis_text, today_str)
    if dated_filename:
        print(f"Analysis saved to '{dated_filename}' and 'LATEST_ANALYSIS.md'.")
    else:
        print("Analysis saved to 'LATEST_ANALYSIS.md' only (dated file failed).")

    send_email(email_subject, analysis_text)


if __name__ == "__main__":
    main()
