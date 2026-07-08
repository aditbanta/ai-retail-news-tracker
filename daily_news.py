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

CLAUDE_MODEL = "claude-sonnet-4-20250514"
LOOKBACK_HOURS = 24
SKIP_TOKEN = "SKIP"

PROMPT_TEMPLATE = (
    "You are an AI analyst for Value Retail (owner of the Bicester "
    "Collection). Given this article title and summary, write a "
    "1-sentence summary focused on what matters for luxury retail, "
    "leasing, or fashion. If this is generic AI hype with no concrete "
    "retail example, respond with 'SKIP'. Title: {title}. Summary: {summary}"
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
                model=CLAUDE_MODEL,
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
    Save the analysis text to a dated markdown file and overwrite
    LATEST_ANALYSIS.md. Returns the dated filename, or None on failure.
    """
    dated_filename = f"daily_analysis_{date_str}.md"
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
            todays_articles.append({
                "title": title,
                "summary": summary,
                "source": feed_url,
                "link": link,
            })
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
