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

# Cost optimization: use haiku for fast classification tasks, sonnet only for
# strategic analysis where quality matters.
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-5"
LOOKBACK_HOURS = 24
SKIP_TOKEN = "SKIP"

# Filtering prompt: concise, focused on signal vs. noise classification.
PROMPT_TEMPLATE = (
    "Is this concrete retail/fashion/leasing AI news (not generic hype)? "
    "Respond 'SKIP' if generic, otherwise 1 sentence on retail impact. "
    "Title: {title}. Summary: {summary}"
)

# Web search stage: casts a wider net than the fixed RSS feeds by letting
# Claude search the open web directly for AI-related retail/fashion/leasing
# news from the last 24 hours.
WEB_SEARCH_MAX_USES = 6
WEB_SEARCH_MAX_TOKENS = 1500
WEB_SEARCH_PROMPT = (
    "Search the web for AI/ML/generative AI news in retail/fashion/leasing "
    "from the last 24 hours. Run several searches (luxury retail, leasing "
    "pricing, fashion brands, in-store tech). Return ONLY a JSON array of "
    "substantive articles (not hype), each with: title, summary (1-2 "
    "sentences), link, source. If none found, return []"
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
    "retail/leasing/fashion drove the most discussion.\n\n"
    "Format your response in clean HTML. Use these exact elements:\n"
    "- <h3> for each cluster name (sentence case)\n"
    "- <p> for the synthesis paragraph\n"
    "- <div class='so-what'><span class='so-what-label'>So what for Value Retail</span> followed by the insight text</div> for each So What callout\n"
    "- For the WHO TO WATCH section, use <div class='watch-grid'> containing individual <div class='watch-card'><strong>Company name</strong><span>one-line description</span></div> entries\n"
    "- Use <p> tags for all body text\n"
    "Do NOT include <html>, <head>, <body>, <style>, or any section title tags "
    "for THEME CLUSTERS or WHO TO WATCH — the template handles those.\n"
    "Separate the theme clusters section from the who to watch section with "
    "exactly this marker on its own line: |||WHO_TO_WATCH|||\n"
    "Articles:\n{articles}"
)

ANALYSIS_MAX_TOKENS = 2000
ANALYSIS_ARCHIVE_DIR = "analysis"

# Email configuration
EMAIL_RECIPIENTS = [
    "abanta@valueretail.com",
    "ofriedman@valueretail.com",
]
NO_NEWS_EMAIL_BODY = "No significant AI retail news found in the last 24 hours."

# Bicester Collection brand colours
COLOR_NATURAL_GREEN = "#7E8A4A"
COLOR_SANDSTONE = "#F5F0E6"
COLOR_RADIANT_GREEN = "#BAF763"
COLOR_RACING_GREEN = "#233B2B"
COLOR_SANDSTONE_CARD = "#fafaf7"
COLOR_BORDER = "#e8e4da"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clean_csv_skip_rows(csv_path):
    """
    Remove any rows where the Article Summary starts with 'SKIP' —
    these were written erroneously during broken model runs and should
    never appear in the log. Rewrites the file in place.
    Returns the number of bad rows removed.
    """
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        clean_rows = [
            r for r in rows
            if not r.get("Article Summary", "").strip().upper().startswith("SKIP")
        ]
        removed = len(rows) - len(clean_rows)

        if removed > 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(clean_rows)
            print(f"Cleaned CSV: removed {removed} erroneous SKIP row(s).")

        return removed
    except Exception as e:
        print(f"Warning: could not clean CSV ({e}).")
        return 0


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
    Call Claude Haiku to classify an article as signal or noise/hype.
    Returns the stripped text response, or None on repeated failure.
    """
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)

    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=150,
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
    Use Claude Haiku with the web search tool to find AI-related
    retail/fashion/leasing news from across the web. Returns a list of
    dicts with keys title/summary/link/source, or an empty list on failure.
    """
    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=WEB_SEARCH_MAX_TOKENS,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                }],
                messages=[{"role": "user", "content": WEB_SEARCH_PROMPT}],
            )
            text_parts = [
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            raw_text = "".join(text_parts).strip()

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

            return [
                a for a in articles
                if isinstance(a, dict) and a.get("title") and a.get("link")
            ]

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
    Shared pipeline for a single candidate article: checks for a duplicate
    link, calls Claude Haiku for a retail-focused summary, and applies the
    SKIP filter.

    Returns a tuple (status, row, article) where status is one of
    'duplicate', 'error', 'skipped_hype', or 'added'.
    """
    if link in existing_links:
        return ("duplicate", None, None)

    claude_response = call_claude(client, title, summary)

    if claude_response is None:
        return ("error", None, None)

    if SKIP_TOKEN in claude_response.strip().upper():
        return ("skipped_hype", None, None)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = [date_str, claude_response, link, source]
    article = {"title": title, "summary": summary, "source": source, "link": link}
    return ("added", row, article)


def format_articles_for_analysis(articles):
    """
    Format a list of article dicts into a numbered plain-text block
    suitable for insertion into the analysis prompt.
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
    Call Claude Sonnet with the full list of today's articles to produce
    a theme-cluster analysis with structured HTML output.
    Returns the analysis text, or None on repeated failure.
    """
    formatted = format_articles_for_analysis(articles)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(articles=formatted)

    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=SONNET_MODEL,
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
    and overwrite LATEST_ANALYSIS.md at the repo root.
    Returns the dated filename, or None on failure.
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


def build_inline_styles():
    """Return a dict of reusable inline style strings for the email template."""
    return {
        "so_what_div": (
            f"border-left:3px solid {COLOR_NATURAL_GREEN}; "
            f"background:{COLOR_SANDSTONE}; "
            f"padding:16px 20px; margin-top:4px; margin-bottom:0;"
        ),
        "so_what_label": (
            f"display:block; font-size:10px; font-weight:400; "
            f"color:{COLOR_NATURAL_GREEN}; text-transform:uppercase; "
            f"letter-spacing:2px; margin-bottom:8px; font-family:Arial,sans-serif;"
        ),
        "so_what_text": (
            "margin:0; font-size:13px; line-height:1.7; color:#333; "
            "font-family:Arial,sans-serif; font-weight:300;"
        ),
        "watch_grid": (
            "display:grid; grid-template-columns:1fr 1fr; gap:12px;"
        ),
        "watch_card": (
            f"border:1px solid {COLOR_BORDER}; padding:16px 18px; "
            f"background:{COLOR_SANDSTONE_CARD};"
        ),
        "watch_name": (
            f"display:block; font-size:14px; font-weight:400; "
            f"color:{COLOR_RACING_GREEN}; margin-bottom:6px; "
            f"font-family:Georgia,serif;"
        ),
        "watch_desc": (
            "display:block; font-size:13px; color:#666; line-height:1.6; "
            "font-family:Arial,sans-serif; font-weight:300;"
        ),
    }


def wrap_analysis_in_html_template(analysis_text, date_str):
    """
    Split Claude's output on the |||WHO_TO_WATCH||| marker, apply inline
    CSS transformations for branded callouts and watch cards, then wrap
    everything in the Bicester Collection email shell.
    """
    styles = build_inline_styles()

    # Split on the section marker Claude was asked to include
    parts = analysis_text.split("|||WHO_TO_WATCH|||", 1)
    clusters_html = parts[0].strip() if parts else analysis_text.strip()
    who_html = parts[1].strip() if len(parts) > 1 else ""

    # Replace Claude's class-based so-what divs with fully inline versions
    clusters_html = clusters_html.replace(
        "<div class='so-what'>",
        f"<div style='{styles['so_what_div']}'>"
    ).replace(
        '<div class="so-what">',
        f"<div style='{styles['so_what_div']}'>"
    ).replace(
        "<span class='so-what-label'>",
        f"<span style='{styles['so_what_label']}'>"
    ).replace(
        '<span class="so-what-label">',
        f"<span style='{styles['so_what_label']}'>"
    )

    # Replace watch grid and card classes with inline styles
    who_html = who_html.replace(
        "<div class='watch-grid'>",
        f"<div style='{styles['watch_grid']}'>"
    ).replace(
        '<div class="watch-grid">',
        f"<div style='{styles['watch_grid']}'>"
    ).replace(
        "<div class='watch-card'>",
        f"<div style='{styles['watch_card']}'>"
    ).replace(
        '<div class="watch-card">',
        f"<div style='{styles['watch_card']}'>"
    ).replace(
        "<strong>",
        f"<strong style='{styles['watch_name']}'>"
    ).replace(
        "<span>",
        f"<span style='{styles['watch_desc']}'>"
    )

    # Style h3 cluster headings
    clusters_html = clusters_html.replace(
        "<h3>",
        f"<h3 style='margin:0 0 10px 0; font-size:17px; font-weight:400; "
        f"color:{COLOR_RACING_GREEN}; font-family:Georgia,serif; letter-spacing:-0.3px;'>"
    )

    # Style body paragraphs
    clusters_html = clusters_html.replace(
        "<p>",
        "<p style='margin:0 0 14px 0; font-size:14px; line-height:1.75; "
        "color:#444; font-family:Arial,sans-serif; font-weight:300;'>"
    )
    who_html = who_html.replace(
        "<p>",
        "<p style='margin:0 0 14px 0; font-size:14px; line-height:1.75; "
        "color:#444; font-family:Arial,sans-serif; font-weight:300;'>"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:20px; background-color:{COLOR_SANDSTONE}; font-family:Arial,sans-serif;">
<div style="max-width:680px; margin:0 auto; background:#ffffff; overflow:hidden;">

    <!-- Header -->
    <div style="background:{COLOR_NATURAL_GREEN}; padding:40px 40px 32px; text-align:center;">
        <div style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.65); margin-bottom:14px; font-family:Arial,sans-serif;">The Bicester Collection</div>
        <h1 style="margin:0; font-size:30px; font-weight:400; color:#ffffff; font-family:Georgia,serif; letter-spacing:-0.5px;">AI Retail Intel</h1>
        <div style="margin-top:12px; font-size:12px; color:rgba(255,255,255,0.6); font-family:Arial,sans-serif; letter-spacing:1px;">{date_str}</div>
    </div>

    <!-- Radiant Green accent bar -->
    <div style="height:4px; background:{COLOR_RADIANT_GREEN};"></div>

    <!-- Main content -->
    <div style="padding:36px 40px; background:#ffffff;">

        <!-- Theme Clusters section -->
        <div style="margin-bottom:36px;">
            <div style="margin-bottom:24px; padding-bottom:12px; border-bottom:1px solid {COLOR_BORDER};">
                <h2 style="margin:0; font-size:11px; font-weight:400; color:{COLOR_NATURAL_GREEN}; text-transform:uppercase; letter-spacing:3px; font-family:Arial,sans-serif;">Theme clusters</h2>
            </div>
            {clusters_html}
        </div>

        <!-- Divider -->
        <div style="height:1px; background:{COLOR_BORDER}; margin-bottom:36px;"></div>

        <!-- Who to Watch section -->
        <div>
            <div style="margin-bottom:24px; padding-bottom:12px; border-bottom:1px solid {COLOR_BORDER};">
                <h2 style="margin:0; font-size:11px; font-weight:400; color:{COLOR_NATURAL_GREEN}; text-transform:uppercase; letter-spacing:3px; font-family:Arial,sans-serif;">Who to watch</h2>
            </div>
            {who_html}
        </div>

    </div>

    <!-- Footer -->
    <div style="background:{COLOR_RACING_GREEN}; padding:24px 40px; text-align:center;">
        <div style="font-size:10px; color:rgba(255,255,255,0.5); letter-spacing:2px; text-transform:uppercase; font-family:Arial,sans-serif;">Automated AI Intelligence Report</div>
        <div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:6px; font-family:Arial,sans-serif;">Value Retail — Bicester Collection</div>
    </div>

</div>
</body>
</html>"""
    return html


def wrap_no_news_in_html_template(date_str):
    """
    Create a branded HTML email for the 'no news' case.
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:20px; background-color:{COLOR_SANDSTONE}; font-family:Arial,sans-serif;">
<div style="max-width:680px; margin:0 auto; background:#ffffff; overflow:hidden;">

    <!-- Header -->
    <div style="background:{COLOR_NATURAL_GREEN}; padding:40px 40px 32px; text-align:center;">
        <div style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.65); margin-bottom:14px; font-family:Arial,sans-serif;">The Bicester Collection</div>
        <h1 style="margin:0; font-size:30px; font-weight:400; color:#ffffff; font-family:Georgia,serif; letter-spacing:-0.5px;">AI Retail Intel</h1>
        <div style="margin-top:12px; font-size:12px; color:rgba(255,255,255,0.6); font-family:Arial,sans-serif; letter-spacing:1px;">{date_str}</div>
    </div>

    <!-- Radiant Green accent bar -->
    <div style="height:4px; background:{COLOR_RADIANT_GREEN};"></div>

    <!-- Message -->
    <div style="padding:60px 40px; text-align:center;">
        <p style="font-size:15px; color:#666; font-family:Georgia,serif; font-weight:400; margin:0;">No significant AI retail news found in the last 24 hours.</p>
        <p style="font-size:13px; color:#999; font-family:Arial,sans-serif; font-weight:300; margin:16px 0 0 0;">Check back tomorrow for the latest developments.</p>
    </div>

    <!-- Footer -->
    <div style="background:{COLOR_RACING_GREEN}; padding:24px 40px; text-align:center;">
        <div style="font-size:10px; color:rgba(255,255,255,0.5); letter-spacing:2px; text-transform:uppercase; font-family:Arial,sans-serif;">Automated AI Intelligence Report</div>
        <div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:6px; font-family:Arial,sans-serif;">Value Retail — Bicester Collection</div>
    </div>

</div>
</body>
</html>"""
    return html


def send_email(subject, body, is_html=True):
    """
    Send an HTML email using SMTP credentials from environment variables.
    Returns True on success, False on failure.
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

    msg_type = "html" if is_html else "plain"
    msg = MIMEText(body, msg_type, "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(EMAIL_RECIPIENTS)

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, EMAIL_RECIPIENTS, msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, EMAIL_RECIPIENTS, msg.as_string())
        print(f"Email sent to {', '.join(EMAIL_RECIPIENTS)}.")
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
    clean_csv_skip_rows(CSV_FILE)
    existing_links = load_existing_links(CSV_FILE)

    total_found = 0
    total_skipped_dupe = 0
    total_skipped_claude = 0
    total_errors = 0
    total_added = 0

    rows_to_write = []
    todays_articles = []

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
                existing_links.add(link)
                total_added += 1

    # ------------------------------------------------------------------
    # Web search stage
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
            existing_links.add(link)
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
    print(f"Skipped as duplicates (already logged):               {total_skipped_dupe}")
    print(f"Skipped by Claude (generic AI hype):                  {total_skipped_claude}")
    print(f"Skipped due to errors:                                {total_errors}")
    print(f"New articles added to {CSV_FILE}:          {total_added}")

    # ------------------------------------------------------------------
    # Stage 2: analysis + branded email
    # ------------------------------------------------------------------
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    email_subject = f"AI Retail Intel - {today_str}"

    if not todays_articles:
        print("\nNo new articles today — sending 'no news' email.")
        send_email(email_subject, wrap_no_news_in_html_template(today_str), is_html=True)
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

    full_html = wrap_analysis_in_html_template(analysis_text, today_str)
    send_email(email_subject, full_html, is_html=True)


if __name__ == "__main__":
    main()
