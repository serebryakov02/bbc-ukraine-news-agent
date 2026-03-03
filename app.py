# Orchestration pattern: a manager agent coordinates specialized worker agents
# (news writer styles + email formatter/sender), enforcing order and handoffs
# so each run produces one controlled, tool-grounded output.
import asyncio
import datetime as dt
import json
import os
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import sendgrid
from agents import Agent, Runner, function_tool, trace
from dotenv import load_dotenv
from sendgrid.helpers.mail import Content, Email, Mail, To


load_dotenv(override=True)
MODEL = "gpt-4o-mini"
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")
BBC_UKRAINE_FEED_URL = os.getenv(
    "BBC_UKRAINE_FEED_URL", "https://feeds.bbci.co.uk/news/world/europe/rss.xml"
)
BBC_KEYWORDS = ("ukraine", "kyiv", "kiev")
NEWS_MAX_AGE_MINUTES = int(os.getenv("NEWS_MAX_AGE_MINUTES", "180"))


def _parse_rss_pub_date(value: str) -> dt.datetime:
    if not value:
        return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _empty_news_payload(now_utc: dt.datetime) -> Dict[str, object]:
    return {
        "fetched_at_utc": now_utc.isoformat(),
        "count": 0,
        "freshness_mode": "strict_recent_only",
        "freshness_window_minutes": NEWS_MAX_AGE_MINUTES,
        "fallback_used": False,
        "items": [],
    }


def _bounded_news_limit(limit: int) -> int:
    if limit < 1:
        return 1
    if limit > 5:
        return 5
    return limit


# Two distinct writing agents
polite_instructions = (
    "You are a polite news digest writer. "
    "Write a concise, neutral email body about Ukraine news with a short greeting and a short polite sign-off. "
    "Use only the BBC items provided in the input, and include source links. "
    "If the input does not contain concrete news details, say that latest developments could not be verified."
)

news_only_instructions = (
    "You are a news-only digest writer. "
    "Write only a list of Ukraine news items from the provided BBC input. "
    "Use concise bullet points and include source links. "
    "Do not add greeting, intro, outro, sign-off, opinions, or any extra text outside the news list. "
    "If concrete news details are missing, output a single bullet saying latest developments could not be verified."
)

polite_agent = Agent(
    name="Polite News Email Agent",
    instructions=polite_instructions,
    model=MODEL,
)

news_only_agent = Agent(
    name="News-Only Email Agent",
    instructions=news_only_instructions,
    model=MODEL,
)


# Convert the two writer agents into tools for the sequencing manager.
polite_tool = polite_agent.as_tool(
    tool_name="polite_news_writer",
    tool_description="Write a polite Ukraine-news email body with greeting and sign-off",
)

news_only_tool = news_only_agent.as_tool(
    tool_name="news_only_writer",
    tool_description="Write a news-only Ukraine update as a plain list",
)


@function_tool
def get_bbc_ukraine_news(limit: int = 5) -> Dict[str, object]:
    """Fetch Ukraine-related BBC headlines, prioritizing recent items and falling back to latest available."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    items = []

    try:
        with urlopen(BBC_UKRAINE_FEED_URL, timeout=10) as response:
            root = ET.fromstring(response.read())
    except Exception:
        return _empty_news_payload(now_utc)

    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        haystack = f"{title} {item.findtext('description') or ''}".lower()
        if not link or not any(k in haystack for k in BBC_KEYWORDS):
            continue

        pub_dt = _parse_rss_pub_date(pub_date)
        age_minutes = (now_utc - pub_dt).total_seconds() / 60
        items.append(
            {
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "pub_date_utc": pub_dt.isoformat(),
                "pub_timestamp": pub_dt.timestamp(),
                "age_minutes": round(max(age_minutes, 0.0), 1),
                "source_feed": BBC_UKRAINE_FEED_URL,
            }
        )

    items.sort(key=lambda x: x["pub_timestamp"], reverse=True)

    max_items = _bounded_news_limit(limit)
    fresh_cutoff = now_utc - dt.timedelta(minutes=NEWS_MAX_AGE_MINUTES)
    fresh_items = []
    for item in items:
        if item["pub_timestamp"] >= fresh_cutoff.timestamp():
            fresh_items.append(item)

    if fresh_items:
        selected_items = fresh_items[:max_items]
        selection_mode = "strict_recent_only"
        fallback_used = False
    else:
        selected_items = items[:max_items]
        selection_mode = "latest_available_fallback"
        fallback_used = bool(selected_items)

    payload_items = []
    for item in selected_items:
        payload_items.append(
            {
                "title": item["title"],
                "link": item["link"],
                "pub_date": item["pub_date"],
                "pub_date_utc": item["pub_date_utc"],
                "age_minutes": item["age_minutes"],
                "source_feed": item["source_feed"],
            }
        )

    return {
        "fetched_at_utc": now_utc.isoformat(),
        "count": len(payload_items),
        "freshness_mode": selection_mode,
        "freshness_window_minutes": NEWS_MAX_AGE_MINUTES,
        "fallback_used": fallback_used,
        "items": payload_items,
    }


# Email formatting/sending helpers used by Email Manager handoff.
subject_instructions = (
    "You write email subjects. "
    "Given an email body, create a short, specific subject line that matches the tone and content."
)

html_instructions = (
    "You convert plain text email bodies into clean HTML email bodies. "
    "Use simple sections, short paragraphs, and readable formatting."
)

subject_writer = Agent(
    name="Email Subject Writer",
    instructions=subject_instructions,
    model=MODEL,
)
subject_tool = subject_writer.as_tool(
    tool_name="subject_writer",
    tool_description="Write a subject line for an email body",
)

html_converter = Agent(
    name="HTML Email Converter",
    instructions=html_instructions,
    model=MODEL,
)
html_tool = html_converter.as_tool(
    tool_name="html_converter",
    tool_description="Convert plain text email body to HTML",
)


def _send_email_via_sendgrid(subject: str, body: str, content_type: str) -> Dict[str, str]:
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SENDGRID_API_KEY in environment.")
    if not FROM_EMAIL:
        raise RuntimeError("Missing FROM_EMAIL in environment.")
    if not TO_EMAIL:
        raise RuntimeError("Missing TO_EMAIL in environment.")

    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    from_email = Email(FROM_EMAIL)
    to_email = To(TO_EMAIL)
    final_subject = subject.strip() or "Ukraine news update"

    mail = Mail(from_email, to_email, final_subject, Content(content_type, body)).get()
    response = sg.client.mail.send.post(request_body=mail)
    status_code = int(getattr(response, "status_code", 0))
    body_text = getattr(response, "body", b"")
    if isinstance(body_text, bytes):
        body_text = body_text.decode("utf-8", errors="replace")
    headers = getattr(response, "headers", {}) or {}
    message_id = headers.get("X-Message-Id") or headers.get("x-message-id")

    if status_code < 200 or status_code >= 300:
        raise RuntimeError(f"SendGrid send failed: status={status_code}, body={body_text}")

    return {
        "status": "success",
        "subject": final_subject,
        "sendgrid_status_code": str(status_code),
        "sendgrid_message_id": str(message_id or ""),
    }


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send an email with the given subject and HTML body."""
    return _send_email_via_sendgrid(subject=subject, body=html_body, content_type="text/html")


emailer_instructions = (
    "You are an Email Manager. "
    "You receive one complete email body. "
    "First call subject_writer, then html_converter, then send_html_email. "
    "Send exactly one email per handoff."
)

emailer_agent = Agent(
    name="Email Manager",
    instructions=emailer_instructions,
    tools=[subject_tool, html_tool, send_html_email],
    model=MODEL,
    handoff_description="Format one email body as HTML and send it",
)


rotation_manager_instructions = """
You are a News Email Rotation Manager.

Your objective is to produce and send exactly 1 email per run.
The user message tells you which style to use: polite or news_only.

Workflow rules:
- First, call get_bbc_ukraine_news(limit=5) to fetch the latest BBC Ukraine headlines.
- If style is polite, call polite_news_writer once.
- If style is news_only, call news_only_writer once.
- Provide the fetched BBC headlines and links to the chosen writer tool as its input.
- The email body must be grounded in the fetched BBC items, not model memory.
- Include source links in the final email.
- Prioritize items when freshness_mode=strict_recent_only.
- If freshness_mode=latest_available_fallback, use those BBC items and clearly state they are older than
  freshness_window_minutes.
- If no BBC items are available at all, produce a transparent fallback email saying latest BBC updates
  could not be retrieved right now, and include no fabricated facts.
- After generating one body, immediately hand off that body to Email Manager.
- Send exactly one email total.
- Do not write the email body yourself; always use exactly one writer tool.
"""

rotation_manager = Agent(
    name="News Email Rotation Manager",
    instructions=rotation_manager_instructions,
    tools=[get_bbc_ukraine_news, polite_tool, news_only_tool],
    handoffs=[emailer_agent],
    model=MODEL,
)

STYLE_ORDER = ["polite", "news_only"]
STATE_FILE = Path(".email_style_state.json")


def get_style_index() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        data = json.loads(STATE_FILE.read_text())
        return int(data.get("next_style_index", 0)) % len(STYLE_ORDER)
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0


def save_style_index(index: int) -> None:
    STATE_FILE.write_text(json.dumps({"next_style_index": index % len(STYLE_ORDER)}))


async def main() -> None:
    style_index = get_style_index()
    selected_style = STYLE_ORDER[style_index]

    message = (
        f"Create and send exactly one {selected_style} email about the latest Ukraine news, "
        "using BBC reporting as primary context when available. "
        "For polite style include a brief greeting and polite sign-off; for news_only style output only a list of news items."
    )

    with trace("Ukraine News Email Rotation"):
        result = await Runner.run(rotation_manager, message)

    # Advance rotation only after a successful run.
    save_style_index(style_index + 1)

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
