# BBC Ukraine News Agent

Agent-based Python app that fetches BBC Ukraine headlines, generates a digest in alternating styles, and sends exactly one HTML email per run via SendGrid.

## Workflow
1. A rotation manager fetches BBC RSS headlines and filters for Ukraine-related items.
2. Style rotates between `polite` and `news_only` using `.email_style_state.json`.
3. A writer agent creates the email body from fetched BBC items only.
4. An email manager agent writes the subject, converts body to HTML, and sends one email.

Freshness behavior: the app prioritizes items newer than `NEWS_MAX_AGE_MINUTES` (default `180`). If none are fresh, it falls back to latest available BBC items and marks them as older.

## Requirements
- Python 3.10+
- `openai-agents`
- `sendgrid`
- `python-dotenv`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install openai-agents sendgrid python-dotenv
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
FROM_EMAIL=sender@example.com
TO_EMAIL=recipient@example.com
BBC_UKRAINE_FEED_URL=https://feeds.bbci.co.uk/news/world/europe/rss.xml
NEWS_MAX_AGE_MINUTES=180
```

## Run
```bash
python app.py
```

Each run sends one email and then advances the style rotation for the next run.
