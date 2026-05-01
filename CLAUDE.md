# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a two-component tool for automatically monitoring and analyzing YouTube trading videos from Justin Bennett (Daily Price Action channel). It:

1. Fetches the latest videos from the channel's RSS feed
2. Downloads audio via `yt-dlp` and transcribes via OpenAI Whisper
3. Runs GPT analysis to extract structured trade setups (entries, stop-loss, take-profits, key levels, bias)
4. Displays results as a Streamlit app with live Plotly candlestick charts powered by `yfinance`
5. Persists all data in a local SQLite database

## Components

### `justin_bennett_trader.py` — Main Streamlit app

Three-tab UI: **Dashboard** (latest snapshot + recent videos), **Analyze** (trigger analysis), **History** (all analyzed videos).

- `main()` sets up the page, sidebar controls, and global timeframe radio (D / 4H / 1H)
- `analyze_workflow()` is the top-level orchestrator: feed → transcribe → analyze → snapshot synthesis → render
- `transcribe_auto_whisper()` implements a 3-step fallback: (1) local yt-dlp, (2) relay server, (3) local yt-dlp with cookies
- `analyze_one_video()` sends transcript + rolling context of previous 8 analyses to GPT and parses the JSON response
- `synthesize_overall()` merges new + recent analyses into a single updated trading plan
- `get_chart_data()` resolves instrument aliases → yfinance tickers, handles D/4H/1H timeframes; 4H is fetched as 1H then resampled
- `_candidates_for_instrument()` maps instrument names (e.g. `GOLD`, `USOIL`, `DXY`) to yfinance ticker candidates

### `audio_relay.py` — FastAPI relay server

A lightweight server meant to run separately (e.g., on a VPS or Render) for cases where Streamlit Cloud's IP is blocked by YouTube. Receives a URL + optional cookies via POST `/fetch`, downloads audio with yt-dlp, and returns raw audio bytes.

## Running the App

```bash
# Install main app dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run justin_bennett_trader.py
```

The devcontainer automatically runs the app on attach at port 8501.

```bash
# Install relay server dependencies
pip install -r "relay requirements.txt"

# Run the relay server
uvicorn audio_relay:app --host 0.0.0.0 --port $PORT
```

## Required Environment Variables / Secrets

The app reads secrets from Streamlit's secrets mechanism (`st.secrets`) with `os.getenv` as fallback:

| Key | Required | Purpose |
|-----|----------|---------|
| `OPENAI_API_KEY` | Yes | Whisper transcription + GPT analysis |
| `AUDIO_RELAY_URL` | No | URL of the relay server (e.g. `https://my-relay.onrender.com`) |
| `YTDLP_COOKIES` | No | Netscape-format cookies string for age-restricted/auth-gated videos |
| `JB_OPENAI_MODEL` | No | GPT model override (default: `gpt-4o-mini`) |

## Data Storage

SQLite database is created at `./data/jb_trader.db` with two tables:

- `videos` — one row per analyzed video; upserted on re-analysis, never overwrites existing `transcript`/`summary`/`idea_json` with NULL
- `snapshots` — synthesis snapshots produced after each `analyze_workflow()` run

The `get_db()` connection uses WAL mode and is cached via `@st.cache_resource`.

## GPT Analysis JSON Schema

`analyze_one_video()` and `synthesize_overall()` both use `SYSTEM_PROMPT`, which demands strict JSON with these top-level keys: `short_summary`, `instruments`, `key_levels`, `directional_bias`, `trade_ideas`, `confidence`, `assumptions`. When adding features that consume this JSON, parse with `json.loads` and fall back to regex extraction (`re.search(r"\{.*\}", content, re.S)`), matching the existing error-tolerant pattern.

## Transcription Fallback Chain

When adding or modifying transcription logic, preserve the fallback order in `transcribe_auto_whisper()`:
1. Local yt-dlp without cookies
2. Relay (`AUDIO_RELAY_URL`) with optional `YTDLP_COOKIES`
3. Local yt-dlp with cookies

Both the relay and local paths share the same yt-dlp options (geo_bypass, android+web player clients, Chrome User-Agent). Keep them in sync if changing download behavior.
