# justin_bennett_trader.py
# requirements: streamlit, feedparser, youtube-transcript-api, openai, yt-dlp, requests, yfinance, fastapi (NOT needed here)
#
# Whisper-only mode + external bridge:
# - If WHISPER_BRIDGE_URL is set, app POSTs {url} to /transcribe and expects {"text": "..."}.
# - If not set, app uses local yt-dlp + optional cookies to fetch audio and OpenAI Whisper.
# - UI toggle "Whisper-only (skip captions and other STT)" defaults to ON.
#
# Other features preserved: menu navigation, maintenance reset, live prices, SQLite history, trade cards.

import os, re, json, sqlite3, tempfile, datetime as dt, threading, time
from typing import List, Dict, Optional, Tuple

import requests
import streamlit as st
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import yfinance as yf

_YTDLP = None

APP_NAME = "Justin Bennett — Video Trading Tool"
DEFAULT_CHANNEL_ID = "UCaWQprRy3TgktPvsyBLUNxw"

BASE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "jb_trader.db")

DB_LOCK = threading.Lock()
FEED_URL_TMPL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

DEFAULT_MODEL = os.getenv("JB_OPENAI_MODEL", "gpt-4o-mini")

def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def clamp_text(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + "\n\n[...truncated...]")

def pick_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if (v and v.strip()) else None

def ensure_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos(
      video_id TEXT PRIMARY KEY,
      channel_id TEXT,
      published TEXT,
      title TEXT,
      url TEXT,
      transcript TEXT,
      summary TEXT,
      idea_json TEXT,
      analyzed_at TEXT,
      model TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS snapshots(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT,
      channel_id TEXT,
      summary TEXT,
      idea_json TEXT,
      considered_video_ids TEXT
    );
    """)
    conn.commit()

@st.cache_resource(show_spinner=False)
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    ensure_db(conn)
    return conn

def feed_latest(channel_id: str, limit: int = 10) -> List[Dict]:
    url = FEED_URL_TMPL.format(channel_id=channel_id)
    d = feedparser.parse(url)
    items = []
    for e in d.entries[:limit]:
        vid = e.get("yt_videoid") or (e.get("id","").split(":")[-1])
        items.append({
            "video_id": vid,
            "title": (e.get("title") or "").strip(),
            "published": e.get("published",""),
            "url": e.get("link",""),
        })
    items.sort(key=lambda x: x.get("published",""), reverse=True)
    return items

def get_known_ids(conn, channel_id: str) -> set:
    cur = conn.cursor()
    cur.execute("SELECT video_id FROM videos WHERE channel_id=?", (channel_id,))
    return {r[0] for r in cur.fetchall()}

def save_video_row(conn, row: Dict):
    with DB_LOCK:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO videos(video_id, channel_id, published, title, url, transcript, summary, idea_json, analyzed_at, model)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(video_id) DO UPDATE SET
          published=excluded.published,
          title=excluded.title,
          url=excluded.url,
          transcript=COALESCE(excluded.transcript, videos.transcript),
          summary=COALESCE(excluded.summary, videos.summary),
          idea_json=COALESCE(excluded.idea_json, videos.idea_json),
          analyzed_at=COALESCE(excluded.analyzed_at, videos.analyzed_at),
          model=COALESCE(excluded.model, videos.model)
        """, (
            row.get("video_id"),
            row.get("channel_id"),
            row.get("published"),
            row.get("title"),
            row.get("url"),
            row.get("transcript"),
            row.get("summary"),
            row.get("idea_json"),
            row.get("analyzed_at"),
            row.get("model"),
        ))
        conn.commit()

def list_recent_analyses(conn, channel_id: str, limit: int = 20) -> List[Dict]:
    cur = conn.cursor()
    cur.execute("""
      SELECT video_id, published, title, url, summary, idea_json, analyzed_at
      FROM videos
      WHERE channel_id=?
      ORDER BY published DESC
      LIMIT ?
    """, (channel_id, limit))
    rows = cur.fetchall()
    out=[]
    for r in rows:
        out.append({
            "video_id": r[0], "published": r[1], "title": r[2], "url": r[3],
            "summary": r[4], "idea_json": r[5], "analyzed_at": r[6]
        })
    return out

def list_all_analyses(conn, channel_id: str) -> List[Dict]:
    cur = conn.cursor()
    cur.execute("""
      SELECT video_id, published, title, url, summary, idea_json, analyzed_at
      FROM videos
      WHERE channel_id=?
      ORDER BY published DESC
    """, (channel_id,))
    rows = cur.fetchall()
    out=[]
    for r in rows:
        out.append({
            "video_id": r[0], "published": r[1], "title": r[2], "url": r[3],
            "summary": r[4], "idea_json": r[5], "analyzed_at": r[6]
        })
    return out

def save_snapshot(conn, channel_id: str, summary: str, idea_json: str, considered_ids: List[str]):
    with DB_LOCK:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO snapshots(created_at, channel_id, summary, idea_json, considered_video_ids)
        VALUES(?,?,?,?,?)
        """, (now_utc(), channel_id, summary, idea_json, json.dumps(considered_ids)))
        conn.commit()

def list_snapshots(conn, channel_id: str, limit: int = 10) -> List[Dict]:
    cur = conn.cursor()
    cur.execute("""
      SELECT id, created_at, summary, idea_json, considered_video_ids
      FROM snapshots
      WHERE channel_id=?
      ORDER BY created_at DESC
      LIMIT ?
    """, (channel_id, limit))
    rows = cur.fetchall()
    out=[]
    for r in rows:
        out.append({
            "id": r[0], "created_at": r[1], "summary": r[2],
            "idea_json": r[3], "considered_video_ids": json.loads(r[4] or "[]")
        })
    return out

# ---------- Maintenance ----------
def wipe_db_tables():
    conn = get_db()
    with DB_LOCK:
        conn.execute("DELETE FROM videos;")
        conn.execute("DELETE FROM snapshots;")
        conn.commit()

def hard_reset_db_file():
    try:
        conn = get_db()
        conn.close()
    except Exception:
        pass
    st.cache_resource.clear()
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except Exception as e:
        st.warning(f"Could not delete DB file: {e}")

def nuke_everything():
    hard_reset_db_file()
    st.cache_data.clear()
    st.session_state.clear()
    st.success("All data and caches cleared. Rerunning…")
    st.rerun()

# ---------- Transcript (Whisper-only helpers) ----------
def _lazy_import_ytdlp():
    global _YTDLP
    if _YTDLP is None:
        import yt_dlp as _YTDLP
    return _YTDLP

def fetch_audio_and_transcribe_openai(video_url: str, client: "OpenAI", cookies_bytes: Optional[bytes] = None) -> Tuple[Optional[str], Optional[str]]:
    """Local yt-dlp download → OpenAI Whisper."""
    try:
        ydlp = _lazy_import_ytdlp()
        tmpdir = tempfile.mkdtemp(prefix="jb_audio_")
        outpath = os.path.join(tmpdir, "%(id)s.%(ext)s")

        cookiefile = None
        if cookies_bytes:
            cookiefile = os.path.join(tmpdir, "cookies.txt")
            with open(cookiefile, "wb") as cf:
                cf.write(cookies_bytes)

        ydl_opts = {
            'format': 'ba[ext=m4a]/bestaudio/best',
            'outtmpl': outpath,
            'quiet': True,
            'noprogress': True,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'force_ipv4': True,
            'retries': 20,
            'fragment_retries': 20,
            'concurrent_fragment_downloads': 1,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            },
        }
        if cookiefile:
            ydl_opts['cookiefile'] = cookiefile

        with ydlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filepath = ydl.prepare_filename(info)

        try:
            with open(filepath, "rb") as f:
                try:
                    r = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
                    text = r.text
                except Exception:
                    f.seek(0)
                    r = client.audio.transcriptions.create(model="whisper-1", file=f)
                    text = r.text
            text = re.sub(r"\s+", " ", (text or "")).strip()
            return (text, None) if text else (None, "OpenAI returned empty transcript")
        finally:
            try:
                for fn in os.listdir(tmpdir):
                    try: os.remove(os.path.join(tmpdir, fn))
                    except: pass
                os.rmdir(tmpdir)
            except: pass
    except Exception as e:
        return None, f"Whisper yt-dlp fallback failed: {e}"

def transcribe_via_bridge(video_url: str, bridge_url: str, timeout_s: int = 600) -> Tuple[Optional[str], Optional[str]]:
    """Call external Whisper Bridge: POST /transcribe {url} -> {text}."""
    try:
        url = bridge_url.rstrip("/") + "/transcribe"
        r = requests.post(url, json={"url": video_url}, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        txt = (data.get("text") or "").strip()
        if not txt:
            return None, data.get("error") or "Bridge returned empty text"
        return txt, None
    except Exception as e:
        return None, f"Bridge error: {e}"

# ---------- OpenAI ----------
def get_openai_client() -> Optional["OpenAI"]:
    key = pick_env("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=key)
        return client
    except Exception:
        return None

# ---------- Analysis prompts ----------
SYSTEM_PROMPT = """You are a trading analyst specialized in summarizing Justin Bennett (Daily Price Action) videos.
Return STRICT JSON with:
- short_summary (<=120 words)
- instruments [array]
- key_levels [{instrument, level, type, note}]
- directional_bias {instrument: bias}
- trade_ideas [ up to 3 items: {instrument, timeframe, bias, plan, entries[], stop_loss{level,rationale}, take_profits[], risk_notes[]} ]
- confidence (0..1)
- assumptions [array]
If numbers are not in transcript, use "no numeric levels stated". Do not invent prices.
"""

def analyze_one_video(client: "OpenAI", model_name: str, title: str, transcript: str, previous_context: List[Dict]) -> Tuple[str, str]:
    ctx_snips = []
    for row in previous_context[-8:]:
        c = f"- {row['title']}: {row.get('summary','')}"
        try:
            idea = json.loads(row.get("idea_json") or "{}")
            ideas = idea.get("trade_ideas") or ([idea.get("trade_idea")] if idea.get("trade_idea") else [])
            if ideas:
                for it in ideas[:2]:
                    if not it: continue
                    c += f"\n  {it.get('instrument','?')} • {it.get('bias','?')} • plan: {it.get('plan','')[:60]}"
        except Exception:
            pass
        ctx_snips.append(c)

    user_prompt = f"""
CONTEXT (previous):
{os.linesep.join(ctx_snips) if ctx_snips else "none"}

VIDEO TITLE:
{title}

TRANSCRIPT (truncated if long):
{clamp_text(transcript, 18000)}
"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        idea_json = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        idea_json = json.loads(m.group(0)) if m else None
    if not isinstance(idea_json, dict):
        raise RuntimeError("Model did not return valid JSON.")
    short_summary = idea_json.get("short_summary", "")
    return short_summary, json.dumps(idea_json, ensure_ascii=False)

def synthesize_overall(client: "OpenAI", model_name: str, recent_rows: List[Dict], new_rows: List[Dict]) -> Tuple[str, str]:
    ctx = []
    merged = (new_rows + [x for x in recent_rows if x not in new_rows])[:10]
    for r in merged:
        try:
            idea = json.loads(r.get("idea_json") or "{}")
        except Exception:
            idea = {}
        ctx.append({"title": r.get("title",""), "published": r.get("published",""), "summary": r.get("summary",""), "idea": idea})
    prompt = "Past & new analyses to consider:\
