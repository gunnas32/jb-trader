# apps/justin_bennett_trader.py
# requirements: streamlit, feedparser, youtube-transcript-api, openai, yt-dlp, requests, yfinance
#
# Justin Bennett / Daily Price Action video analyzer:
# - Pulls latest YouTube videos via RSS.
# - Gets transcript (captions first; Whisper fallback with yt-dlp + optional cookies).
# - Uses OpenAI to create a SHORT SUMMARY + STRUCTURED TRADE IDEA (strict JSON).
# - Stores everything in SQLite (remembers what’s analyzed; only processes new).
# - Synthesizes an updated overall plan using recent context.
# - Live prices per instrument, crisp trade cards with Entry / SL / TPs.
#
# ⚠ INFO/EDU ONLY — NOT FINANCIAL ADVICE.
import os, re, json, sqlite3, tempfile, datetime as dt, threading
from typing import List, Dict, Optional, Tuple

import streamlit as st
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# live prices
import yfinance as yf

_YTDLP = None

APP_NAME = "Justin Bennett — Video Trading Tool"
DEFAULT_CHANNEL_ID = "UCaWQprRy3TgktPvsyBLUNxw"   # Daily Price Action (@JustinBennettfx)

# --------- Paths / DB ---------
BASE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "jb_trader.db")

DB_LOCK = threading.Lock()
FEED_URL_TMPL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

# Default analysis model (can be overridden in sidebar)
DEFAULT_MODEL = os.getenv("JB_OPENAI_MODEL", "gpt-4o-mini")

# --------- Utilities ---------
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
    # Allow use across Streamlit threads + better concurrency with WAL
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

# ---------- Transcript ----------
def fetch_transcript_youtube(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        caps = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','en-US','en-GB'])
        text = " ".join([c.get("text","") for c in caps])
        text = re.sub(r"\s+", " ", text).strip()
        return (text, None) if text else (None, "Empty transcript")
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        return None, f"No captions: {e}"
    except Exception as e:
        return None, f"Transcript error: {e}"

def _lazy_import_ytdlp():
    global _YTDLP
    if _YTDLP is None:
        import yt_dlp as _YTDLP  # lazy import
    return _YTDLP

def fetch_audio_and_transcribe_openai(video_url: str, client: "OpenAI", cookies_bytes: Optional[bytes] = None) -> Tuple[Optional[str], Optional[str]]:
    """Download audio with yt-dlp → transcribe via OpenAI (4o-mini-transcribe or whisper-1)."""
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
            'format': 'bestaudio/best',
            'outtmpl': outpath,
            'quiet': True,
            'noprogress': True,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            # try different player client to dodge 403/age/region issues
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
        return None, f"Fallback transcription failed: {e}"

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

# Updated schema: prefer trade_ideas (array). Backwards compatible with trade_idea (single).
SYSTEM_PROMPT = """You are a trading analyst specialized in summarizing Justin Bennett (Daily Price Action) videos.
Extract price-action ideas without hype. Prefer clarity over verbosity.

Return STRICT JSON with these fields:
- short_summary: one concise paragraph (<= 120 words).
- instruments: array of symbols he discussed (e.g., ["DXY","EURUSD","GBPUSD","USDJPY","XAUUSD"]).
- key_levels: array of {instrument, level, type, note} where type ∈ {"support","resistance","trendline","range","fibo","other"}.
- directional_bias: object mapping instrument -> {"bullish","bearish","neutral"} (only for instruments he mentioned).
- trade_ideas: array of objects (1–3 max). Each item:
  {
    instrument: "EURUSD",
    timeframe: "Daily",
    bias: "bullish" | "bearish" | "neutral",
    plan: "one-line plan plain English",
    entries: [
      { type: "breakout" | "retest" | "limit", level: "1.0800", condition: "daily close above then retest as support" }
    ],
    stop_loss: { level: "1.0740", rationale: "below swing low / invalidates structure" },
    take_profits: [
      { level: "1.0890", label: "TP1" },
      { level: "1.1000", label: "TP2" }
    ],
    risk_notes: ["trade what you see", "no numeric levels stated" if missing]
  }

If the transcript lacks numbers, put "no numeric levels stated" in the relevant fields (do NOT invent prices).
Also include a top-level `confidence` number 0..1 and `assumptions` (short bullet list).
If you only find one valid opportunity, return a single-element trade_ideas array.
"""

def analyze_one_video(client: "OpenAI", model_name: str, title: str, transcript: str, previous_context: List[Dict]) -> Tuple[str, str]:
    """Returns (short_summary, idea_json_str)."""
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
CONTEXT (previous videos, recent first):
{os.linesep.join(ctx_snips) if ctx_snips else "none"}

VIDEO TITLE:
{title}

TRANSCRIPT (truncated if long):
{clamp_text(transcript, 18000)}
"""

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user_prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    # Parse strict/fenced JSON
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
    """Combine last ~10 analyses to produce an updated plan (same JSON schema)."""
    ctx = []
    merged = (new_rows + [x for x in recent_rows if x not in new_rows])[:10]
    for r in merged:
        try:
            idea = json.loads(r.get("idea_json") or "{}")
        except Exception:
            idea = {}
        ctx.append({
            "title": r.get("title",""),
            "published": r.get("published",""),
            "summary": r.get("summary",""),
            "idea": idea
        })
    prompt = "Past & new analyses to consider:\n" + json.dumps(ctx, ensure_ascii=False) + \
             "\n\nSynthesize an UPDATED plan that reconciles conflicts. Return strict JSON using the same schema (with `trade_ideas`)."

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        data = json.loads(m.group(0)) if m else None
    if not isinstance(data, dict):
        raise RuntimeError("Synthesis returned invalid JSON.")

    return data.get("short_summary",""), json.dumps(data, ensure_ascii=False)

# ---------- Live prices ----------
def _candidates_for_instrument(inst: str) -> List[str]:
    inst = (inst or "").upper().replace("/", "")
    cands = []
    # FX like EURUSD
    if re.fullmatch(r"[A-Z]{3}[A-Z]{3}", inst):
        cands.append(f"{inst}=X")
    # Crypto
    if inst.endswith("USD") and len(inst) in (6,7):
        cands.append(f"{inst[:-3]}-USD")
    # DXY
    if inst in ("DXY", "USDIDX", "USDX"):
        cands.extend(["^DXY", "DX-Y.NYB"])
    # Gold spot
    if inst in ("XAUUSD","GOLD","XAU"):
        cands.extend(["XAUUSD=X", "GC=F"])
    # S&P 500
    if inst in ("SPX","SPY","S&P500","GSPC"):
        cands.extend(["^GSPC","SPY"])
    # Fallback
    cands.append(inst)
    return list(dict.fromkeys(cands))

@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(inst: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (price, ticker_used) or (None,None)."""
    for t in _candidates_for_instrument(inst):
        try:
            ti = yf.Ticker(t)
            fi = getattr(ti, "fast_info", None)
            if fi and getattr(fi, "last_price", None):
                return float(fi.last_price), t
            hist = ti.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]), t
            hist = ti.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]), t
        except Exception:
            continue
    return None, None

def _normalize_trade_ideas(idea: Dict) -> List[Dict]:
    if not isinstance(idea, dict): return []
    if isinstance(idea.get("trade_ideas"), list):
        return [x for x in idea["trade_ideas"] if isinstance(x, dict)]
    ti = idea.get("trade_idea")
    return [ti] if isinstance(ti, dict) else []

def _pretty_levels(level) -> str:
    if level is None: return "—"
    if isinstance(level, (int, float)):
        return f"{level:.5f}".rstrip("0").rstrip(".")
    return str(level)

def render_trade_cards(idea: Dict, title: str = "", where: str = "Video"):
    trades = _normalize_trade_ideas(idea)
    if not trades:
        st.info("No structured trade ideas found in this item.")
        return

    for i, tr in enumerate(trades, start=1):
        instrument = (tr.get("instrument") or "").upper()
        bias = tr.get("bias") or idea.get("directional_bias", {}).get(instrument, "")
        plan = tr.get("plan") or ""
        timeframe = tr.get("timeframe") or idea.get("timeframe", "")
        entries = tr.get("entries") or []
        sl = tr.get("stop_loss") or {}
        tps = tr.get("take_profits") or []

        price, used = (None, None)
        if instrument:
            price, used = get_live_price(instrument)

        box = st.container(border=True)
        with box:
            hdr = f"**{where} Trade {i} — {instrument}**"
            if price is not None:
                hdr += f"  |  Live: `{_pretty_levels(price)}` ({used})"
            st.markdown(hdr)
            cols = st.columns([1.2, 2.2, 1.2])
            with cols[0]:
                st.markdown(f"**Bias:** {bias or '—'}")
                st.markdown(f"**TF:** {timeframe or '—'}")
            with cols[1]:
                st.markdown(f"**Plan:** {plan or '—'}")
                if entries:
                    st.markdown("**Entries:**")
                    for e in entries:
                        st.markdown(f"- {e.get('type','entry')}: `{_pretty_levels(e.get('level'))}` — {e.get('condition','')}")
            with cols[2]:
                st.markdown("**Stop Loss:**")
                if sl:
                    st.markdown(f"- SL: `{_pretty_levels(sl.get('level'))}`")
                    if sl.get("rationale"): st.caption(sl["rationale"])
                else:
                    st.markdown("—")
                st.markdown("**Take Profits:**")
                if tps:
                    for t in tps:
                        lbl = t.get("label","TP")
                        st.markdown(f"- {lbl}: `{_pretty_levels(t.get('level'))}`")
                else:
                    st.markdown("—")
        st.caption("Risk: position sizing, slippage, news risk. Not financial advice.")
        st.divider()

# ---------- Analysis orchestrators ----------
def analyze_workflow(channel_id: str, model_name: str, fallback_enabled: bool = True, cookies_bytes: Optional[bytes] = None, first_run_take: int = 3):
    conn = get_db()
    client = get_openai_client()
    if not client:
        st.error("OpenAI client not available. Set OPENAI_API_KEY via environment or .streamlit/secrets.toml")
        return

    st.write("### 1) Checking channel feed…")
    feed = feed_latest(channel_id, limit=12)
    if not feed:
        st.warning("No items found on the channel feed.")
        return

    known = get_known_ids(conn, channel_id)
    unseen = [v for v in feed if v["video_id"] not in known]

    if len(known) == 0:
        targets = feed[:first_run_take]
        st.info(f"First run: analyzing the last {len(targets)} videos.")
    else:
        targets = unseen
        if targets:
            st.info(f"Found {len(targets)} new video(s). Analyzing…")
        else:
            st.success("No new videos. Will still synthesize an updated plan from recent context.")
    st.divider()

    recent_rows = list_recent_analyses(conn, channel_id, limit=10)
    analyzed_rows = []

    for i, v in enumerate(targets, start=1):
        st.write(f"#### Video {i}/{len(targets)} — {v['title']}")
        with st.status("Transcribing…", expanded=False) as status:
            text, err = fetch_transcript_youtube(v["video_id"])
            if not text and fallback_enabled:
                status.update(label="Captions not available. Trying Whisper fallback…", state="running")
                text2, err2 = fetch_audio_and_transcribe_openai(v["url"], client, cookies_bytes=cookies_bytes)
                text, err = text2, err2

            if not text:
                reason = f"({err})" if err else "(no transcript)"
                if not fallback_enabled:
                    status.update(label=f"No captions; Whisper fallback disabled. Skipping. {reason}", state="error")
                else:
                    status.update(label=f"❌ Couldn’t get transcript {reason}. Skipping.", state="error")
                save_video_row(conn, {
                    "video_id": v["video_id"], "channel_id": channel_id, "published": v["published"],
                    "title": v["title"], "url": v["url"], "transcript": None, "summary": None,
                    "idea_json": None, "analyzed_at": now_utc(), "model": model_name
                })
                st.divider()
                continue

            status.update(label="Analyzing with GPT…", state="running")
            try:
                summary, idea_json = analyze_one_video(client, model_name, v["title"], text, previous_context=recent_rows + analyzed_rows)
                save_video_row(conn, {
                    "video_id": v["video_id"], "channel_id": channel_id, "published": v["published"],
                    "title": v["title"], "url": v["url"], "transcript": text,
                    "summary": summary, "idea_json": idea_json,
                    "analyzed_at": now_utc(), "model": model_name
                })
                status.update(label="✅ Done", state="complete")
                st.success(summary)

                # Render trade cards + raw JSON
                try:
                    idea = json.loads(idea_json)
                    render_trade_cards(idea, title=v["title"], where="Video")
                    with st.expander("Structured trade idea (JSON)", expanded=False):
                        st.code(json.dumps(idea, indent=2))
                except Exception:
                    st.code(idea_json)

                analyzed_rows.append({
                    "video_id": v["video_id"], "published": v["published"], "title": v["title"], "url": v["url"],
                    "summary": summary, "idea_json": idea_json
                })
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(str(e))
        st.divider()

    # Synthesis
    st.write("### 2) Synthesizing overall trade idea…")
    recent_rows2 = list_recent_analyses(conn, channel_id, limit=10)
    try:
        summary, idea_json = synthesize_overall(client, model_name, recent_rows2, analyzed_rows)
        save_snapshot(conn, channel_id, summary, idea_json, [x["video_id"] for x in analyzed_rows] or [x["video_id"] for x in recent_rows2])
        st.success(summary)
        with st.expander("Overall plan (JSON)", expanded=True):
            try:
                idea = json.loads(idea_json)
                render_trade_cards(idea, title="Overall", where="Overall")
                st.code(json.dumps(idea, indent=2))
            except Exception:
                st.code(idea_json)
    except Exception as e:
        st.error(f"Couldn’t synthesize overall idea: {e}")

# ---------- UI ----------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
    st.title("📈 Justin Bennett Video Trading Tool")
    st.caption("Auto-summarize videos and produce structured trade ideas. Stores history and updates ideas on next run.")
    with st.expander("Disclaimer", expanded=False):
        st.write("This app is for **informational and educational purposes only** and is **not financial advice**. Markets involve risk.")

    # Sidebar
    st.sidebar.header("Settings")
    channel_id = st.sidebar.text_input("YouTube Channel ID", value=DEFAULT_CHANNEL_ID, help="Default: Daily Price Action (@JustinBennettfx)")

    # Model stored in session_state (no globals)
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = DEFAULT_MODEL
    current_model = st.sidebar.text_input("OpenAI model", value=st.session_state["current_model"], help="e.g., gpt-4o-mini")
    st.session_state["current_model"] = current_model

    # Transcription options
    st.sidebar.subheader("Transcription options")
    fallback_enabled = st.sidebar.checkbox(
        "Try Whisper fallback if captions missing",
        value=True,
        help="Downloads audio via yt-dlp and transcribes with OpenAI. If you see 403, add cookies below."
    )

    cookies_bytes = None
    cookies_secret = st.secrets.get("YTDLP_COOKIES", None)
    if cookies_secret:
        cookies_bytes = cookies_secret.encode("utf-8")

    cookie_file = st.sidebar.file_uploader(
        "YouTube cookies.txt (optional)",
        type=["txt"],
        help="Export with a browser extension (Netscape format). Helps bypass 403/age/region limits."
    )
    if cookie_file:
        cookies_bytes = cookie_file.read()

    api_present = bool(pick_env("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None))
    st.sidebar.write(f"OpenAI Key: {'✅ set' if api_present else '❌ missing'}")
    st.sidebar.write(f"DB file: `{DB_PATH}`")

    conn = get_db()

    # Latest snapshots
    st.subheader("Latest Overall Snapshots")
    snaps = list_snapshots(conn, channel_id, limit=5)
    if not snaps:
        st.info("No snapshots yet.")
    else:
        for sn in snaps:
            st.markdown(f"**{sn['created_at']}** — considered {len(sn['considered_video_ids'])} video(s)")
            st.write(sn["summary"])
            try:
                idea = json.loads(sn["idea_json"])
                render_trade_cards(idea, title="Snapshot", where="Snapshot")
            except Exception:
                pass
            with st.expander("Snapshot JSON", expanded=False):
                try:
                    st.code(json.dumps(json.loads(sn["idea_json"]), indent=2))
                except Exception:
                    st.code(sn["idea_json"])
            st.divider()

    # Feed preview
    st.subheader("Latest Channel Videos")
    try:
        feed = feed_latest(channel_id, limit=8)
        known = get_known_ids(conn, channel_id)
        for v in feed:
            mark = "🟢 NEW" if v["video_id"] not in known else "⚪ Analyzed"
            st.write(f"{mark} — **{v['title']}**  \nPublished: {v['published']}  \n{v['url']}")
    except Exception as e:
        st.warning(f"Couldn’t read feed: {e}")

    st.divider()
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("▶ Run analysis now", type="primary"):
            analyze_workflow(channel_id, current_model, fallback_enabled, cookies_bytes)
    with col2:
        if st.button("♻ Rebuild overall idea (no new videos)"):
            client = get_openai_client()
            if not client:
                st.error("OpenAI client not available.")
            else:
                rows = list_recent_analyses(conn, channel_id, limit=10)
                if not rows:
                    st.info("No analyzed videos yet.")
                else:
                    try:
                        summary, idea_json = synthesize_overall(client, current_model, rows, [])
                        save_snapshot(conn, channel_id, summary, idea_json, [x["video_id"] for x in rows])
                        st.success(summary)
                        idea = json.loads(idea_json)
                        render_trade_cards(idea, title="Overall", where="Overall")
                        with st.expander("Overall plan (JSON)", expanded=True):
                            st.code(json.dumps(idea, indent=2))
                    except Exception as e:
                        st.error(str(e))

if __name__ == "__main__":
    main()
