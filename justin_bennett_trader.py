# justin_bennett_trader.py
# Minimal, pro UI — Whisper-only transcription, fully automatic.
# Features:
#   • Pulls Justin Bennett channel feed (YouTube RSS)
#   • Transcribes ONLY with OpenAI Whisper
#   • Robust auto-flow (no prompts):
#        1) yt-dlp direct
#        2) audio relay (if AUDIO_RELAY_URL is set in secrets)
#        3) yt-dlp with cookies from secrets (YTDLP_COOKIES)
#   • Analyses via OpenAI into structured trade ideas
#   • Live interactive candlestick chart beside each idea/snapshot
#     - Zoom, pan, and global timeframe (sidebar)
#   • History view
#   • One-click “Clear ALL Data” (DB + caches + session)

import os, re, json, sqlite3, tempfile, datetime as dt, threading, time
from typing import List, Dict, Optional, Tuple

import streamlit as st
import feedparser
from openai import OpenAI
import yfinance as yf
import plotly.graph_objects as go
import requests

# ----------------- App basics -----------------
APP_NAME = "Justin Bennett — Whisper Analyzer"
DEFAULT_CHANNEL_ID = "UCaWQprRy3TgktPvsyBLUNxw"   # Daily Price Action
DEFAULT_MODEL = os.getenv("JB_OPENAI_MODEL", "gpt-4o-mini")  # for analysis (not transcription)

BASE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "jb_trader.db")

DB_LOCK = threading.Lock()
FEED_URL_TMPL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

# ----------------- Utilities -----------------
def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def clamp_text(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + "\n\n[...truncated...]")

def get_openai_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

# ----------------- DB -----------------
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

def list_snapshots(conn, channel_id: str, limit: int = 3) -> List[Dict]:
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

def nuke_everything():
    try:
        conn = get_db()
        conn.close()
    except: pass
    st.cache_resource.clear()
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except Exception as e:
        st.warning(f"Could not delete DB file: {e}")
    st.cache_data.clear()
    st.session_state.clear()
    st.success("All data, caches, and session cleared. Rerunning…")
    st.rerun()

# ----------------- Feed -----------------
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

# ----------------- Whisper-only transcription (auto) -----------------
_YTDLP = None
def _lazy_import_ytdlp():
    global _YTDLP
    if _YTDLP is None:
        import yt_dlp as _YTDLP  # lazy import
    return _YTDLP

def _transcribe_local(video_url: str, client: OpenAI, cookies_bytes: Optional[bytes]) -> Tuple[Optional[str], Optional[str]]:
    """Try yt-dlp locally (Streamlit Cloud machine), optionally with cookies."""
    try:
        ydlp = _lazy_import_ytdlp()
        tmpdir = tempfile.mkdtemp(prefix="jb_audio_")
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        cookiefile = None
        if cookies_bytes:
            cookiefile = os.path.join(tmpdir, "cookies.txt")
            with open(cookiefile, "wb") as cf:
                cf.write(cookies_bytes)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noprogress": True,
            "nocheckcertificate": True,
            "geo_bypass": True,
            "geo_bypass_country": "US",
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
        }
        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile
        with ydlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filepath = ydl.prepare_filename(info)
        try:
            with open(filepath, "rb") as f:
                r = client.audio.transcriptions.create(model="whisper-1", file=f)
                text = (r.text or "").strip()
            return (text if text else None, "empty whisper text" if not text else None)
        finally:
            try:
                for fn in os.listdir(tmpdir):
                    try: os.remove(os.path.join(tmpdir, fn))
                    except: pass
                os.rmdir(tmpdir)
            except: pass
    except Exception as e:
        return None, f"local yt-dlp+whisper failed: {e}"

def _transcribe_via_relay(video_url: str, client: OpenAI, relay_url: str, cookies_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Ask external relay to fetch audio, then send bytes to whisper."""
    try:
        url = relay_url.rstrip("/") + "/fetch"
        payload = {"url": video_url}
        if cookies_text:
            payload["cookies"] = cookies_text
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code != 200:
            return None, f"relay HTTP {r.status_code}: {r.text[:200]}"
        data = r.content
        # temp file -> whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(data)
            tmp.flush()
            p = tmp.name
        try:
            with open(p, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = (resp.text or "").strip()
            return (text if text else None, "relay whisper empty" if not text else None)
        finally:
            try: os.remove(p)
            except: pass
    except Exception as e:
        return None, f"relay failed: {e}"

def transcribe_auto_whisper(video_url: str, client: OpenAI) -> Tuple[Optional[str], Optional[str]]:
    """
    Automatic Whisper flow (no UI):
      1) Local yt-dlp without cookies
      2) Relay (if AUDIO_RELAY_URL provided)
      3) Local yt-dlp with cookies from secrets (YTDLP_COOKIES)
    """
    # 1) local without cookies
    text, err = _transcribe_local(video_url, client, cookies_bytes=None)
    if text: return text, None

    # 2) relay
    relay_url = st.secrets.get("AUDIO_RELAY_URL", os.getenv("AUDIO_RELAY_URL", "")).strip()
    cookies_secret = st.secrets.get("YTDLP_COOKIES", None)
    cookies_text = cookies_secret if (cookies_secret and cookies_secret.strip()) else None
    if relay_url:
        text2, err2 = _transcribe_via_relay(video_url, client, relay_url, cookies_text)
        if text2: return text2, None
        err = f"{err} | {err2}"

    # 3) local with cookies (if any)
    if cookies_text:
        text3, err3 = _transcribe_local(video_url, client, cookies_bytes=cookies_text.encode("utf-8"))
        if text3: return text3, None
        err = f"{err} | {err3}"

    return None, err or "transcription failed"

# ----------------- Analysis prompt -----------------
SYSTEM_PROMPT = """You are a trading analyst summarizing Justin Bennett (Daily Price Action) videos.
Return STRICT JSON:
- short_summary: <=120 words
- instruments: ["DXY","EURUSD","GBPUSD","USDJPY","XAUUSD", ...]
- key_levels: [{instrument, level, type, note}], type ∈ {"support","resistance","trendline","range","fibo","other"}
- directional_bias: {instrument: "bullish"|"bearish"|"neutral"}
- trade_ideas: 1–3 items, each:
  {
    instrument, timeframe, bias, plan,
    entries: [{type:"breakout"|"retest"|"limit", level, condition}],
    stop_loss: {level, rationale},
    take_profits: [{level, label}],
    risk_notes: [...]
  }
- confidence: 0..1
- assumptions: [..]
Do NOT invent numeric levels if absent; write "no numeric levels stated".
"""

def analyze_one_video(client: OpenAI, model_name: str, title: str, transcript: str, previous_context: List[Dict]) -> Tuple[str, str]:
    ctx_snips = []
    for row in previous_context[-8:]:
        c = f"- {row['title']}: {row.get('summary','')}"
        try:
            idea = json.loads(row.get("idea_json") or "{}")
            ideas = idea.get("trade_ideas") or ([idea.get("trade_idea")] if idea.get("trade_idea") else [])
            for it in (ideas or [])[:2]:
                if not it: continue
                c += f"\n  {it.get('instrument','?')} • {it.get('bias','?')} • plan: {it.get('plan','')[:60]}"
        except Exception:
            pass
        ctx_snips.append(c)
    user_prompt = f"CONTEXT:\n{os.linesep.join(ctx_snips) if ctx_snips else 'none'}\n\nTITLE:\n{title}\n\nTRANSCRIPT:\n{clamp_text(transcript, 18000)}"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user_prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        idea_json = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        idea_json = json.loads(m.group(0)) if m else {}
    short_summary = idea_json.get("short_summary", "")
    return short_summary, json.dumps(idea_json, ensure_ascii=False)

# ----------------- Prices & Charts -----------------
def _candidates_for_instrument(inst: str) -> List[str]:
    inst = (inst or "").upper().replace("/", "")
    # common aliases
    alias = {
        "GOLD":"XAUUSD", "XAU":"XAUUSD",
        "SILVER":"XAGUSD", "XAG":"XAGUSD",
        "SPX":"^GSPC", "S&P500":"^GSPC",
        "USOIL":"CL=F", "WTI":"CL=F", "BRENT":"BZ=F", "XTIUSD":"CL=F",
        "DOLLARINDEX":"^DXY", "USDINDEX":"^DXY"
    }
    inst = alias.get(inst, inst)
    cands = []
    if re.fullmatch(r"[A-Z]{3}[A-Z]{3}", inst):  # FX like EURUSD
        cands.append(f"{inst}=X")
    if inst.endswith("USD") and len(inst) in (6,7):  # Crypto like BTCUSD
        cands.append(f"{inst[:-3]}-USD")
    if inst in ("DXY", "^DXY", "USDIDX", "USDX"):
        cands.extend(["^DXY", "DX-Y.NYB"])
    if inst in ("XAUUSD","GC=F"):
        cands.extend(["XAUUSD=X", "GC=F"])
    if inst in ("XAGUSD","SI=F"):
        cands.extend(["XAGUSD=X", "SI=F"])
    if inst in ("CL=F", "BZ=F"):
        cands.append(inst)
    if inst in ("^GSPC","SPY"):
        cands.extend(["^GSPC","SPY"])
    cands.append(inst)  # fallback
    # dedupe
    return list(dict.fromkeys(cands))

# global timeframe/interval mapping
TIMEFRAME_MAP = {
    "1D":  ("1d",  "5m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "30m"),
    "3M":  ("3mo", "60m"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
}

@st.cache_data(ttl=90, show_spinner=False)
def get_chart_data(inst: str, period: str, interval: str):
    """Returns (df, used_ticker)."""
    for t in _candidates_for_instrument(inst):
        try:
            ti = yf.Ticker(t)
            df = ti.history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df.dropna(), t
        except Exception:
            continue
    return None, None

def plot_live_chart(instrument: str, period: str, interval: str):
    """Interactive candlestick (zoom/pan) with plotly."""
    if not instrument:
        st.info("No instrument specified.")
        return
    df, used = get_chart_data(instrument, period, interval)
    if df is None or df.empty:
        st.info("Chart data unavailable for this instrument.")
        return
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#06c167", decreasing_line_color="#d94f4f", name=used or instrument
        )
    ])
    fig.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=380,
        xaxis=dict(title=None, rangeslider=dict(visible=False)),
        yaxis=dict(title=None),
        showlegend=False,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Render helpers -----------------
def _normalize_trade_ideas(idea: Dict) -> List[Dict]:
    if not isinstance(idea, dict): return []
    if isinstance(idea.get("trade_ideas"), list):
        return [x for x in idea["trade_ideas"] if isinstance(x, dict)]
    ti = idea.get("trade_idea")
    return [ti] if isinstance(ti, dict) else []

def render_trade_cards_with_charts(idea: Dict, where: str, chart_period: str, chart_interval: str):
    trades = _normalize_trade_ideas(idea)
    if not trades:
        st.info("No structured trade ideas found.")
        return
    for i, tr in enumerate(trades, 1):
        instrument = (tr.get("instrument") or "").upper()
        colL, colR = st.columns([1.2, 2.0], vertical_alignment="top")
        with colL:
            with st.container(border=True):
                st.markdown(f"**{where} Trade {i} — {instrument or '—'}**")
                st.markdown(f"**Bias:** {tr.get('bias','—')} • **TF:** {tr.get('timeframe','—')}")
                st.write(tr.get("plan",""))
        with colR:
            plot_live_chart(instrument, chart_period, chart_interval)
        st.divider()

# ----------------- Workflow -----------------
def analyze_workflow(channel_id: str, model_name: str, chart_period: str, chart_interval: str, first_run_take: int = 3):
    conn = get_db()
    client = get_openai_client()
    if not client:
        st.error("OpenAI key missing.")
        return

    feed = feed_latest(channel_id, limit=12)
    if not feed:
        st.warning("No items found on the channel feed.")
        return

    # First run: take last N; otherwise only unseen
    known_ids = {r["video_id"] for r in list_all_analyses(conn, channel_id)}
    targets = feed[:first_run_take] if not known_ids else [v for v in feed if v["video_id"] not in known_ids]

    recent_rows = list_recent_analyses(conn, channel_id, limit=10)
    analyzed_rows = []

    for v in targets:
        with st.status(f"Transcribing: {v['title']}", expanded=False) as status:
            text, err = transcribe_auto_whisper(v["url"], client)
            if not text:
                status.update(label="Transcription failed", state="error")
                st.error(f"{v['title']} — {err}")
                # Skip silently; we continue with others
                continue

            status.update(label="Analyzing…", state="running")
            try:
                summary, idea_json = analyze_one_video(client, model_name, v["title"], text, previous_context=recent_rows + analyzed_rows)
                save_video_row(conn, {
                    "video_id": v["video_id"], "channel_id": channel_id, "published": v["published"],
                    "title": v["title"], "url": v["url"], "transcript": text,
                    "summary": summary, "idea_json": idea_json,
                    "analyzed_at": now_utc(), "model": model_name
                })
                status.update(label="Done", state="complete")
                try:
                    idea = json.loads(idea_json)
                    render_trade_cards_with_charts(idea, where="Video", chart_period=chart_period, chart_interval=chart_interval)
                except Exception:
                    st.code(idea_json)
                analyzed_rows.append({"video_id": v["video_id"], "published": v["published"], "title": v["title"],
                                      "url": v["url"], "summary": summary, "idea_json": idea_json})
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(str(e))

    # Snapshot synthesis
    rows_for_snapshot = analyzed_rows or recent_rows
    if rows_for_snapshot:
        try:
            summary, idea_json = synthesize_overall(get_openai_client(), model_name, recent_rows, analyzed_rows)
            save_snapshot(conn, channel_id, summary, idea_json, [x["video_id"] for x in analyzed_rows] or [x["video_id"] for x in recent_rows])
            st.subheader("Overall Snapshot")
            st.success(summary)
            try:
                idea = json.loads(idea_json)
                render_trade_cards_with_charts(idea, where="Overall", chart_period=chart_period, chart_interval=chart_interval)
            except Exception:
                st.code(idea_json)
        except Exception as e:
            st.warning(f"Couldn’t synthesize overall idea: {e}")

# ----------------- Synthesis helper -----------------
def synthesize_overall(client: OpenAI, model_name: str, recent_rows: List[Dict], new_rows: List[Dict]) -> Tuple[str, str]:
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
             "\n\nSynthesize an UPDATED plan (same JSON schema with `trade_ideas`)."
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
        data = json.loads(m.group(0)) if m else {}
    return data.get("short_summary",""), json.dumps(data, ensure_ascii=False)

# ----------------- Pages -----------------
def page_dashboard(channel_id: str, chart_period: str, chart_interval: str):
    conn = get_db()
    snaps = list_snapshots(conn, channel_id, limit=1)
    st.subheader("Latest Snapshot")
    if not snaps:
        st.info("No snapshots yet. Go to Analyze to produce the first one.")
    else:
        sn = snaps[0]
        st.caption(f"Snapshot time: {sn['created_at']}")
        st.write(sn["summary"])
        try:
            idea = json.loads(sn["idea_json"])
            render_trade_cards_with_charts(idea, where="Snapshot", chart_period=chart_period, chart_interval=chart_interval)
        except Exception:
            st.code(sn["idea_json"])

    st.subheader("Recently Analyzed Videos")
    rows = list_recent_analyses(conn, channel_id, limit=6)
    if not rows:
        st.info("No analyzed videos yet.")
    else:
        for r in rows:
            with st.expander(r["title"], expanded=False):
                st.write(r["url"])
                try:
                    idea = json.loads(r["idea_json"])
                    render_trade_cards_with_charts(idea, where="Video", chart_period=chart_period, chart_interval=chart_interval)
                except Exception:
                    st.code(r["idea_json"])

def page_analyze(channel_id: str, current_model: str, chart_period: str, chart_interval: str):
    st.button("Analyze latest videos", type="primary", use_container_width=True,
              on_click=lambda: analyze_workflow(channel_id, current_model, chart_period, chart_interval))

def page_history(channel_id: str, chart_period: str, chart_interval: str):
    conn = get_db()
    rows = list_all_analyses(conn, channel_id)
    st.subheader(f"History ({len(rows)} videos)")
    for r in rows:
        with st.expander(f"{r['published']} — {r['title']}", expanded=False):
            st.write(r["url"])
            try:
                idea = json.loads(r["idea_json"])
                render_trade_cards_with_charts(idea, where="History", chart_period=chart_period, chart_interval=chart_interval)
            except Exception:
                st.code(r["idea_json"])

# ----------------- Main -----------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

    # Sidebar — minimal controls
    st.sidebar.title("Settings")
    channel_id = st.sidebar.text_input("YouTube Channel ID", value=DEFAULT_CHANNEL_ID)
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = DEFAULT_MODEL
    current_model = st.sidebar.text_input("Model for analysis", value=st.session_state["current_model"])
    st.session_state["current_model"] = current_model

    # Global chart timeframe (applies to all charts)
    tf_label = st.sidebar.selectbox("Chart timeframe", list(TIMEFRAME_MAP.keys()), index=2)  # default 1M
    chart_period, chart_interval = TIMEFRAME_MAP[tf_label]

    # One-click hard reset
    if st.sidebar.button("🗑️ Clear ALL Data", type="secondary", use_container_width=True):
        nuke_everything()

    # Tabs
    st.title("Justin Bennett — Whisper Analyzer")
    tabs = st.tabs(["Dashboard", "Analyze", "History"])
    with tabs[0]:
        page_dashboard(channel_id, chart_period, chart_interval)
    with tabs[1]:
        page_analyze(channel_id, current_model, chart_period, chart_interval)
    with tabs[2]:
        page_history(channel_id, chart_period, chart_interval)

if __name__ == "__main__":
    main()
