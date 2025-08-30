# justin_bennett_trader.py
# Minimal, professional UI — Whisper-only transcription
# Features:
#   • Pulls Justin Bennett channel feed
#   • Transcribes with OpenAI Whisper ONLY (yt-dlp for audio; prompts for cookies if needed)
#   • Analyzes via OpenAI (structured trade ideas)
#   • Shows a LIVE chart next to every trade idea/snapshot (with entry/SL/TP overlays when present)
#   • History view
#   • One-click "Clear ALL Data" in sidebar (deletes DB, caches, and session)

import os, re, json, sqlite3, tempfile, datetime as dt, threading, time
from typing import List, Dict, Optional, Tuple

import streamlit as st
import feedparser
from openai import OpenAI
import yfinance as yf
import plotly.graph_objects as go

# ----------------- App basics -----------------
APP_NAME = "Justin Bennett — Whisper-Only Analyzer"
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

# ----------------- Whisper-only transcription -----------------
_YTDLP = None
def _lazy_import_ytdlp():
    global _YTDLP
    if _YTDLP is None:
        import yt_dlp as _YTDLP  # lazy import
    return _YTDLP

def transcribe_with_whisper(video_url: str, client: OpenAI, cookies_bytes: Optional[bytes] = None) -> Tuple[Optional[str], Optional[str]]:
    """Download audio with yt-dlp (optionally with cookies) then send to OpenAI whisper-1."""
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
            if not text:
                return None, "Whisper returned empty text"
            return text, None
        finally:
            try:
                for fn in os.listdir(tmpdir):
                    try: os.remove(os.path.join(tmpdir, fn))
                    except: pass
                os.rmdir(tmpdir)
            except: pass
    except Exception as e:
        return None, f"Whisper download/transcription failed: {e}"

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
    cands = []
    if re.fullmatch(r"[A-Z]{3}[A-Z]{3}", inst):  # FX like EURUSD
        cands.append(f"{inst}=X")
    if inst.endswith("USD") and len(inst) in (6,7):  # Crypto like BTCUSD
        cands.append(f"{inst[:-3]}-USD")
    if inst in ("DXY", "USDIDX", "USDX"):
        cands.extend(["^DXY", "DX-Y.NYB"])
    if inst in ("XAUUSD","GOLD","XAU"):
        cands.extend(["XAUUSD=X", "GC=F"])
    if inst in ("SPX","SPY","S&P500","GSPC"):
        cands.extend(["^GSPC","SPY"])
    cands.append(inst)  # fallback
    return list(dict.fromkeys(cands))

@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(inst: str) -> Tuple[Optional[float], Optional[str]]:
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

@st.cache_data(ttl=300, show_spinner=False)
def get_chart_data(inst: str):
    """Returns (df, used_ticker)."""
    for t in _candidates_for_instrument(inst):
        try:
            ti = yf.Ticker(t)
            df = ti.history(period="3mo", interval="1h")
            if df is None or df.empty:
                df = ti.history(period="6mo", interval="1d")
            if df is not None and not df.empty:
                df = df.dropna()
                return df, t
        except Exception:
            continue
    return None, None

def _fmt_level(x):
    if x is None: return "—"
    if isinstance(x, (int,float)): return f"{x:.5f}".rstrip("0").rstrip(".")
    return str(x)

def plot_live_chart(instrument: str, entries=None, sl=None, tps=None):
    """Candlestick chart with optional horizontal lines for entries/SL/TPs."""
    df, used = get_chart_data(instrument)
    if df is None or df.empty:
        st.info("Chart data unavailable.")
        return
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#06c167", decreasing_line_color="#d94f4f", name=used or instrument
        )
    ])
    # Overlay levels
    shapes = []
    annos = []
    def add_hline(y, color, name):
        if y is None: return
        try: yval = float(y)
        except: return
        shapes.append(dict(type="line", xref="x", x0=df.index[0], x1=df.index[-1],
                           yref="y", y0=yval, y1=yval, line=dict(color=color, width=1.5, dash="dot")))
        annos.append(dict(x=df.index[-1], y=yval, xref="x", yref="y",
                          xanchor="left", text=name, showarrow=False, font=dict(color=color, size=10)))
    # entries
    for e in (entries or []):
        add_hline(e.get("level"), "#3b82f6", f"Entry ({e.get('type','')}) {_fmt_level(e.get('level'))}")
    # SL
    if sl and isinstance(sl, dict):
        add_hline(sl.get("level"), "#ef4444", f"SL {_fmt_level(sl.get('level'))}")
    # TPs
    for t in (tps or []):
        add_hline(t.get("level"), "#10b981", f"{t.get('label','TP')}: {_fmt_level(t.get('level'))}")

    fig.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=360,
        shapes=shapes,
        annotations=annos,
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

def render_trade_cards(idea: Dict, where: str = "Video"):
    trades = _normalize_trade_ideas(idea)
    if not trades:
        st.info("No structured trade ideas found.")
        return
    for i, tr in enumerate(trades, 1):
        instrument = (tr.get("instrument") or "").upper()
        price, used = (None, None)
        if instrument: price, used = get_live_price(instrument)
        colL, colR = st.columns([1.2, 2.0], vertical_alignment="top")
        with colL:
            with st.container(border=True):
                hdr = f"**{where} Trade {i} — {instrument or '—'}**"
                if price is not None: hdr += f"  |  Live: `{_fmt_level(price)}` ({used})"
                st.markdown(hdr)
                st.markdown(f"**Bias:** {tr.get('bias','—')}  \n**TF:** {tr.get('timeframe','—')}")
                st.markdown(f"**Plan:** {tr.get('plan','—')}")
                if tr.get("entries"):
                    st.markdown("**Entries:**")
                    for e in tr["entries"]:
                        st.markdown(f"- {e.get('type','entry')} @ `{_fmt_level(e.get('level'))}` — {e.get('condition','')}")
                sl = tr.get("stop_loss") or {}
                tps = tr.get("take_profits") or []
                st.markdown("**Stop Loss:** " + (f"`{_fmt_level(sl.get('level'))}`" if sl else "—"))
                if tps:
                    st.markdown("**Take Profits:**")
                    for t in tps:
                        st.markdown(f"- {t.get('label','TP')}: `{_fmt_level(t.get('level'))}`")
        with colR:
            plot_live_chart(instrument, tr.get("entries"), tr.get("stop_loss"), tr.get("take_profits"))
        st.divider()

# ----------------- Workflow -----------------
def analyze_workflow(channel_id: str, model_name: str, cookies_bytes: Optional[bytes] = None, first_run_take: int = 3):
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
            text, err = transcribe_with_whisper(v["url"], client, cookies_bytes=cookies_bytes)

            # If Whisper path failed, PROMPT for cookies
            if not text:
                status.update(label="Transcription failed", state="error")
                st.error(f"Whisper error: {err or 'unknown error'}")
                box = st.container(border=True)
                with box:
                    st.markdown("##### Upload Netscape `cookies.txt` to fix YouTube access")
                    up = st.file_uploader("Upload cookies.txt", type=["txt"], key=f"cookies_{v['video_id']}")
                    c1, c2 = st.columns([1,1])
                    if up is not None:
                        st.session_state["cookies_bytes"] = up.read()
                        st.success("Cookies loaded.")
                    if c1.button("Retry now", use_container_width=True, key=f"retry_{v['video_id']}"):
                        cb = st.session_state.get("cookies_bytes", None)
                        text2, err2 = transcribe_with_whisper(v["url"], client, cookies_bytes=cb)
                        if not text2:
                            st.error(f"Still failing with cookies: {err2 or 'unknown error'}")
                            st.stop()
                        text = text2
                    if c2.button("Skip", use_container_width=True, key=f"skip_{v['video_id']}"):
                        save_video_row(conn, {
                            "video_id": v["video_id"], "channel_id": channel_id, "published": v["published"],
                            "title": v["title"], "url": v["url"], "transcript": None,
                            "summary": None, "idea_json": None, "analyzed_at": now_utc(), "model": model_name
                        })
                        continue

            if not text:
                st.stop()

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
                st.success(v["title"])
                try:
                    idea = json.loads(idea_json)
                    render_trade_cards(idea, where="Video")
                except Exception:
                    st.code(idea_json)
                analyzed_rows.append({"video_id": v["video_id"], "published": v["published"], "title": v["title"],
                                      "url": v["url"], "summary": summary, "idea_json": idea_json})
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(str(e))

    # Snapshot synthesis
    if analyzed_rows or recent_rows:
        try:
            summary, idea_json = synthesize_overall(get_openai_client(), model_name, recent_rows, analyzed_rows)
            save_snapshot(conn, channel_id, summary, idea_json, [x["video_id"] for x in analyzed_rows] or [x["video_id"] for x in recent_rows])
            st.subheader("Overall Snapshot")
            st.success(summary)
            try:
                idea = json.loads(idea_json)
                render_trade_cards(idea, where="Overall")
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
def page_dashboard(channel_id: str):
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
            render_trade_cards(idea, where="Snapshot")
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
                if r["summary"]:
                    st.success(r["summary"])
                try:
                    idea = json.loads(r["idea_json"])
                    render_trade_cards(idea, where="Video")
                except Exception:
                    st.code(r["idea_json"])

def page_analyze(channel_id: str, current_model: str, cookies_bytes: Optional[bytes]):
    st.button("Analyze latest videos", type="primary", use_container_width=True, on_click=lambda: analyze_workflow(channel_id, current_model, cookies_bytes))

    st.caption("If transcription fails, you'll be prompted to upload cookies and retry.")
    # Optional manual preload of cookies:
    up = st.file_uploader("Upload cookies.txt (optional)", type=["txt"])
    if up is not None:
        st.session_state["cookies_bytes"] = up.read()
        st.success("Cookies loaded for this session.")

def page_history(channel_id: str):
    conn = get_db()
    rows = list_all_analyses(conn, channel_id)
    st.subheader(f"History ({len(rows)} videos)")
    for r in rows:
        with st.expander(f"{r['published']} — {r['title']}", expanded=False):
            st.write(r["url"])
            if r["summary"]:
                st.success(r["summary"])
            try:
                idea = json.loads(r["idea_json"])
                render_trade_cards(idea, where="History")
            except Exception:
                st.code(r["idea_json"])

# ----------------- Main -----------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

    # Sidebar — minimal & practical
    st.sidebar.title("Settings")
    channel_id = st.sidebar.text_input("YouTube Channel ID", value=DEFAULT_CHANNEL_ID)
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = DEFAULT_MODEL
    current_model = st.sidebar.text_input("Model for analysis", value=st.session_state["current_model"])
    st.session_state["current_model"] = current_model

    # One-click hard reset (requested)
    if st.sidebar.button("🗑️ Clear ALL Data", type="secondary", use_container_width=True):
        nuke_everything()

    # Tabs-based navigation (clean)
    st.title("Justin Bennett — Whisper-Only Analyzer")
    tabs = st.tabs(["Dashboard", "Analyze", "History"])
    with tabs[0]:
        page_dashboard(channel_id)
    with tabs[1]:
        cookies_bytes = st.session_state.get("cookies_bytes", None)
        page_analyze(channel_id, current_model, cookies_bytes)
    with tabs[2]:
        page_history(channel_id)

if __name__ == "__main__":
    main()
