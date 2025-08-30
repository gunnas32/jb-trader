# audio_relay.py
# A tiny service that downloads YouTube audio with yt-dlp and returns the bytes.
# Deploy: uvicorn audio_relay:app --host 0.0.0.0 --port $PORT

import os, tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response

app = FastAPI(title="Audio Relay (yt-dlp)")

_YTDLP = None
def _lazy_import_ytdlp():
    global _YTDLP
    if _YTDLP is None:
        import yt_dlp as _YTDLP
    return _YTDLP

class FetchReq(BaseModel):
    url: str
    cookies: Optional[str] = None  # optional Netscape cookies text

@app.post("/fetch")
def fetch(req: FetchReq):
    try:
        ydlp = _lazy_import_ytdlp()
        tmpdir = tempfile.mkdtemp(prefix="relay_")
        outpath = os.path.join(tmpdir, "%(id)s.%(ext)s")
        cookiefile = None
        if req.cookies:
            cookiefile = os.path.join(tmpdir, "cookies.txt")
            with open(cookiefile, "w", encoding="utf-8") as cf:
                cf.write(req.cookies)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outpath,
            'quiet': True,
            'noprogress': True,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            },
        }
        if cookiefile:
            ydl_opts['cookiefile'] = cookiefile

        with ydlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            path = ydl.prepare_filename(info)

        with open(path, "rb") as f:
            data = f.read()

        try:
            for fn in os.listdir(tmpdir):
                try: os.remove(os.path.join(tmpdir, fn))
                except: pass
            os.rmdir(tmpdir)
        except: pass

        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
