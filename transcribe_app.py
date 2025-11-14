#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google AI Studio (Gemini API) で音声/動画を文字起こし（CLI + Web）
- 動画は ffmpeg で音声抽出（m4a/mono/16kHz/64kbps）
- 出力は ./outputs/ に保存（音声: ./outputs/audio, テキスト: ./outputs/transcripts）
- UI は ./web/index.html に分離（左右2カラム: 左=設定フォーム, 右=結果パネル）

CLI:
  python transcribe_app.py --file "C:\\path\\to\\input.mp4" --audio-first
Web:
  uvicorn transcribe_app:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import io
import re
import time
import tempfile
import logging
import mimetypes
import subprocess
from typing import Optional, Tuple
from pathlib import Path

# ---------- .env ----------
def _load_env():
    paths = [Path(__file__).with_name(".env"), Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv
        for p in paths:
            if p.exists():
                load_dotenv(p, override=True)
                print(f"[INFO] .env loaded from: {p}")
                return
        print("[INFO] .env not found. Using process env vars only.")
    except Exception as e:
        print(f"[WARN] dotenv not used ({e}). If needed: pip install python-dotenv")
_load_env()

# ---------- mimetypes 補強（Windows保険） ----------
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('audio/mp4', '.m4a')
mimetypes.add_type('video/quicktime', '.mov')
mimetypes.add_type('video/webm', '.webm')
mimetypes.add_type('video/x-matroska', '.mkv')
mimetypes.add_type('audio/mpeg', '.mp3')
mimetypes.add_type('audio/wav', '.wav')
mimetypes.add_type('audio/aac', '.aac')
mimetypes.add_type('audio/flac', '.flac')
mimetypes.add_type('audio/ogg', '.ogg')

# ---------- ロギング ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("transcribe")

# ---------- FastAPI ----------
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- google-genai ----------
from google import genai
from google.genai import types          # enums & config
from google.genai.errors import ServerError

# ---------- 出力ディレクトリ ----------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT  = SCRIPT_DIR / "outputs"
AUDIO_DIR = OUT_ROOT / "audio"
TRANS_DIR = OUT_ROOT / "transcripts"
for d in (AUDIO_DIR, TRANS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Web(静的)ディレクトリ ----------
WEB_DIR = SCRIPT_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)  # ない場合も作っておく

# ---------- 設定 ----------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_PROMPT = (
    "次の音声/動画を文字起こししてください。"
    "話し言葉は読みやすい文章に整形してください。"
    "不要なノイズやフィラーは可能な範囲で削除してください。"
    "日本語・英語が混在していても、そのまま自然に出力してください。"
)
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

# ---------- クライアント ----------
def _get_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("環境変数 GOOGLE_API_KEY / GEMINI_API_KEY が未設定です。")
    return genai.Client(api_key=key)

# ---------- ユーティリティ ----------
def _is_video(path: str) -> bool:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext in VIDEO_EXTS:
        return True
    mt, _ = mimetypes.guess_type(path)
    return (mt or "").startswith("video/")

def _which_ffmpeg() -> Optional[str]:
    # .env の明示設定を優先
    for key in ("FFMPEG_BIN", "FFMPEG_PATH"):
        p = os.environ.get(key)
        if p and os.path.exists(p):
            return p
    from shutil import which
    for cmd in ("ffmpeg", "ffmpeg.exe"):
        p = which(cmd)
        if p:
            return p
    return None

def _sanitize_tmp_filename(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"[^\w.\-]+", "_", base)  # 日本語や空白を安全名に置換
    return base or "audio"

def extract_audio_ffmpeg(input_path: str, ar: int = 16000, ac: int = 1, abr: str = "64k") -> Tuple[str, bool]:
    """動画から音声のみ抽出（m4a/mono/16kHz/64kbps）。戻り値: (出力パス, 作成したか)"""
    ff = _which_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg が見つかりません。PATH を通すか、.env に FFMPEG_BIN を指定してください。")
    base = _sanitize_tmp_filename(os.path.basename(input_path))
    out_path = str((AUDIO_DIR / f"{base}.m4a").resolve())
    cmd = [ff, "-y", "-i", input_path, "-vn", "-ac", str(ac), "-ar", str(ar), "-b:a", abr, out_path]
    logger.info("extract audio via ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        logger.error("ffmpeg failed: %s", msg[-500:])
        raise RuntimeError("ffmpeg による音声抽出に失敗しました。")
    return out_path, True

# ---------- アップロード & ACTIVE 待機 ----------
def upload_and_wait_ready(client: genai.Client, file_path: str):
    base = os.path.basename(file_path)
    logger.info("uploading file via path: %s", base)
    uploaded = client.files.upload(file=file_path)  # パス文字列でOK
    logger.info("uploaded: name=%s state=%s", getattr(uploaded, "name", None), getattr(uploaded.state, "name", None))

    max_wait_s = 600
    waited = 0.0
    interval = 2.0
    while True:
        state_name = getattr(uploaded.state, "name", None)
        logger.info("file state: %s (waited=%ds)", state_name, int(waited))
        if state_name == "ACTIVE":
            break
        if state_name not in ("PROCESSING", "CREATING"):
            raise RuntimeError(f"unexpected file state: {state_name}")
        if waited >= max_wait_s:
            raise TimeoutError(f"file not ACTIVE within {max_wait_s}s (state={state_name})")
        time.sleep(interval)
        waited += interval
        uploaded = client.files.get(name=uploaded.name)
    return uploaded

# ---------- セーフティ設定 ----------
def _safety_settings():
    return [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,         threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

# ---------- 生成（リトライ＋フォールバック） ----------
def _generate_with_retry(client, model, prompt, uploaded):
    backoffs = [2, 4, 8]
    try_models = [model, "gemini-1.5-pro", "gemini-1.5-flash-8b"]
    last_err = None
    for m in try_models:
        for i, sec in enumerate([0] + backoffs):
            if sec:
                logger.info("retrying generate (model=%s, attempt=%d) in %ds", m, i+1, sec)
                time.sleep(sec)
            try:
                logger.info("calling generateContent (model=%s)", m)
                resp = client.models.generate_content(
                    model=m,
                    contents=[prompt, uploaded],
                    config=types.GenerateContentConfig(safety_settings=_safety_settings()),
                )
                return resp
            except ServerError as e:
                last_err = e
                logger.warning("server error (%s): %s", type(e).__name__, str(e))
                continue
            except Exception as e:
                last_err = e
                logger.warning("non-retriable error (%s): %s", type(e).__name__, str(e))
                break
    raise last_err

# ---------- 文字起こし本体 ----------
def transcribe_with_gemini(file_path: str, prompt: Optional[str] = None, model: Optional[str] = None, audio_first: bool = False) -> str:
    client = _get_client()
    model  = model or GEMINI_MODEL
    prompt = prompt or DEFAULT_PROMPT

    tmp_audio = None
    created  = False
    try:
        if audio_first or _is_video(file_path):
            tmp_audio, created = extract_audio_ffmpeg(file_path)
            src = tmp_audio
            logger.info("use extracted audio: %s", tmp_audio)
        else:
            src = file_path

        uploaded = upload_and_wait_ready(client, src)
        logger.info("start generate: model=%s file=%s", model, getattr(uploaded, "name", None))
        resp = _generate_with_retry(client, model, prompt, uploaded)
        logger.info("done generate")

        text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        if not text:
            try:
                text = "".join([p.text for p in resp.candidates[0].content.parts])
            except Exception:
                text = ""
        return text or "[No transcription text returned]"
    finally:
        if created and tmp_audio and os.path.exists(tmp_audio):
            try:
                os.remove(tmp_audio)
            except Exception:
                pass

# ---------- CLI ----------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Google AI Studio (Gemini) 音声/動画 文字起こし CLI")
    parser.add_argument("--file", required=True, help="音声/動画ファイル (mp3, wav, m4a, mp4, mov, webm, mkv, aac, flac, ogg)")
    parser.add_argument("--prompt", default=None, help="整形指示（未指定でデフォルト）")
    parser.add_argument("--model",  default=None, help=f"使用モデル（未指定で {GEMINI_MODEL}）")
    parser.add_argument("--audio-first", action="store_true", help="動画は先に音声抽出してから解析（推奨）")
    parser.add_argument("--out", default=None, help="出力テキストファイル（未指定なら ./outputs/transcripts/<元名>.txt）")
    args = parser.parse_args()

    text = transcribe_with_gemini(args.file, prompt=args.prompt, model=args.model, audio_first=args.audio_first)

    out_path = args.out
    if not out_path:
        stem = Path(args.file).stem
        safe_stem = _sanitize_tmp_filename(stem)
        out_path = str((TRANS_DIR / f"{safe_stem}.txt").resolve())

    with io.open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] 書き出し: {out_path}")

# ---------- Web (FastAPI) ----------
app = FastAPI(
    title="Gemini Transcription Demo",
    description="Google AI Studio (Gemini) を使った簡易文字起こしWebアプリ（動画は自動で音声抽出）",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公開時は限定してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /web 配下に静的ファイルをマウント（CSS/JSを増やす場合もOK）
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# ルートは web/index.html を返す
@app.get("/", response_class=FileResponse)
def root():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        # ファイルが無い場合の導線
        return PlainTextResponse("UIファイルがありません。web/index.html を作成してください。", status_code=500)
    return FileResponse(index_path)

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {
        "status": "ok" if bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")) else "no-key",
        "model": GEMINI_MODEL,
        "ffmpeg": bool(_which_ffmpeg()),
    }

@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model:  Optional[str] = Form(None),
):
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise HTTPException(status_code=500, detail="環境変数 GOOGLE_API_KEY/GEMINI_API_KEY が未設定です。")

    tmp_path = None
    try:
        suffix  = os.path.splitext(file.filename or "")[1] or ""
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="空ファイルです。")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("web transcription start: name=%s size=%.2fMB", file.filename, len(content)/1024/1024)

        text = transcribe_with_gemini(
            tmp_path,
            prompt=prompt or DEFAULT_PROMPT,
            model=model or GEMINI_MODEL,
            audio_first=_is_video(file.filename or ""),
        )

        # 出力保存（./outputs/transcripts/<元ファイル名>.txt）
        safe_stem = _sanitize_tmp_filename(Path(file.filename or "uploaded").stem)
        out_path  = (TRANS_DIR / f"{safe_stem}.txt").resolve()
        with io.open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("saved transcript to: %s", out_path)

        return PlainTextResponse(text, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("transcription error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# ---------- エントリ ----------
if __name__ == "__main__":
    import sys
    if any(arg.startswith("--file") for arg in sys.argv):
        _cli()
    else:
        try:
            import uvicorn
            uvicorn.run("transcribe_app:app", host="127.0.0.1", port=8000, reload=True)
        except Exception:
            print("Web起動: uvicorn transcribe_app:app --host 127.0.0.1 --port 8000 --reload")
