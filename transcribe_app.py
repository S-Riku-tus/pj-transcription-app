#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google AI Studio (Gemini API) で音声/動画を文字起こし（CLI + 簡易Web）

使い方（CLI）:
  python transcribe_app.py --file "C:\\path\\to\\input.mp4" --out transcript.txt

使い方（Web）:
  uvicorn transcribe_app:app --host 0.0.0.0 --port 8000 --reload
  -> http://127.0.0.1:8000

必要な環境変数 (.env または OS 環境変数):
  GOOGLE_API_KEY = <Google AI Studio (Gemini) APIキー>

依存:
  pip install google-genai fastapi uvicorn python-multipart python-dotenv
"""

import os
import io
import time
import tempfile
import logging
import mimetypes
from typing import Optional
from pathlib import Path

# =========================
# .env を堅牢に読み込む
# =========================
def _load_env():
    paths = [
        Path(__file__).with_name(".env"),  # スクリプトと同じ階層
        Path.cwd() / ".env",               # 実行時カレント
    ]
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

# =========================
# Windows での mimetypes 補強
# =========================
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

# =========================
# ロギング
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("transcribe")

# =========================
# Webサーバ（FastAPI）
# =========================
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================
# google-genai
# =========================
from google import genai
from google.genai import types  # UploadFileConfig / GenerateContentConfig など
from google.genai.types import SafetySetting

# ====== 設定 ======
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_PROMPT = (
    "次の音声/動画を文字起こししてください。"
    "話し言葉は読みやすい文章に整形してください。"
    "不要なノイズやフィラーは可能な範囲で削除してください。"
    "日本語・英語が混在していても、そのまま自然に出力してください。"
)

# ====== クライアント初期化 ======
def _get_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "環境変数 GOOGLE_API_KEY / GEMINI_API_KEY が未設定です。"
            " .env に GOOGLE_API_KEY=... を書くか、OS 環境変数に設定してください。"
        )
    return genai.Client(api_key=key)

# ====== ファイル取り込み & ACTIVE待ち ======
def upload_and_wait_ready(client: genai.Client, file_path: str):
    """
    SDK仕様に合わせ、file= にパス文字列を渡す。
    display_name は渡さない（日本語名がヘッダーに入り httpx が落ちるのを回避）。
    取り込みが ACTIVE になるまで状態をログ出しして待機、最大10分でタイムアウト。
    """
    base = os.path.basename(file_path)
    logger.info("uploading file via path: %s", base)

    # パス文字列だけ渡す（MIMEは拡張子から推定）
    uploaded = client.files.upload(file=file_path)
    logger.info("uploaded: name=%s state=%s", getattr(uploaded, "name", None), getattr(uploaded.state, "name", None))

    # 取り込み完了まで待つ
    max_wait_s = 600   # 必要に応じて延長（秒）
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

# ====== 本処理 ======
def transcribe_with_gemini(
    file_path: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    client = _get_client()
    model = model or GEMINI_MODEL
    prompt = prompt or DEFAULT_PROMPT

    uploaded = upload_and_wait_ready(client, file_path)
    logger.info("start generate: model=%s file=%s", model, getattr(uploaded, "name", None))

    safety = [
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_SEXUAL_CONTENT", threshold="BLOCK_NONE"),
    ]

    resp = client.models.generate_content(
        model=model,
        contents=[prompt, uploaded],  # prompt とファイルの順序はどちらでも可
        config=types.GenerateContentConfig(safety_settings=safety),
    )

    logger.info("done generate")

    # 新SDKは resp.text を優先、なければ output_text でフォールバック
    text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
    if not text:
        try:
            text = "".join([p.text for p in resp.candidates[0].content.parts])
        except Exception:
            text = ""
    return text or "[No transcription text returned]"

# ====== CLI ======
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Google AI Studio (Gemini) 音声/動画 文字起こし CLI")
    parser.add_argument("--file", required=True, help="音声/動画ファイル (mp3, wav, m4a, mp4, mov, webm, mkv, aac, flac, ogg)")
    parser.add_argument("--prompt", default=None, help="整形指示など（未指定でデフォルトプロンプト）")
    parser.add_argument("--model", default=None, help=f"使用モデル（未指定で {GEMINI_MODEL}）")
    parser.add_argument("--out",   default=None, help="出力テキストファイル（未指定なら標準出力）")
    args = parser.parse_args()

    text = transcribe_with_gemini(args.file, prompt=args.prompt, model=args.model)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] 書き出し: {args.out}")
    else:
        print(text)

# ====== Web (FastAPI) ======
app = FastAPI(
    title="Gemini Transcription Demo",
    description="Google AI Studio (Gemini) を使った簡易文字起こしWebアプリ",
    version="0.4.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公開時は適切に制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8"/>
  <title>Gemini Transcription Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans JP", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif; padding: 24px; max-width: 860px; margin: 0 auto; }
    .card { padding: 20px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.05); }
    label { display:block; margin: 12px 0 6px; font-weight: 600; }
    input[type="file"], textarea, input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
    button { padding: 10px 16px; border: none; border-radius: 10px; background: #111827; color: white; font-weight: 600; cursor: pointer; }
    button:hover { opacity: .9; }
    .hint { color: #666; font-size: 12px; }
    .result { white-space: pre-wrap; background: #fafafa; padding: 16px; border-radius: 8px; border: 1px solid #eee; }
  </style>
</head>
<body>
  <h1>Google AI Studio (Gemini) 文字起こしデモ</h1>
  <div class="card">
    <form id="form" method="post" enctype="multipart/form-data" action="/transcribe">
      <label>音声/動画ファイル</label>
      <input name="file" type="file" accept=".mp3,.wav,.m4a,.mp4,.mov,.webm,.mkv,.aac,.flac,.ogg" required />

      <label>プロンプト（任意）</label>
      <textarea name="prompt" rows="5" placeholder="整形・要約・言語指定などがあれば書いてください。"></textarea>
      <div class="hint">未指定の場合はデフォルトの整形プロンプトを使用します。</div>

      <label>モデル（任意）</label>
      <input name="model" type="text" placeholder="例: gemini-2.5-pro" />

      <div style="margin-top: 16px;">
        <button type="submit">文字起こしする</button>
      </div>
    </form>
  </div>

  <div id="out" style="margin-top: 24px;"></div>

  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      out.innerHTML = '<p>処理中...</p>';
      const fd = new FormData(form);
      const res = await fetch('/transcribe', { method: 'POST', body: fd });
      const text = await res.text();
      out.innerHTML = '<h2>結果</h2><div class="result"></div>';
      out.querySelector('.result').textContent = text;
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {
        "status": "ok" if bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")) else "no-key",
        "model": GEMINI_MODEL,
    }

@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise HTTPException(status_code=500, detail="環境変数 GOOGLE_API_KEY/GEMINI_API_KEY が未設定です。")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ""
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
        )
        logger.info("web transcription done: name=%s chars=%d", file.filename, len(text))
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

# ====== エントリポイント ======
if __name__ == "__main__":
    import sys
    if any(arg.startswith("--file") for arg in sys.argv):
        _cli()
    else:
        try:
            import uvicorn
            uvicorn.run("transcribe_app:app", host="127.0.0.1", port=8000, reload=True)
        except Exception:
            print("Web起動: uvicorn transcribe_app:app --host 0.0.0.0 --port 8000 --reload")
