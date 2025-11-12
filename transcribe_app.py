#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google AI Studio (Gemini API) を使った音声/動画の文字起こしスクリプト（たたき台）
- 使い方（CLI）:  python transcribe_app.py --file path/to/audio_or_video.mp4
- 使い方（Web）:  uvicorn transcribe_app:app --reload --port 8000
  -> http://127.0.0.1:8000 でアップロードして文字起こし

必要な環境変数:
  GOOGLE_API_KEY: Google AI Studio (Gemini API) のAPIキー

主な依存ライブラリ:
  google-genai>=0.3.0
  fastapi
  uvicorn
  python-multipart

インストール例:
  pip install google-genai fastapi uvicorn python-multipart
"""

import os
import io
import time
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai.types import SafetySetting

# ====== 設定 ======
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")  # 必要に応じて変更
DEFAULT_PROMPT = (
    "次の音声/動画を文字起こししてください。"
    "話し言葉は読みやすい文章に整形してください。"
    "不要なノイズやフィラーは可能な範囲で削除してください。"
    "日本語・英語が混在していても、そのまま自然に出力してください。"
)

# ====== クライアント初期化 ======
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    # FastAPI 起動時に落ちないよう遅延チェックもするが、CLIでは早めに気付けるようにここでも注意喚起
    print("[WARN] 環境変数 GOOGLE_API_KEY が設定されていません。CLI/Web実行前に設定してください。")

def _get_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("環境変数 GOOGLE_API_KEY が未設定です。Google AI StudioのAPIキーを設定してください。")
    return genai.Client(api_key=key)

# ====== 共通ユーティリティ ======
def upload_and_wait_ready(client: genai.Client, file_path: str):
    """
    GeminiのファイルAPIにアップロードし、利用可能(=ACTIVE)になるまで待機してFileオブジェクトを返す。
    """
    # アップロード
    with open(file_path, "rb") as f:
        uploaded = client.files.upload(file=f)

    # 処理完了待ち
    # state: PROCESSING -> ACTIVE
    while getattr(uploaded, "state", None) and getattr(uploaded.state, "name", None) == "PROCESSING":
        time.sleep(1.5)
        uploaded = client.files.get(name=uploaded.name)

    # 失敗チェック
    state_name = getattr(uploaded.state, "name", None)
    if state_name != "ACTIVE":
        raise RuntimeError(f"ファイルの準備に失敗しました。state={state_name}, file={uploaded.name}")

    return uploaded


def transcribe_with_gemini(file_path: str, prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    ローカルの音声/動画ファイルをGemini APIに渡し、文字起こしテキストを返す。
    """
    client = _get_client()
    model = model or GEMINI_MODEL
    prompt = prompt or DEFAULT_PROMPT

    # ファイルアップロード & READY待ち
    uploaded = upload_and_wait_ready(client, file_path)

    # セーフティ設定（必要に応じて緩和/強化）
    safety = [
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        SafetySetting(category="HARM_CATEGORY_SEXUAL_CONTENT", threshold="BLOCK_NONE"),
    ]

    # 推論リクエスト
    # input は [ファイル参照, プロンプト文字列] の配列で与えられる
    resp = client.responses.create(
        model=model,
        input=[uploaded, prompt],
        safety_settings=safety,
    )

    # 出力テキストを取り出し
    # 新しいクライアントでは response.output_text に最終テキストが入る
    text = getattr(resp, "output_text", None)
    if not text:
        # 念のためフォールバック
        try:
            text = "".join([c.text for c in resp.candidates[0].content.parts])  # 仕様変更に備えた保険
        except Exception:
            text = ""

    if not text:
        text = "[No transcription text returned]"

    return text


# ====== CLI モード ======
def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Google AI Studio (Gemini) 音声/動画 文字起こし CLI")
    parser.add_argument("--file", required=True, help="音声/動画ファイルへのパス (mp3, wav, m4a, mp4, mov, webm など)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Gemini に渡すプロンプト（整形指示など）")
    parser.add_argument("--model", default=GEMINI_MODEL, help="使用するモデル（例: gemini-1.5-pro）")
    parser.add_argument("--out", default="", help="出力テキストファイルパス（未指定なら標準出力）")
    args = parser.parse_args()

    text = transcribe_with_gemini(args.file, prompt=args.prompt, model=args.model)

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] 書き出し: {args.out}")
    else:
        print(text)


# ====== Web アプリ (FastAPI) ======
app = FastAPI(title="Gemini Transcription Demo", description="Google AI Studio (Gemini) を使った簡易文字起こしWebアプリ", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて制限
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
      <input name="model" type="text" placeholder="例: gemini-1.5-pro" />

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

@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    # APIキー未設定のハンドリング
    if not os.environ.get("GOOGLE_API_KEY"):
        return PlainTextResponse(
            "サーバ側の環境変数 GOOGLE_API_KEY が未設定です。管理者にご確認ください。",
            status_code=500,
        )

    # 一時ファイルに保存してから処理
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        text = transcribe_with_gemini(tmp_path, prompt=prompt or DEFAULT_PROMPT, model=model or GEMINI_MODEL)
        return PlainTextResponse(text, status_code=200)
    except Exception as e:
        return PlainTextResponse(f"[ERROR] {str(e)}", status_code=500)
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ====== エントリポイント ======
if __name__ == "__main__":
    # 引数があればCLI、なければWebの雰囲気にする…が、
    # 明示的にCLI想定: --file を指定している場合のみCLIとして動作
    import sys
    if any(arg.startswith("--file") for arg in sys.argv):
        _cli()
    else:
        # 簡易起動（uvicorn未使用でもデバッグサーバで動作）
        try:
            import uvicorn
            uvicorn.run("transcribe_app:app", host="127.0.0.1", port=8000, reload=True)
        except Exception:
            # uvicornが無い場合のフォールバック（FastAPIのdev serverは無いので案内だけ）
            print("Webモードで起動するには uvicorn をインストールし、以下で起動してください:")
            print("  uvicorn transcribe_app:app --reload --port 8000")
