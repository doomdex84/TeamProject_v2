# app.py
from pathlib import Path
from dotenv import load_dotenv

# ✅ 환경 변수: .env가 없어도 .env.sample을 자동 인식
BASE_DIR = Path(__file__).resolve().parent
for name in (".env", ".env.local", ".env.sample"):
    load_dotenv(BASE_DIR / name, override=False)  # 기존 OS env는 덮어쓰지 않음

import os
import uuid
import base64
from typing import Dict, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from openai import OpenAI

# ✅ env 로드가 끝난 뒤에 analysis import (키가 제대로 반영됨)
from analysis import analyze_voice, OPENAI_MODEL, OPENAI_API_KEY

app = FastAPI(title="Voice → Description → Image Prompt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# (옵션) 이미지 엔드포인트용 OpenAI
if not os.getenv("OPENAI_API_KEY"):
    print("[app.py] WARNING: OPENAI_API_KEY is not set. /image/render disabled.")
    image_client: Optional[OpenAI] = None
else:
    image_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    idx = TEMPLATES_DIR / "index.html"
    if idx.exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h3>Server is running. POST /analyze_upload</h3>", status_code=200)

import tempfile

@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() or ".wav"
    data = await file.read()

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        finally:
            tmp.close()

        analysis = analyze_voice(tmp_path)

        # ✅ 목소리가 불명확할 경우 차단
        feat = analysis.get("features", {})
        if feat.get("is_silent") or (feat.get("duration_sec") or 0) < 0.6:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "목소리가 명확하지 않습니다. 주변 소음을 줄이고 조금 더 길게 녹음해주세요."
                },
                status_code=400
            )

        payload: Dict = {
            "ok": True,
            "file": {"name": file.filename, "saved_as": None, "path": None, "persisted": False},
            "features": feat,
            "description": analysis.get("description_ko", ""),
            "en_prompt": analysis.get("en_prompt", ""),
            "negative": analysis.get("negative", ""),
            "style_tags": analysis.get("style_tags", ""),
            "palette": analysis.get("palette", []),
            "seed": analysis.get("seed", ""),
            "avatar": analysis.get("avatar", {}),
            "llm_status": analysis.get("llm_status", {}),
        }
        return JSONResponse(payload)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ✅ 백워드 호환
@app.post("/analyze_api")
async def analyze_api(file: UploadFile = File(...)):
    return await analyze_upload(file)

# (옵션) OpenAI 이미지 엔드포인트
class ImageReq(BaseModel):
    prompt: str
    negative: Optional[str] = ""
    size: Optional[str] = "square"  # "square"|"portrait"|"landscape"

def _pick_openai_size(req_size: str) -> str:
    if req_size == "portrait": return "1024x1536"
    if req_size == "landscape": return "1536x1024"
    return "1024x1024"

def _save_b64_png(b64: str) -> str:
    out_dir = STATIC_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64)
    name = f"{uuid.uuid4().hex}.png"
    p = out_dir / name
    with open(p, "wb") as f: f.write(raw)
    return f"/static/out/{name}"

@app.post("/image/render")
def image_render(req: ImageReq):
    if image_client is None:
        raise HTTPException(status_code=503, detail="OpenAI image generation is not configured.")
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    size_str = _pick_openai_size(req.size)
    final_prompt = req.prompt + (f"\n\n### Negative:\n{req.negative}" if req.negative else "")
    try:
        resp = image_client.images.generate(
            model="gpt-image-1",
            prompt=final_prompt,
            size=size_str,
            n=1,
            quality="high",
        )
        b64 = resp.data[0].b64_json
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OpenAI image generation failed: {e}")

    url = _save_b64_png(b64)
    return {"imageUrl": url}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_enabled": bool(OPENAI_API_KEY),
        "llm_model": OPENAI_MODEL if OPENAI_API_KEY else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
