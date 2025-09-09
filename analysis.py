# analysis.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import librosa
import requests

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# 환경 변수 로드 & OpenAI 준비
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
# ✅ .env가 없어도 .env.sample을 읽도록 보장
for name in (".env", ".env.local", ".env.sample"):
    load_dotenv(BASE_DIR / name, override=False)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 키가 없으면 클라이언트를 만들지 않음 (import 시점 크래시 방지)
if not OPENAI_API_KEY:
    print("[analysis.py] WARNING: OPENAI_API_KEY is not set. Using fallback without LLM.")
    client: Optional[OpenAI] = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ LLM 상태(프론트에서 확인용)
_last_llm_status: Dict[str, Any] = {
    "used": False,
    "ok": False,
    "error": None,
    "model": OPENAI_MODEL,
    "has_api_key": bool(OPENAI_API_KEY),
}

# ---------------------------
# 유틸
# ---------------------------
def _safe_float(x, ndigits: int = 5) -> Optional[float]:
    try:
        return round(float(x), ndigits)
    except Exception:
        return None

def _nan_robust(values: np.ndarray, fn, default=None):
    try:
        v = fn(values[~np.isnan(values)])
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

@dataclass
class VoiceFeatures:
    duration_sec: Optional[float]
    f0_med: Optional[float]
    f0_range: Optional[float]
    energy_mean: Optional[float]
    zcr_mean: Optional[float]
    sc_mean: Optional[float]
    tempo_bpm_like: Optional[float]
    is_silent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "f0_med": self.f0_med,
            "f0_range": self.f0_range,
            "energy_mean": self.energy_mean,
            "zcr_mean": self.zcr_mean,
            "sc_mean": self.sc_mean,
            "tempo_bpm_like": self.tempo_bpm_like,
            "is_silent": self.is_silent,
        }

# ---------------------------
# 특징 추출
# ---------------------------
def _extract_features(file_path: str, target_sr: int = 16000) -> VoiceFeatures:
    """
    파일에서 음성 특징을 추출합니다. (내부 계산용 — 출력에 숫자를 노출하지 않음)
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    duration = _safe_float(librosa.get_duration(y=y, sr=sr), 6)

    # 무성 여부(안전장치)
    rms = librosa.feature.rms(y=y).flatten()
    is_silent = bool(np.mean(rms) < 1e-3)

    # F0 추정(YIN)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=1100, sr=sr)
    except Exception:
        f0 = np.array([np.nan])

    f0_med = _safe_float(_nan_robust(f0, np.nanmedian, default=np.nan), 2)
    f0_p95 = _nan_robust(f0, lambda a: np.nanpercentile(a, 95), default=np.nan)
    f0_p05 = _nan_robust(f0, lambda a: np.nanpercentile(a, 5), default=np.nan)
    f0_range = _safe_float((f0_p95 - f0_p05) if (f0_p95 is not None and f0_p05 is not None) else np.nan, 2)

    # 에너지(RMS 평균)
    energy_mean = _safe_float(float(np.mean(rms)) if len(rms) else np.nan, 6)

    # ZCR 평균
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    zcr_mean = _safe_float(float(np.mean(zcr)) if len(zcr) else np.nan, 6)

    # 스펙트럴 센트로이드 평균
    sc = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    sc_mean = _safe_float(float(np.mean(sc)) if len(sc) else np.nan, 2)

    # 템포 근사(bpm)
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr)
        tempo_bpm_like = _safe_float(float(tempo[0]) if tempo.size else np.nan, 2)
    except Exception:
        tempo_bpm_like = None

    return VoiceFeatures(
        duration_sec=duration,
        f0_med=f0_med,
        f0_range=f0_range,
        energy_mean=energy_mean,
        zcr_mean=zcr_mean,
        sc_mean=sc_mean,
        tempo_bpm_like=tempo_bpm_like,
        is_silent=is_silent
    )

# ---------------------------
# 성별/나이대 추정 (간단 휴리스틱)
# ---------------------------
def _gender_kor_from_features(feat: VoiceFeatures) -> str:
    f0 = feat.f0_med or 0
    if f0 >= 180:
        return "여"
    if f0 <= 140:
        return "남"
    return "남"

def _age_phrase_from_features(feat: VoiceFeatures) -> str:
    f0 = feat.f0_med or 0
    sc = feat.sc_mean or 0
    score = 0
    if f0 >= 250: score += 2
    elif f0 <= 120: score -= 2
    if sc >= 2800: score += 1
    elif sc <= 1600: score -= 1
    if score >= 2: return "10대~20대 초반"
    if score >= 0: return "20대"
    if score >= -1: return "30대"
    return "40대 이상"

# ---------------------------
# 숫자 → 쉬운 말 단서로 변환 (출력에는 숫자/전문용어 절대 금지)
# ---------------------------
def _describe_bins(features: VoiceFeatures) -> Dict[str, str]:
    """
    분석 숫자를 '쉬운 말' 단서로만 변환해서 LLM에 넘긴다.
    절대 숫자(Hz/BPM/dB 등)나 전문 용어(피치/스펙트럴/ZCR/RMS 등)를 쓰지 않는다.
    """
    f = features.to_dict()
    f0_med = f.get("f0_med") or 0.0
    f0_range = f.get("f0_range") or 0.0
    energy = f.get("energy_mean") or 0.0
    tempo  = f.get("tempo_bpm_like") or 0.0
    sc     = f.get("sc_mean") or 0.0
    zcr    = f.get("zcr_mean") or 0.0

    # 톤(대략적)
    if f0_med <= 140: tone = "중저음"
    elif f0_med >= 200: tone = "밝은 고음"
    else: tone = "중간 음역"

    # 음높이 변화(폭)
    if f0_range >= 180: variety = "넓음"
    elif f0_range <= 60: variety = "좁음"
    else: variety = "보통"

    # 힘/세기 느낌
    if energy is None:
        power = "보통"
    elif energy >= 0.04: power = "탄탄함"
    elif energy <= 0.02: power = "약함"
    else: power = "보통"

    # 속도/리듬
    if tempo is None: pace = "적당"
    elif tempo >= 120: pace = "빠른 편"
    elif tempo <= 90: pace = "느린 편"
    else: pace = "적당"

    # 밝기(음색의 따뜻/밝음 느낌)
    if sc >= 2500: brightness = "밝은 편"
    elif sc <= 1600: brightness = "따뜻한 편"
    else: brightness = "중간"

    # 명료도(거칠/호흡 섞임 정도 → 매우 러프)
    if zcr >= 0.12: clarity = "조금 거친 느낌"
    elif zcr <= 0.06: clarity = "또렷한 편"
    else: clarity = "괜찮은 편"

    return {
        "tone": tone,
        "variety": variety,
        "power": power,
        "pace": pace,
        "brightness": brightness,
        "clarity": clarity,
    }

# ---------------------------
# LLM 설명 생성 (동물의 숲 문구 제거)
# ---------------------------
def _llm_describe_voice(features: VoiceFeatures) -> str:
    # ✅ 전역 상태 먼저 선언
    global client, OPENAI_API_KEY, _last_llm_status

    # 상태 초기화
    _last_llm_status = {
        "used": False,
        "ok": False,
        "error": None,
        "model": OPENAI_MODEL,
        "has_api_key": bool(OPENAI_API_KEY),
    }

    # 🔧 런타임 재초기화: .env/.env.local/.env.sample 재시도
    if client is None and not OPENAI_API_KEY:
        for name in (".env", ".env.local", ".env.sample"):
            load_dotenv(BASE_DIR / name, override=True)
        OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
        _last_llm_status["has_api_key"] = bool(OPENAI_API_KEY)
        if OPENAI_API_KEY:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                _last_llm_status.update({"error": f"client_init_failed: {e}"})

    # 무성/짧은 입력
    if features.is_silent or (features.duration_sec or 0) < 0.6:
        return (
            "소리가 거의 없어서 뭐라 말하기가 어렵네. 주변 소음을 조금 줄이고, "
            "조금만 더 길게 말해주면 느낌을 더 잘 잡아볼게."
        )

    # 성별/나이대/톤
    gender = _gender_kor_from_features(features)
    gender_word = "여자" if gender == "여" else "남자"
    age_phrase = _age_phrase_from_features(features)
    bins = _describe_bins(features)

    # 키 없거나 client 실패 → 폴백 (ACNH 문장 없이)
    if client is None:
        return (
            f"{age_phrase} {gender_word} 목소리 같고, {bins['tone']}이라 안정적으로 들려. "
            "음높이 변화가 넓어서 말에 생동감이 느껴지고, 리듬은 적당해서 흐름이 매끄러워. "
            "가끔 힘이 살짝 빠져 보일 때가 있지만 전체적으로는 또렷하고 이해하기 쉬워."
        )

    system_msg = (
        "너는 '음성 평가 전문가'지만 말투는 친구처럼 편한 반말로 해.\n"
        "반드시 지켜야 할 규칙:\n"
        "1) 출력은 3~5문장으로.\n"
        "2) 장점/단점 균형 있게, 조언·지시는 금지.\n"
        "3) 숫자·단위·전문 용어 절대 금지.\n"
        "4) 첫 문장은 성별/나이대 추측으로 시작.\n"
        "5) 동물의 숲, 캐릭터화 등은 언급하지 마. 오직 목소리 느낌만 말해.\n"
    )

    user_msg = f"""
- 성별 힌트: {gender_word}
- 나이대 힌트: {age_phrase}
- 톤: {bins['tone']}
- 음높이 변화: {bins['variety']}
- 힘: {bins['power']}
- 속도: {bins['pace']}
- 밝기: {bins['brightness']}
- 명료도: {bins['clarity']}
"""

    try:
        _last_llm_status["used"] = True
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.9,
            max_tokens=240,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (res.choices[0].message.content or "").strip()
        _last_llm_status.update({"ok": True})
        return text
    except Exception as e:
        _last_llm_status.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        print(f"[analysis.py] LLM call failed: {type(e).__name__}: {e}")
        return (
            f"{age_phrase} {gender_word} 목소리 같고, {bins['tone']}이라 안정적으로 들려. "
            "음높이 변화가 넓어서 말에 생동감이 느껴지고, 리듬은 적당해서 흐름이 매끄러워. "
            "가끔 힘이 살짝 빠져 보일 때가 있지만 전체적으로는 또렷하고 이해하기 쉬워."
        )

# ---------------------------
# 이미지 프롬프트(사람 이미지, 네가 준 문구 그대로)
# ---------------------------
def _build_dynamic_visual_fields(features: VoiceFeatures) -> Dict[str, Any]:
    """
    목소리 특징을 기반으로 캐릭터 속성을 동적으로 반영한 en_prompt 생성
    """
    bins = _describe_bins(features)
    
    # 기본 캐릭터 프롬프트 (고정 요소)
    base_prompt = (
        "high quality, stylized semi-realistic character art, "
        "soft watercolor tones, painterly texture, "
        "storybook character design, warm and charming mood, "
        "simple plain background"
    )

    dynamic_attrs = ""
    if client:
        # LLM에게 목소리 기반 속성 추천 요청
        try:
            user_msg = f"""
            이 목소리를 가진 사람을 그릴 때 어울리는 캐릭터 속성을 영어로 추천해줘.
            - 톤: {bins['tone']}
            - 음높이 변화: {bins['variety']}
            - 힘/세기: {bins['power']}
            - 속도/리듬: {bins['pace']}
            - 밝기: {bins['brightness']}
            - 명료도: {bins['clarity']}
            속성에는 머리 색, 눈 색, 머리스타일, 옷 스타일, 표정 등을 포함하고 1문장으로.
            """
            res = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.7,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": "너는 캐릭터 아트 전문가야. 영어로 1문장만 추천해."},
                    {"role": "user", "content": user_msg},
                ],
            )
            dynamic_attrs = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[analysis.py] Dynamic attribute generation failed: {e}")

    # 최종 en_prompt
    en_prompt = f"{base_prompt}, {dynamic_attrs}" if dynamic_attrs else base_prompt

    return {
        "en_prompt": en_prompt,
        "negative": "text, watermark, logo, extra limbs, distorted anatomy, 3d render",
        "style_tags": ["semi-realistic", "watercolor", "storybook"],
        "palette": ["#F2E5F2", "#FDE2E4", "#E2F0FE", "#FFF4CC"],
        "seed": 3501657520,
    }




# ---------------------------
# (선택) 아바타 힌트 — en_prompt에는 더 이상 붙이지 않음
# ---------------------------
def _avatar_suggestions(features: VoiceFeatures) -> Dict[str, Any]:
    f = features.to_dict()
    f0_med = f.get("f0_med") or 0.0
    energy = f.get("energy_mean") or 0.0
    tempo  = f.get("tempo_bpm_like") or 0.0

    gender_guess = "female" if f0_med >= 180 else "male"
    vibe = "lively" if tempo and tempo >= 110 else "calm"
    hint = f"{gender_guess} portrait, {vibe} mood"
    if energy and energy >= 0.04:
        hint += ", confident posture"
    return {"avatar_hint_text": hint}

# ---------------------------
# WebUI 이미지 생성
# ---------------------------
def generate_image(prompt: str, negative_prompt: str = "", width: int = 512, height: int = 512, steps: int = 20, seed: int = None):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed or -1,
        "sampler_name": "Euler a",
        "cfg_scale": 7.0,
    }
    WEBUI_URL = "http://127.0.0.1:7860"
    resp = requests.post(f"{WEBUI_URL}/sdapi/v1/txt2img", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["images"][0]  # Base64

# ---------------------------
# 메인 함수
# ---------------------------
def analyze_voice(file_path: str, generate_img: bool = False) -> Dict[str, Any]:
    """
    음성 파일을 분석하고:
    - 특징 추출
    - LLM 설명 생성
    - 이미지 프롬프트/아바타 힌트
    결과를 딕셔너리로 반환
    """
    # 1️⃣ 특징 추출
    features = _extract_features(file_path)

    # 2️⃣ LLM 설명 생성
    description_ko = _llm_describe_voice(features)

    # 3️⃣ 이미지 프롬프트/아바타 힌트
    visual = _build_dynamic_visual_fields(features)
    avatar = _avatar_suggestions(features)
    en_prompt = visual["en_prompt"]

    # 4️⃣ WebUI 이미지 생성 (선택)
    img_base64 = None
    if generate_img:
        try:
            img_base64 = generate_image(
                prompt=en_prompt,
                negative_prompt=visual.get("negative", ""),
                width=512,
                height=512,
                steps=25,
                seed=visual.get("seed")
            )
        except Exception as e:
            print(f"[analysis.py] Image generation failed: {e}")

    # 5️⃣ 결과 반환
    return {
        "features": features.to_dict(),
        "description_ko": description_ko,   # ✅ LLM 결과
        "en_prompt": en_prompt,
        "avatar": avatar,
        "image_base64": img_base64,
        "llm_status": _last_llm_status,     # ✅ 프론트에서 상태 확인용
        **visual,                           # ⚠ style_tags, palette 등도 포함
    }



