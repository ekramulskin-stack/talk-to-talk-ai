import io
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import edge_tts
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Talk to Talk AI")
STATIC_DIR = Path(__file__).parent / "static"

class TranslateRequest(BaseModel):
    speaker_name: str
    spoken_text: str
    target_lang: str = "bn"
    context: str = "general"
    use_cloud_ai: bool = False

LANG_LABELS = {"en": "English", "bn": "Bengali (বাংলা)", "hi": "Hindi (हिन्दी)", "mr": "Marathi (मराठी)", "kn": "Kannada (ಕನ್ನಡ)"}
NEURAL_VOICES = {"bn": "bn-IN-TanishaaNeural", "hi": "hi-IN-SwaraNeural", "mr": "mr-IN-AarohiNeural", "kn": "kn-IN-SapnaNeural", "en": "en-IN-NeerjaNeural"}

STRONG_HINGLISH = {"badhiya", "kya", "kaise", "kahan", "mujhe", "kitne", "chahiye", "hun", "hoon", "dhanyavad", "shukriya"}
WEAK_HINGLISH = {"main", "hai", "hain", "aap", "tum", "thik", "mera", "naam", "nahi", "haan"}
STRONG_BANGLISH = {"kemon", "achen", "bhalo", "koto", "kothay", "sahajyo", "dorkar", "kokhon", "asbe"}
WEAK_BANGLISH = {"ami", "tumi", "apni", "achi", "dam", "amar", "tomar", "lagbe"}

def check_api_keys() -> Dict[str, bool]:
    return {
        "openai_configured": len(os.getenv("OPENAI_API_KEY", "").strip()) > 10,
        "gemini_configured": len(os.getenv("GEMINI_API_KEY", "").strip()) > 10,
    }

def detect_language_mode(text: str) -> str:
    if re.search(r"[\u0980-\u09FF]", text): return "bn_native"
    if re.search(r"[\u0C80-\u0CFF]", text): return "kn_native"
    if re.search(r"[\u0900-\u097F]", text):
        return "mr_native" if any(w in text for w in ("आहे", "मला", "तुम्हाला", "कशी", "कोणत्या")) else "hi_native"
    tokens = set(re.sub(r"[^a-zA-Z\s]", "", text.lower()).split())
    if tokens.intersection(STRONG_HINGLISH) or len(tokens.intersection(WEAK_HINGLISH)) >= 2: return "hinglish"
    if tokens.intersection(STRONG_BANGLISH) or len(tokens.intersection(WEAK_BANGLISH)) >= 2: return "banglish"
    return "en_or_auto"

def format_detected_label(mode: str) -> str:
    return {"bn_native": "Bengali (বাংলা)", "kn_native": "Kannada (ಕನ್ನಡ)", "mr_native": "Marathi (मराठी)", "hi_native": "Hindi (हिन्दी)", "hinglish": "Hinglish", "banglish": "Banglish", "en_or_auto": "English / Auto"}.get(mode, "Auto")

def transliterate_roman(text: str, lang_code: str) -> str:
    itc = {"hi": "hi-t-i0-und", "bn": "bn-t-i0-und"}.get(lang_code)
    if not itc: return text
    try:
        url = f"https://inputtools.google.com/request?text={urllib.parse.quote(text)}&itc={itc}&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if payload[0] == "SUCCESS":
                return " ".join(item[1][0] if item[1] else item[0] for item in payload[1])
    except Exception:
        pass
    return text

def online_translate(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            parts = [seg[0] for seg in raw[0] if seg[0]]
            if parts: return "".join(parts)
    except Exception:
        pass
    return None

def execute_translation(text: str, target_lang: str, context: str) -> Dict[str, str]:
    mode = detect_language_mode(text)
    source_text, source_code = text, "auto"
    if mode == "hinglish":
        source_text = transliterate_roman(text, "hi")
        source_code = "hi"
        if target_lang == "hi": return {"detected": format_detected_label(mode), "translated": source_text}
    elif mode == "banglish":
        source_text = transliterate_roman(text, "bn")
        source_code = "bn"
        if target_lang == "bn": return {"detected": format_detected_label(mode), "translated": source_text}
    translated = online_translate(source_text, source_code, target_lang) or f"[{LANG_LABELS.get(target_lang, target_lang)}] {text}"
    return {"detected": format_detected_label(mode), "translated": translated}

@app.get("/")
async def serve_home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {"status": "healthy", "keys": check_api_keys()}

@app.get("/api/tts")
async def neural_tts(text: str, lang: str = "en"):
    clean_text = text.strip()[:250]
    if not clean_text: return Response(status_code=204)
    try:
        comm = edge_tts.Communicate(clean_text, NEURAL_VOICES.get(lang, "en-IN-NeerjaNeural"))
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio": buf.write(chunk["data"])
        if buf.tell() > 0: return Response(content=buf.getvalue(), media_type="audio/mpeg")
    except Exception:
        pass
    try:
        tts_url = f"https://translate.googleapis.com/translate_tts?ie=UTF-8&client=tw-ob&tl={lang}&q={urllib.parse.quote(clean_text[:180])}"
        req = urllib.request.Request(tts_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return Response(content=resp.read(), media_type="audio/mpeg")
    except Exception:
        return Response(status_code=204)

@app.post("/api/translate")
async def translate_api(req: TranslateRequest):
    keys = check_api_keys()
    key_notice = None
    if req.use_cloud_ai and not (keys["openai_configured"] or keys["gemini_configured"]):
        key_notice = "Cloud AI key not set (OPENAI_API_KEY / GEMINI_API_KEY missing). Using built-in multilingual engine."
    outcome = execute_translation(req.spoken_text, req.target_lang, req.context)
    return {
        "detected_language": outcome["detected"],
        "translated_text": outcome["translated"],
        "target_language_name": LANG_LABELS.get(req.target_lang, "English"),
        "context": req.context.title(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "key_notice": key_notice,
    }
