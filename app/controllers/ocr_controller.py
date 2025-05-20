from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, Any
import httpx
import os
from paddleocr import PaddleOCR
import traceback
import numpy as np
import json
import re
from fastapi.responses import JSONResponse
import tempfile
import shutil
import ollama

# Configuration par défaut (modifiable via variables d'environnement)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
    responses={404: {"description": "Not found"}}
)

class OCRSummaryRequest(BaseModel):
    base_url: str
    model: str
    api_key: str

class OCRSummaryResponse(BaseModel):
    text_image1: str
    combined_text: str
    summary: Any  # Peut être un tableau JSON ou une chaîne
    ollama_model: str
    ollama_url: str
    ollama_payload: Dict[str, Any]
    debug: Optional[str] = None
    ocr_raw1: Optional[Any] = None

async def extract_text_from_image(image_file: UploadFile, debug_info: list) -> (str, Any):
    content = await image_file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    image_np = np.array(image)
    ocr = PaddleOCR(use_angle_cls=True, lang="fr")
    result = ocr.ocr(image_np, cls=True)
    debug_info.append(f"PaddleOCR result: {result}")
    text = " ".join([line[1][0] for line in result[0]]) if result and result[0] else ""
    debug_info.append(f"Texte extrait: '{text}'")
    return text.strip(), result

async def summarize_with_openai(text: str, base_url: str, model: str, api_key: str) -> (Any, dict):
    prompt = (
        "Voici un texte extrait d'une image de carte d'identité marocaine. "
        "Extrais-moi les informations principales sous forme d'un tableau JSON contenant deux objets : "
        "le premier en français avec les champs : pays, type_document, nom, prenom, date_naissance, lieu_naissance, autorite_emission, nom_autorite, numero_carte, numero_identification, date_expiration ; "
        "le second en arabe avec les champs : البلد, نوع_المستند, الاسم, اللقب, تاريخ_الميلاد, مكان_الميلاد, السلطة_المصدرة, اسم_المسؤول, رقم_البطاقة, رقم_التعريف, تاريخ_الانتهاء. "
        "Réponds STRICTEMENT par ce tableau JSON, sans aucune explication, sans balise, sans markdown, sans texte autour. "
        "Ta réponse doit commencer par [ et finir par ].\n\n"
        f"Texte extrait :\n{text}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Tu es un assistant OCR."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        summary_text = data["choices"][0]["message"]["content"]
        match = re.search(r'\[.*\]', summary_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                summary_json = json.loads(json_str)
            except Exception:
                summary_json = summary_text
        else:
            summary_json = summary_text
        return summary_json, payload

@router.post("/extractjson")
async def extract_json(
    image1: UploadFile = File(...),
    config: OCRSummaryRequest = Depends()
):
    debug_info = []
    if not image1.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    text1, _ = await extract_text_from_image(image1, debug_info)
    combined = text1
    summary, _ = await summarize_with_openai(combined, config.base_url, config.model, config.api_key)
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except Exception:
            pass
    return JSONResponse(content=summary) 