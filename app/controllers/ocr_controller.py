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
from urllib.parse import unquote

# Configuration par défaut (modifiable via variables d'environnement)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
    responses={404: {"description": "Not found"}}
)

class OCRSummaryRequest(BaseModel):
    model: str = "ishumilin/deepseek-r1-coder-tools:14b"
    ollama_url: str = "http://192.168.88.164"

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

async def summarize_with_ollama(text: str, model: str, ollama_url: str) -> (Any, dict):
    # Nettoyage du nom du modèle
    model_clean = unquote(model).strip().replace(' ', '')
    prompt = (
        "Tu es un expert en extraction d'informations de documents d'identité marocains. "
        "Voici un texte OCR extrait d'une carte d'identité marocaine. "
        "Ta tâche : Extrais les informations principales et structure-les dans un tableau JSON de deux objets. "
        "Premier objet (français) : pays, type_document, nom, prenom, date_naissance, lieu_naissance, autorite_emission, nom_autorite, numero_carte, numero_identification, date_expiration. "
        "Second objet (arabe) : البلد, نوع_المستند, الاسم, اللقب, تاريخ_الميلاد, مكان_الميلاد, السلطة_المصدرة, اسم_المسؤول, رقم_البطاقة, رقم_التعريف, تاريخ_الانتهاء. "
        "Respecte l'ordre des champs. Si une information est absente, mets une chaîne vide. "
        "Réponds STRICTEMENT par ce tableau JSON, sans explication, sans balise, sans markdown, sans texte autour. "
        "La réponse doit commencer par [ et finir par ]. "
        "N'inclus aucun retour à la ligne, ni indentation, ni espace inutile. "
        "Exemple de réponse attendue : [{\"pays\":\"...\",...},{\"البلد\":\"...\",...}] "
        f"\n\nTexte OCR : {text}"
    )
    print(f"PROMPT ENVOYE AU MODELE OLLAMA :\n{prompt}\n")
    payload = {
        "model": model_clean,
        "prompt": prompt,
        "stream": False
    }
    try:
        # S'assurer que l'URL ne finit pas par un slash
        base_url = ollama_url.rstrip('/')
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{base_url}/api/generate", json=payload)
            r.raise_for_status()
            response = r.json()
            summary_text = response.get("response", str(response))
    except Exception as e:
        print(f"[ERREUR] Appel Ollama: {e}")
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {e}")
    # Nettoyage du JSON pour enlever les \n, \ et espaces inutiles
    match = re.search(r'\[.*\]', summary_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Nettoyer les \n, \ et espaces inutiles
            json_str_clean = json_str.replace('\\n', '').replace('\\', '').replace('\n', '').replace('\r', '').strip()
            summary_json = json.loads(json_str_clean)
        except Exception:
            summary_json = json_str
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
    prompt_envoye = (
        "Voici un texte extrait d'une image de carte d'identité marocaine. "
        "Extrais-moi les informations principales sous forme d'un tableau JSON contenant deux objets : "
        "le premier en français avec les champs : pays, type_document, nom, prenom, date_naissance, lieu_naissance, autorite_emission, nom_autorite, numero_carte, numero_identification, date_expiration ; "
        "le second en arabe avec les champs : البلد, نوع_المستند, الاسم, اللقب, تاريخ_الميلاد, مكان_الميلاد, السلطة_المصدرة, اسم_المسؤول, رقم_البطاقة, رقم_التعريف, تاريخ_الانتهاء. "
        "Réponds STRICTEMENT par ce tableau JSON, sans aucune explication, sans balise, sans markdown, sans texte autour. "
        "Ta réponse doit commencer par [ et finir par ].\n\n"
        f"Texte extrait :\n{combined}"
    )
    summary, _ = await summarize_with_ollama(combined, config.model, config.ollama_url)
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except Exception:
            pass
    return JSONResponse(content={"summary": summary, "prompt_envoye": prompt_envoye}) 