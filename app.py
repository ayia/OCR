from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import json
import re
import os

app = FastAPI(title="Document Understanding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 Mo

# Magic numbers for JPEG, PNG, WEBP
def is_allowed_image_magic(data: bytes):
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None

@app.post("/extract-info", summary="Extract and validate information from an identity document image using OCR and return formatted JSON.")
async def extract_info(
    image: UploadFile = File(..., description="Image du document (JPEG, PNG, WEBP, max 10 Mo)")
):
    # Lire la config
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        ollama_url = config["ollama_url"]
        ollama_model = config["ollama_model"]
        prompt_text = config["extract_info_prompt"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier config.json : {str(e)}")

    # Validation et lecture de l'image
    image_data = await image.read()
    if len(image_data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="La taille du fichier dépasse 10 Mo")
    mime_type = is_allowed_image_magic(image_data)
    if not mime_type:
        raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG, PNG, WEBP)")

    # Encoder l'image en base64
    try:
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'encodage de l'image: {str(e)}")

    # Préparer la requête pour Ollama
    payload = {
        "model": ollama_model,
        "prompt": prompt_text,
        "images": [image_base64],
        "stream": False
    }
    print(f"[LOG] Prompt envoyé à Ollama (/extract-info) : {prompt_text}")
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        ollama_json = response.json()
        raw_response = ollama_json.get('response', '')
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = raw_response.strip()
        try:
            data = json.loads(json_str)
            return data
        except Exception:
            return {"response": raw_response}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}")

@app.post("/face-similarity", summary="Analyze two facial images and return a similarity score as JSON.")
async def face_similarity(
    image1: UploadFile = File(..., description="First facial image (JPEG, PNG, WEBP, max 10 MB)"),
    image2: UploadFile = File(..., description="Second facial image (JPEG, PNG, WEBP, max 10 MB)")
):
    # Lire la config
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        ollama_url = config["ollama_url"]
        ollama_model = config["ollama_model"]
        prompt_text = config["face_similarity_prompt"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier config.json : {str(e)}")

    # Validation et lecture des deux images
    images_data = []
    for img in (image1, image2):
        data = await img.read()
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="La taille d'une des images dépasse 10 Mo")
        mime_type = is_allowed_image_magic(data)
        if not mime_type:
            raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG, PNG, WEBP)")
        images_data.append(base64.b64encode(data).decode('utf-8'))

    # Préparer la requête pour Ollama
    payload = {
        "model": ollama_model,
        "prompt": prompt_text,
        "images": images_data,
        "stream": False
    }
    print(f"[LOG] Prompt envoyé à Ollama (/face-similarity) : {prompt_text}")
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        ollama_json = response.json()
        raw_response = ollama_json.get('response', '')
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = raw_response.strip()
        try:
            data = json.loads(json_str)
            return data
        except Exception:
            return {"response": raw_response}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}") 