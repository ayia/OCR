import base64
import json
import re
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 Mo

def is_allowed_image_magic(data: bytes):
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None

class VisionAnalyticsController:
    @staticmethod
    async def extract_info(prompt_text: str, image: UploadFile, ollama_url: str, ollama_model: str):
        image_data = await image.read()
        if len(image_data) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="La taille du fichier dépasse 10 Mo")
        mime_type = is_allowed_image_magic(image_data)
        if not mime_type:
            raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG, PNG, WEBP)")
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur d'encodage de l'image: {str(e)}")
        payload = {
            "model": ollama_model,
            "prompt": prompt_text,
            "images": [image_base64],
            "stream": False
        }
        print(f"[LOG] Prompt envoyé à Ollama (/extract-info) : {prompt_text}")
        import requests
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

    @staticmethod
    async def face_similarity(image1: UploadFile, image2: UploadFile, ollama_url: str, ollama_model: str, prompt_text: str):
        images_data = []
        for img in (image1, image2):
            data = await img.read()
            if len(data) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail="La taille d'une des images dépasse 10 Mo")
            mime_type = is_allowed_image_magic(data)
            if not mime_type:
                raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG, PNG, WEBP)")
            images_data.append(base64.b64encode(data).decode('utf-8'))
        payload = {
            "model": ollama_model,
            "prompt": prompt_text,
            "images": images_data,
            "stream": False
        }
        print(f"[LOG] Prompt envoyé à Ollama (/face-similarity) : {prompt_text}")
        import requests
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

router = APIRouter(prefix="/vision_analytics", tags=["Vision Analytics"])

@router.post("/extract-info", summary="Extract and validate information from an identity document image using OCR and return formatted JSON.")
async def extract_info(
    prompt_text: str = Form(
        'Analysez minutieusement l\'image de la Carte Nationale d\'Identité Marocaine et extrayez les informations suivantes avec une précision maximale. Structurez les résultats dans un JSON strictement conforme à ce schéma : { "type": "Carte Nationale d\'Identité Marocaine", // Toujours cette valeur exacte "numero": "U123456", // Format: Lettre majuscule + 6 chiffres "nom": "ENNAJI", // En majuscules, accents autorisés "prenom": "Mehdi", // Format titre, traits d\'union permis "date_naissance": "1995-07-23", // Convertir depuis tout format source "lieu_naissance": "Lyon", // Respecter la casse originale "date_emission": "2022-04-15", // Chercher dans la zone MRZ ou corps du document "date_expiration": "2032-04-15", // Vérifier la cohérence avec la durée de validité "pays": "France", // Le pays d ou la carte a ete emise "langues": ["arabe", "français"] // langues present sur la carte } Directives critiques : 1. Extraction optique : - Analyser séparément zone MRZ, texte imprimé et hologrammes - Vérifier les chiffres de contrôle dans la zone MRZ - Différencier clairement 0/O et 1/I 2. Traitement des dates : - Convertir immédiatement toute date lue en ISO 8601 - Gérer les formats jour/mois/année et année-mois-jour - Pour les dates en lettres (ex: "15 Avril 2022"), convertir en numérique 3. Validation : - Numéro CIN : Valider le pattern [A-Z]\\d{6} - Cohérence temporelle : date_emission < date_expiration - Champs obligatoires : tous sauf \'langues\' si non détectables 4. Gestion d\'erreurs : - Si information illisible : null + commentaire en JSON comment - En cas de contradiction entre zones : priorité MRZ > texte imprimé > hologramme Format de sortie exigé : - JSON minifié sans espaces inutiles - Encodage UTF-8 - Chaînes entre guillemets doubles uniquement - Valeurs null autorisées uniquement pour les champs manquants obligatoires',
        description="Prompt détaillé pour analyser l'image"
    ),
    image: UploadFile = File(..., description="Image du document (JPEG, PNG, WEBP, max 10 Mo)")
):
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        ollama_url = config["ollama_url"]
        ollama_model = config["ollama_model"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier config.json : {str(e)}")
    return await VisionAnalyticsController.extract_info(prompt_text, image, ollama_url, ollama_model)

@router.post("/face-similarity", summary="Analyze two facial images and return a similarity score as JSON.")
async def face_similarity(
    image1: UploadFile = File(..., description="First facial image (JPEG, PNG, WEBP, max 10 MB)"),
    image2: UploadFile = File(..., description="Second facial image (JPEG, PNG, WEBP, max 10 MB)")
):
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        ollama_url = config["ollama_url"]
        ollama_model = config["ollama_model"]
        prompt_text = config["face_similarity_prompt"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier config.json : {str(e)}")
    return await VisionAnalyticsController.face_similarity(image1, image2, ollama_url, ollama_model, prompt_text) 