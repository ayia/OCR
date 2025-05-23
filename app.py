from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import json
import re

app = FastAPI(title="Ma_CIN_frontçdata API")

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

@app.post("/Ma_CIN_frontçdata", summary="Envoie une image et un prompt à Ollama et retourne la réponse brute.")
async def ma_cin_frontdata(
    ollama_url: str = Form('http://192.168.88.164', description="URL du serveur Ollama (ex: http://localhost:11434)"),
    ollama_model: str = Form('qwen2.5vl:32b', description="Nom du modèle Ollama (ex: llava)"),
    prompt_text: str = Form(
        "Analysez minutieusement l'image de la Carte Nationale d'Identité Marocaine et extrayez les informations suivantes avec une précision maximale. Structurez les résultats dans un JSON strictement conforme à ce schéma : {   \"type\": \"Carte Nationale d'Identité Marocaine\",  // Toujours cette valeur exacte   \"numero\": \"U123456\",  // Format: Lettre majuscule + 6 chiffres   \"nom\": \"ENNAJI\",      // En majuscules, accents autorisés   \"prenom\": \"Mehdi\",    // Format titre, traits d'union permis   \"date_naissance\": \"1995-07-23\",  // Convertir depuis tout format source   \"lieu_naissance\": \"Lyon\",  // Respecter la casse originale   \"date_emission\": \"2022-04-15\",   // Chercher dans la zone MRZ ou corps du document   \"date_expiration\": \"2032-04-15\", // Vérifier la cohérence avec la durée de validité   \"pays\": \"France\",      // Le pays d ou la carte a ete emise   \"langues\": [\"arabe\", \"français\"]  // langues present sur la carte } Directives critiques : 1. Extraction optique :    - Analyser séparément zone MRZ, texte imprimé et hologrammes    - Vérifier les chiffres de contrôle dans la zone MRZ    - Différencier clairement 0/O et 1/I 2. Traitement des dates :    - Convertir immédiatement toute date lue en ISO 8601    - Gérer les formats jour/mois/année et année-mois-jour    - Pour les dates en lettres (ex: \"15 Avril 2022\"), convertir en numérique 3. Validation :    - Numéro CIN : Valider le pattern [A-Z]\\d{6}    - Cohérence temporelle : date_emission < date_expiration    - Champs obligatoires : tous sauf 'langues' si non détectables 4. Gestion d'erreurs :    - Si information illisible : null + commentaire en JSON comment    - En cas de contradiction entre zones : priorité MRZ > texte imprimé > hologramme Format de sortie exigé : - JSON minifié sans espaces inutiles - Encodage UTF-8 - Chaînes entre guillemets doubles uniquement - Valeurs null autorisées uniquement pour les champs manquants obligatoires",
        description="Prompt détaillé pour analyser l'image"
    ),
    image: UploadFile = File(..., description="Image de la CIN (JPEG, PNG, WEBP, max 10 Mo)")
):
    # 1. Vérifier le type MIME par header magique et la taille
    image_data = await image.read()
    if len(image_data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="La taille du fichier dépasse 10 Mo")
    mime_type = is_allowed_image_magic(image_data)
    if not mime_type:
        raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG, PNG, WEBP)")

    # 2. Encoder l'image en base64
    try:
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'encodage de l'image: {str(e)}")

    # 3. Préparer la requête pour Ollama
    payload = {
        "model": ollama_model,
        "prompt": prompt_text,
        "images": [image_base64],
        "stream": False
    }

    # 4. Envoyer à Ollama et retourner la réponse brute
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        ollama_json = response.json()
        raw_response = ollama_json.get('response', '')
        # Extraction robuste du bloc JSON dans le markdown
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = raw_response.strip()
        try:
            data = json.loads(json_str)
            return data
        except Exception:
            # Si le parsing échoue, retourne le texte brut
            return {"response": raw_response}
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erreur Ollama: {str(e)}") 