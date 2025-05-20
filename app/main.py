from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.ocr_controller import router as ocr_router

# Création de l'application FastAPI
app = FastAPI(
    title="API OCR",
    description="API RESTful avec documentation Swagger",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du router OCR
app.include_router(ocr_router)

@app.get("/")
async def root():
    """
    Page d'accueil de l'API
    """
    return {
        "message": "Bienvenue sur l'API OCR",
        "documentation": "/docs pour la documentation Swagger"
    } 