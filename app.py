from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_analytics_controller import router
from error_handlers import register_error_handlers

app = FastAPI(title="Document Understanding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
register_error_handlers(app) 