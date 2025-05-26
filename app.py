from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vision_analytics_controller import router

app = FastAPI(title="Document Understanding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router) 