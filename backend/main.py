"""
backend/main.py
---------------
FastAPI application entry point.
Mounts the frontend static files and includes API router.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router

# Directories
BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR   = BASE_DIR / "uploads"
RESULTS_DIR  = BASE_DIR / "results"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Pyramid DCT Forgery Detection System",
    description=(
        "Multi-scale image forgery detection using DCT-based frequency analysis "
        "and statistical features. Detects tampered regions, double compression, "
        "and AI-generated artifacts."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS (allow frontend dev server access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes at /api
app.include_router(api_router, prefix="/api")

# Serve result images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
