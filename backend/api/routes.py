"""
backend/api/routes.py
---------------------
FastAPI router with /upload and /analyze endpoints.
"""

import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, Response

from ..core.pipeline import run_forgery_detection

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forgery Detection"])

# Directories relative to backend package
BASE_DIR    = Path(__file__).parent.parent.parent
UPLOAD_DIR  = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Pyramid DCT Forgery Detection"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for analysis.

    Returns:
        JSON with `image_id` and `filename` for use with /analyze.
    """
    _validate_extension(file.filename)

    image_id = str(uuid.uuid4())
    suffix   = Path(file.filename).suffix.lower()
    dest     = UPLOAD_DIR / f"{image_id}{suffix}"

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    finally:
        await file.close()

    file_size = dest.stat().st_size
    logger.info(f"Uploaded {file.filename} → {dest} ({file_size} bytes)")

    return {
        "image_id": image_id,
        "filename": file.filename,
        "stored_as": dest.name,
        "size_bytes": file_size,
        "message": "Upload successful. Call /api/analyze with this image_id.",
    }


@router.post("/analyze")
async def analyze_image(
    image_id: str = Form(...),
    pyramid_levels: int = Form(4),
    threshold: Optional[float] = Form(None),
):
    """
    Analyze an uploaded image for forgery indicators.

    Args:
        image_id: ID returned by /upload
        pyramid_levels: Gaussian pyramid depth (3–5, default 4)
        threshold: binary mask threshold (0–1); None = adaptive

    Returns:
        JSON with statistics + URLs for heatmap/mask images.
    """
    # Validate params
    if not (2 <= pyramid_levels <= 5):
        raise HTTPException(status_code=400, detail="pyramid_levels must be 2–5")
    if threshold is not None and not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be in (0, 1)")

    # Find uploaded image
    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No image found for id={image_id}")
    image_path = str(matches[0])

    # Run pipeline
    try:
        results = run_forgery_detection(
            image_path=image_path,
            pyramid_levels=pyramid_levels,
            threshold=threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Save output images
    result_id = str(uuid.uuid4())[:8]

    def save(key: str, suffix: str) -> str:
        fname = f"{result_id}_{suffix}.png"
        fpath = RESULTS_DIR / fname
        fpath.write_bytes(results[key])
        return f"/results/{fname}"

    heatmap_url   = save("heatmap_overlay_bytes", "heatmap")
    mask_url      = save("mask_overlay_bytes",    "mask")
    heatmap_p_url = save("heatmap_pure_bytes",    "heatmap_pure")
    composite_url = save("side_by_side_bytes",    "composite")

    return JSONResponse({
        "image_id":      image_id,
        "result_id":     result_id,
        "statistics":    results["statistics"],
        "output_images": {
            "heatmap_overlay":  heatmap_url,
            "mask_overlay":     mask_url,
            "heatmap_pure":     heatmap_p_url,
            "composite":        composite_url,
        },
    })


@router.get("/result/{result_id}/{image_type}")
async def get_result_image(result_id: str, image_type: str):
    """
    Serve a result image directly as PNG bytes.
    image_type: heatmap | mask | heatmap_pure | composite
    """
    fname = f"{result_id}_{image_type}.png"
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Result image not found")
    return Response(content=fpath.read_bytes(), media_type="image/png")
