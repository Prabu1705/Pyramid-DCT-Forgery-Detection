<<<<<<< HEAD
# Pyramid DCT Forgery Detection System

A multi-scale image forgery detection system using DCT-based frequency analysis, Gaussian pyramid decomposition, and statistical anomaly detection.

---

## 🏗 Project Structure

```
pyramid_dct_forgery/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                    ← FastAPI app + static mounting
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              ← /upload, /analyze endpoints
│   └── core/
│       ├── __init__.py
│       ├── dct_features.py        ← 2D DCT, kurtosis/entropy/variance per block
│       ├── pyramid.py             ← Gaussian pyramid + multi-scale analysis
│       ├── statistics.py          ← Chi-square test, score combination, masking
│       ├── visualization.py       ← Heatmap overlay, mask overlay, matplotlib render
│       └── pipeline.py            ← End-to-end orchestration
│
├── frontend/
│   └── index.html                 ← Single-page UI (HTML + CSS + JS)
│
├── tests/
│   └── test_pipeline.py           ← pytest unit + integration tests
│
├── uploads/                       ← (auto-created) uploaded images
├── results/                       ← (auto-created) output PNGs
├── requirements.txt
├── run.py                         ← Server entry point
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd pyramid_dct_forgery
pip install -r requirements.txt
```

### 2. Launch Server

```bash
python run.py
# Dev mode with hot reload:
python run.py --reload
```

Server starts at **http://localhost:8000**

### 3. Open Web UI

Navigate to **http://localhost:8000** in your browser.

---

## 🧠 How It Works

### Pipeline Overview

```
Input Image (JPEG/PNG)
        │
        ▼
  Grayscale conversion
        │
        ▼
  Gaussian Pyramid (3–5 levels)
        │
  ┌─────┴──────┐
  │ Each Level │
  │            │
  │ 8×8 blocks │
  │ 2D DCT     │
  │            │
  │ Features:  │
  │ • Kurtosis │
  │ • Entropy  │
  │ • Variance │
  └─────┬──────┘
        │ (upsample + weighted combine)
        ▼
  Aggregate Feature Maps
        │
        ▼
  Chi-Square Anomaly Refinement
        │
        ▼
  Combined Anomaly Score Map [0,1]
        │
        ├──▶ Heatmap Overlay (OpenCV JET colormap)
        ├──▶ Binary Mask (adaptive threshold)
        ├──▶ Pure Heatmap (matplotlib inferno)
        └──▶ Composite Side-by-Side
```

### Feature Extraction (per 8×8 DCT block)

| Feature   | Description                              | Forgery Indicator |
|-----------|------------------------------------------|-------------------|
| Kurtosis  | Tail heaviness of DCT coefficient dist.  | Unusual peaks/flatness in tampering zones |
| Entropy   | Shannon entropy of DCT histogram         | Over/under-randomness in compressed artifacts |
| Variance  | Spread of DCT coefficients               | Inconsistent compression levels |

### Statistical Analysis

- **Chi-square test**: compares local patch histogram vs. global distribution
- **Weighted pyramid fusion**: finer scales weighted more heavily
- **Adaptive threshold**: mean + 1.5σ unless manually overridden
- **Verdict classification**: Authentic / Suspicious / Likely Tampered / Heavily Tampered

---

## 🔌 REST API

### `POST /api/upload`

Upload an image.

**Request:** `multipart/form-data` with `file` field.

**Response:**
```json
{
  "image_id": "uuid-string",
  "filename": "photo.jpg",
  "stored_as": "uuid.jpg",
  "size_bytes": 204800,
  "message": "Upload successful."
}
```

---

### `POST /api/analyze`

Analyze an uploaded image.

**Request:** `multipart/form-data`

| Field            | Type    | Default | Description                        |
|------------------|---------|---------|------------------------------------|
| `image_id`       | string  | required| ID from `/upload`                  |
| `pyramid_levels` | int     | 4       | Pyramid depth (2–5)                |
| `threshold`      | float   | null    | Binary mask threshold (0–1)        |

**Response:**
```json
{
  "image_id": "...",
  "result_id": "abc12345",
  "statistics": {
    "verdict": "Likely Tampered",
    "confidence": "High",
    "forgery_fraction": 12.45,
    "anomaly_score_mean": 0.3821,
    "anomaly_score_max": 0.9134,
    "anomaly_score_std": 0.1765,
    "tampered_pixels": 15982,
    "total_pixels": 131072,
    "image_resolution": "512x256",
    "pyramid_levels_used": 4,
    "block_size": 8,
    "threshold": 0.5293,
    "pyramid_levels": [...]
  },
  "output_images": {
    "heatmap_overlay":  "/results/abc12345_heatmap.png",
    "mask_overlay":     "/results/abc12345_mask.png",
    "heatmap_pure":     "/results/abc12345_heatmap_pure.png",
    "composite":        "/results/abc12345_composite.png"
  }
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

**Interactive docs:** http://localhost:8000/api/docs

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover:
- DCT 2D transform correctness
- Feature extraction (kurtosis, entropy, variance)
- Feature map shapes and normalization
- Pyramid construction and level statistics
- Chi-square anomaly computation
- Score combination and mask generation
- Visualization encoding
- Full pipeline integration (authentic + tampered)
- Edge cases (invalid path, JPEG input, custom threshold)

---

## 🛠 Configuration

| Parameter         | Default | Range | Effect |
|-------------------|---------|-------|--------|
| `pyramid_levels`  | 4       | 2–5   | More levels → finer multi-scale detection, slower |
| `threshold`       | adaptive| 0–1   | Lower → more sensitive, higher → fewer false positives |
| `block_size`      | 8       | 8     | DCT block size (JPEG standard) |
| `max_dim`         | 1024    | —     | Auto-downscale large images for speed |

---

## 🔍 What It Detects

- **JPEG double-compression artifacts** — re-saved regions show DCT coefficient anomalies
- **Copy-move forgery** — duplicated regions have statistically similar but misplaced DCT distributions
- **Splicing** — boundaries between pasted regions produce kurtosis spikes
- **AI-generated content** — GAN/diffusion artifacts create frequency patterns inconsistent with camera noise

---

## 📦 Dependencies

| Package                  | Use |
|--------------------------|-----|
| FastAPI + uvicorn        | REST API server |
| OpenCV (headless)        | Image I/O, colormap, blending |
| NumPy + SciPy            | DCT, statistics, kurtosis, entropy |
| matplotlib               | Publication-quality heatmap rendering |
| Pillow                   | Supplementary image format support |
| scikit-image             | Image processing utilities |
=======
# Pyramid-DCT-Forgery-Detection
>>>>>>> c8f625ffe0f125edc0fa3d02ef0825a119e5747f
