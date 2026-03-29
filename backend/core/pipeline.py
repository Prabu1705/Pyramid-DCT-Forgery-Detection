"""
core/pipeline.py
----------------
Top-level orchestration of the full forgery detection pipeline.
Accepts a raw image path and returns all analysis artifacts.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple

from .pyramid import analyze_pyramid
from .statistics import (
    combine_anomaly_scores,
    apply_chi_square_refinement,
    generate_binary_mask,
    compute_global_statistics,
)
from .visualization import (
    score_map_to_heatmap,
    score_map_to_mask_overlay,
    render_matplotlib_heatmap,
    encode_image_to_bytes,
    generate_side_by_side,
)


def load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate image. Returns (bgr_image, gray_image).
    Raises ValueError for unsupported formats or load failures.
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


def preprocess_for_analysis(gray: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Optionally downsample very large images so block analysis is tractable.
    Maintains aspect ratio.
    """
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray


def run_forgery_detection(
    image_path: str,
    pyramid_levels: int = 4,
    block_size: int = 8,
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Load image
      2. Preprocess (resize if needed)
      3. Gaussian pyramid + DCT feature extraction
      4. Chi-square refinement
      5. Combine features into anomaly score map
      6. Generate binary mask
      7. Visualize: heatmap overlay, mask overlay, standalone heatmap
      8. Compile statistics

    Args:
        image_path: path to input image
        pyramid_levels: number of Gaussian pyramid levels (3–5)
        block_size: DCT block size (default 8)
        threshold: manual threshold for binary mask (None = adaptive)

    Returns:
        dict containing:
          - 'heatmap_overlay_bytes':  PNG bytes of heatmap overlaid on original
          - 'mask_overlay_bytes':     PNG bytes of mask overlaid on original
          - 'heatmap_pure_bytes':     PNG bytes of standalone matplotlib heatmap
          - 'side_by_side_bytes':     PNG bytes of composite comparison image
          - 'score_map':              2D float numpy array (anomaly scores)
          - 'mask':                   2D uint8 binary mask
          - 'threshold_used':         float threshold value
          - 'statistics':             dict of all detection statistics
    """
    # --- Load ---
    img_bgr, img_gray = load_image(image_path)
    orig_h, orig_w = img_gray.shape

    # --- Preprocess ---
    gray_proc = preprocess_for_analysis(img_gray.copy())

    # Also resize BGR for overlays
    proc_h, proc_w = gray_proc.shape
    if (proc_h, proc_w) != (orig_h, orig_w):
        bgr_proc = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        bgr_proc = img_bgr

    # --- Pyramid DCT Analysis ---
    kurt_map, entr_map, var_map, level_stats = analyze_pyramid(
        gray_proc, levels=pyramid_levels, block_size=block_size
    )

    # --- Combine Features ---
    score_map = combine_anomaly_scores(kurt_map, entr_map, var_map)

    # --- Chi-Square Refinement ---
    score_map = apply_chi_square_refinement(score_map, kurt_map)

    # --- Binary Mask ---
    mask, threshold_used = generate_binary_mask(score_map, threshold)

    # --- Statistics ---
    stats = compute_global_statistics(score_map, mask, level_stats)
    stats["image_resolution"] = f"{orig_w}x{orig_h}"
    stats["analysis_resolution"] = f"{proc_w}x{proc_h}"
    stats["pyramid_levels_used"] = pyramid_levels
    stats["block_size"] = block_size
    stats["threshold"] = round(threshold_used, 4)

    # --- Visualizations ---
    heatmap_overlay = score_map_to_heatmap(score_map, bgr_proc, alpha=0.55)
    mask_overlay    = score_map_to_mask_overlay(mask, bgr_proc, alpha=0.45)
    heatmap_pure    = render_matplotlib_heatmap(score_map, "DCT Pyramid Anomaly Score")
    side_by_side    = generate_side_by_side(bgr_proc, heatmap_overlay, mask_overlay)

    return {
        "heatmap_overlay_bytes": encode_image_to_bytes(heatmap_overlay),
        "mask_overlay_bytes":    encode_image_to_bytes(mask_overlay),
        "heatmap_pure_bytes":    encode_image_to_bytes(heatmap_pure),
        "side_by_side_bytes":    encode_image_to_bytes(side_by_side),
        "score_map":             score_map,
        "mask":                  mask,
        "threshold_used":        threshold_used,
        "statistics":            stats,
    }
