"""
core/pyramid.py
---------------
Gaussian pyramid construction and multi-scale DCT analysis.
Builds 3–5 pyramid levels and aggregates feature maps across scales.
"""

import numpy as np
import cv2
from typing import List, Tuple
from .dct_features import compute_feature_maps, normalize_map


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Build a Gaussian (downsampling) pyramid.

    Args:
        image: 2D grayscale float image
        levels: number of pyramid levels (including original)

    Returns:
        List of images from finest (original) to coarsest
    """
    pyramid = [image.copy()]
    current = image.copy()
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def analyze_pyramid(
    image: np.ndarray,
    levels: int = 4,
    block_size: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Perform multi-scale DCT feature extraction using Gaussian pyramid.

    For each level:
      1. Build feature maps (kurtosis, entropy, variance)
      2. Normalize each map
      3. Upsample back to original resolution
      4. Accumulate weighted sum (finer scales weighted more)

    Args:
        image: 2D grayscale uint8 image (H x W)
        levels: number of pyramid levels
        block_size: DCT block size

    Returns:
        (kurtosis_agg, entropy_agg, variance_agg, level_stats)
        Each aggregate map has shape (H, W) in [0, 1]
        level_stats: list of per-level statistics dicts
    """
    h, w = image.shape
    # Ensure divisible by block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    image_cropped = image[:h_crop, :w_crop]

    pyramid = build_gaussian_pyramid(image_cropped.astype(np.float64), levels)

    agg_kurt = np.zeros((h_crop, w_crop), dtype=np.float64)
    agg_entr = np.zeros((h_crop, w_crop), dtype=np.float64)
    agg_var  = np.zeros((h_crop, w_crop), dtype=np.float64)

    level_stats = []
    weight_sum = 0.0

    for lvl_idx, lvl_img in enumerate(pyramid):
        lh, lw = lvl_img.shape
        # Skip if too small for even one block
        if lh < block_size or lw < block_size:
            continue

        # Compute DCT feature maps at this scale
        k_map, e_map, v_map = compute_feature_maps(lvl_img, block_size)

        # Normalize
        k_norm = normalize_map(k_map)
        e_norm = normalize_map(e_map)
        v_norm = normalize_map(v_map)

        # Collect per-level statistics
        level_stats.append({
            "level": lvl_idx,
            "resolution": f"{lw}x{lh}",
            "kurtosis_mean": float(np.mean(k_map)),
            "kurtosis_std":  float(np.std(k_map)),
            "entropy_mean":  float(np.mean(e_map)),
            "entropy_std":   float(np.std(e_map)),
            "variance_mean": float(np.mean(v_map)),
            "variance_std":  float(np.std(v_map)),
        })

        # Upsample feature maps to block-grid resolution, then to full image
        # Feature map -> block-scale image
        def upsample_feature(feat_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
            feat_img = feat_map.astype(np.float32)
            return cv2.resize(feat_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        weight = 1.0 / (2 ** lvl_idx)  # coarser levels have less weight
        agg_kurt += weight * upsample_feature(k_norm, h_crop, w_crop)
        agg_entr += weight * upsample_feature(e_norm, h_crop, w_crop)
        agg_var  += weight * upsample_feature(v_norm, h_crop, w_crop)
        weight_sum += weight

    # Normalize aggregated maps by total weight
    if weight_sum > 0:
        agg_kurt /= weight_sum
        agg_entr /= weight_sum
        agg_var  /= weight_sum

    # Pad back to original size if needed
    def pad_to_original(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        out = np.zeros((orig_h, orig_w), dtype=np.float64)
        out[:arr.shape[0], :arr.shape[1]] = arr
        return out

    agg_kurt = pad_to_original(agg_kurt, h, w)
    agg_entr = pad_to_original(agg_entr, h, w)
    agg_var  = pad_to_original(agg_var,  h, w)

    return agg_kurt, agg_entr, agg_var, level_stats
