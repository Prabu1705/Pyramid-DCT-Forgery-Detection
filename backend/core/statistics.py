"""
core/statistics.py
------------------
Statistical anomaly detection:
- Chi-square test across block distributions
- Anomaly score computation
- Binary mask thresholding
"""

import numpy as np
from scipy.stats import chi2_contingency
from typing import Tuple, Dict, Any


def chi_square_anomaly_map(
    feature_map: np.ndarray,
    n_bins: int = 16,
    patch_size: int = 16
) -> np.ndarray:
    """
    Compute a local chi-square anomaly score by comparing each local patch's
    feature distribution against the global distribution.

    Args:
        feature_map: 2D normalized feature map in [0, 1]
        n_bins: number of histogram bins
        patch_size: size of local comparison window

    Returns:
        chi2_map: 2D anomaly score map (same shape as feature_map)
    """
    h, w = feature_map.shape
    chi2_map = np.zeros((h, w), dtype=np.float64)

    # Global histogram (reference distribution)
    global_hist, bin_edges = np.histogram(feature_map.flatten(), bins=n_bins, range=(0, 1))
    global_hist = global_hist + 1  # Laplace smoothing

    half = patch_size // 2
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            r0, r1 = max(0, r - half), min(h, r + patch_size + half)
            c0, c1 = max(0, c - half), min(w, c + patch_size + half)
            patch = feature_map[r0:r1, c0:c1].flatten()

            local_hist, _ = np.histogram(patch, bins=bin_edges)
            local_hist = local_hist + 1  # Laplace smoothing

            # Chi-square statistic: sum((O-E)^2 / E)
            expected = global_hist * (local_hist.sum() / global_hist.sum())
            chi2_stat = float(np.sum((local_hist - expected) ** 2 / (expected + 1e-9)))

            chi2_map[r:r + patch_size, c:c + patch_size] = chi2_stat

    # Normalize to [0, 1]
    if chi2_map.max() > 0:
        chi2_map = chi2_map / chi2_map.max()

    return chi2_map


def combine_anomaly_scores(
    kurtosis_map: np.ndarray,
    entropy_map: np.ndarray,
    variance_map: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
) -> np.ndarray:
    """
    Combine multi-feature anomaly scores into a single score map.

    Kurtosis is weighted highest as it best reflects DCT coefficient
    distribution abnormalities indicative of forgery.

    Args:
        kurtosis_map: normalized kurtosis anomaly map
        entropy_map:  normalized entropy anomaly map
        variance_map: normalized variance anomaly map
        weights: (w_kurt, w_entr, w_var) summing to 1.0

    Returns:
        combined: 2D score map in [0, 1]
    """
    wk, we, wv = weights
    combined = wk * kurtosis_map + we * entropy_map + wv * variance_map
    combined = np.clip(combined, 0, 1)
    return combined


def apply_chi_square_refinement(
    score_map: np.ndarray,
    kurtosis_map: np.ndarray
) -> np.ndarray:
    """
    Refine the combined score using chi-square analysis on the kurtosis map.
    This helps highlight regions with statistically unusual DCT distributions.
    """
    chi2_map = chi_square_anomaly_map(kurtosis_map)
    refined = 0.6 * score_map + 0.4 * chi2_map
    return np.clip(refined, 0, 1)


def generate_binary_mask(
    score_map: np.ndarray,
    threshold: float = None
) -> Tuple[np.ndarray, float]:
    """
    Generate a binary forgery mask using adaptive Otsu-like thresholding.

    Args:
        score_map: 2D anomaly score map in [0, 1]
        threshold: manual threshold; if None, uses mean + 1.5 * std

    Returns:
        (mask, threshold_used)
        mask: binary uint8 array (0 or 255)
    """
    if threshold is None:
        mu = float(np.mean(score_map))
        sigma = float(np.std(score_map))
        threshold = min(mu + 1.5 * sigma, 0.85)

    mask = (score_map >= threshold).astype(np.uint8) * 255
    return mask, threshold


def compute_global_statistics(
    score_map: np.ndarray,
    mask: np.ndarray,
    level_stats: list
) -> Dict[str, Any]:
    """
    Compile global detection statistics for API response.

    Returns:
        dict with detection summary and per-level stats
    """
    mask_bool = mask > 0
    forgery_fraction = float(mask_bool.mean())

    # Classify overall risk
    if forgery_fraction < 0.02:
        verdict = "Likely Authentic"
        confidence = "High"
    elif forgery_fraction < 0.08:
        verdict = "Suspicious"
        confidence = "Medium"
    elif forgery_fraction < 0.20:
        verdict = "Likely Tampered"
        confidence = "High"
    else:
        verdict = "Heavily Tampered / AI-Generated"
        confidence = "High"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "forgery_fraction": round(forgery_fraction * 100, 2),
        "anomaly_score_mean": round(float(np.mean(score_map)), 4),
        "anomaly_score_max":  round(float(np.max(score_map)), 4),
        "anomaly_score_std":  round(float(np.std(score_map)), 4),
        "tampered_pixels": int(mask_bool.sum()),
        "total_pixels": int(mask.size),
        "pyramid_levels": level_stats,
    }
