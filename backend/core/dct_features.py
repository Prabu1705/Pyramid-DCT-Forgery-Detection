"""
core/dct_features.py
--------------------
DCT-based feature extraction for image forgery detection.
Computes kurtosis, entropy, and variance on 8x8 DCT blocks.
"""

import numpy as np
from scipy.fftpack import dct
from scipy.stats import kurtosis, entropy as scipy_entropy
from typing import Tuple


def dct2d(block: np.ndarray) -> np.ndarray:
    """Apply 2D DCT to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def extract_block_features(block: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract kurtosis, entropy, and variance from a DCT-transformed block.

    Args:
        block: 8x8 numpy array (grayscale intensity values)

    Returns:
        (kurtosis_val, entropy_val, variance_val)
    """
    coeffs = dct2d(block.astype(np.float64)).flatten()

    # Kurtosis: measures tail heaviness / peakedness
    kurt_val = float(kurtosis(coeffs, fisher=True))

    # Entropy: compute from histogram of DCT coefficients
    hist, _ = np.histogram(coeffs, bins=32, density=True)
    hist = hist + 1e-10  # avoid log(0)
    ent_val = float(scipy_entropy(hist))

    # Variance: spread of DCT coefficients
    var_val = float(np.var(coeffs))

    return kurt_val, ent_val, var_val


def compute_feature_maps(
    image: np.ndarray,
    block_size: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide 8x8 blocks over the image and extract DCT features for each block.

    Args:
        image: 2D grayscale image (H x W)
        block_size: size of each block (default 8)

    Returns:
        Three 2D feature maps: kurtosis_map, entropy_map, variance_map
        Each map has shape (H // block_size, W // block_size)
    """
    h, w = image.shape
    rows = h // block_size
    cols = w // block_size

    kurtosis_map = np.zeros((rows, cols), dtype=np.float64)
    entropy_map  = np.zeros((rows, cols), dtype=np.float64)
    variance_map = np.zeros((rows, cols), dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            block = image[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size
            ]
            k, e, v = extract_block_features(block)
            kurtosis_map[r, c] = k
            entropy_map[r, c]  = e
            variance_map[r, c] = v

    return kurtosis_map, entropy_map, variance_map


def normalize_map(feature_map: np.ndarray) -> np.ndarray:
    """Normalize a feature map to [0, 1] range using robust percentile clipping."""
    lo = np.percentile(feature_map, 2)
    hi = np.percentile(feature_map, 98)
    if hi == lo:
        return np.zeros_like(feature_map, dtype=np.float64)
    clipped = np.clip(feature_map, lo, hi)
    return (clipped - lo) / (hi - lo)
