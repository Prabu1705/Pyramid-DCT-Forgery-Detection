"""
core/visualization.py
---------------------
Heatmap and binary mask rendering using OpenCV + matplotlib.
Returns PNG bytes for API transport.
"""

import io
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple


def score_map_to_heatmap(
    score_map: np.ndarray,
    original_bgr: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Overlay a colored heatmap on the original image.

    Args:
        score_map: 2D float array in [0, 1], same spatial size as original_bgr
        original_bgr: original image in BGR format (H x W x 3)
        colormap: OpenCV colormap constant
        alpha: blend factor for overlay (0 = original only, 1 = heatmap only)

    Returns:
        blended: BGR image with heatmap overlay
    """
    h, w = original_bgr.shape[:2]

    # Resize score map to match image if needed
    score_resized = cv2.resize(
        score_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Convert to uint8 and apply colormap
    score_uint8 = np.uint8(score_resized * 255)
    heatmap_colored = cv2.applyColorMap(score_uint8, colormap)

    # Blend with original
    blended = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return blended


def score_map_to_mask_overlay(
    mask: np.ndarray,
    original_bgr: np.ndarray,
    mask_color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay binary mask regions on the original image.

    Args:
        mask: binary uint8 mask (0 or 255), shape (H, W)
        original_bgr: original image BGR
        mask_color: BGR color for highlighted regions
        alpha: blending factor

    Returns:
        overlay: BGR image with mask regions highlighted
    """
    h, w = original_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(original_bgr)
    color_layer[:] = mask_color

    mask_3ch = np.stack([mask_resized] * 3, axis=-1).astype(np.float32) / 255.0

    overlay = original_bgr.astype(np.float32)
    overlay = overlay * (1 - alpha * mask_3ch) + color_layer.astype(np.float32) * alpha * mask_3ch
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_matplotlib_heatmap(
    score_map: np.ndarray,
    title: str = "Anomaly Heatmap"
) -> np.ndarray:
    """
    Render a standalone publication-quality heatmap figure using matplotlib.

    Returns:
        BGR image (numpy array)
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    im = ax.imshow(
        score_map,
        cmap='inferno',
        vmin=0, vmax=1,
        aspect='auto'
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
    cbar.set_label('Anomaly Score', color='white', fontsize=9)

    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('X (pixels)', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('Y (pixels)', color='#aaaaaa', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def encode_image_to_bytes(image_bgr: np.ndarray, ext: str = '.png') -> bytes:
    """Encode a BGR numpy image to PNG/JPEG bytes."""
    success, buf = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"Failed to encode image as {ext}")
    return buf.tobytes()


def generate_side_by_side(
    original_bgr: np.ndarray,
    heatmap_overlay: np.ndarray,
    mask_overlay: np.ndarray
) -> np.ndarray:
    """
    Compose original, heatmap overlay, and mask overlay side by side.
    All images are resized to the same height for consistency.
    """
    target_h = 300
    def resize_h(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))

    panels = [resize_h(original_bgr), resize_h(heatmap_overlay), resize_h(mask_overlay)]

    # Add labels
    labels = ["Original", "Anomaly Heatmap", "Forgery Mask"]
    labeled = []
    for img, label in zip(panels, labels):
        lbl_img = img.copy()
        cv2.putText(
            lbl_img, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            lbl_img, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA
        )
        labeled.append(lbl_img)

    return np.hstack(labeled)
