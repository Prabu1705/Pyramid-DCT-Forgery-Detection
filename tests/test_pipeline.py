"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the Pyramid DCT Forgery Detection pipeline.
Uses only Python stdlib unittest (no pytest required).

Run with:
    python -m unittest discover -s tests -v
  or:
    python tests/test_pipeline.py
"""

import sys
import os
import tempfile
import unittest
import numpy as np
import cv2

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_image(h=128, w=128, tampered=False):
    img = (np.random.rand(h, w) * 80 + 90).astype(np.uint8)
    if tampered:
        img[32:64, 32:64] = (np.random.rand(32, 32) * 255).astype(np.uint8)
    return img

def save_temp_image(arr, suffix='.png'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    cv2.imwrite(path, arr)
    return path


# ─── DCT Features ─────────────────────────────────────────────────────────────

class TestDctFeatures(unittest.TestCase):

    def test_dct2d_shape(self):
        from backend.core.dct_features import dct2d
        block = np.random.rand(8, 8)
        result = dct2d(block)
        self.assertEqual(result.shape, (8, 8))

    def test_extract_block_features_returns_three_floats(self):
        from backend.core.dct_features import extract_block_features
        block = np.random.rand(8, 8) * 200
        k, e, v = extract_block_features(block)
        self.assertIsInstance(k, float)
        self.assertIsInstance(e, float)
        self.assertIsInstance(v, float)

    def test_variance_nonnegative(self):
        from backend.core.dct_features import extract_block_features
        block = np.ones((8, 8)) * 128
        _, _, v = extract_block_features(block)
        self.assertGreaterEqual(v, 0.0)

    def test_entropy_nonnegative(self):
        from backend.core.dct_features import extract_block_features
        block = np.random.rand(8, 8) * 255
        _, e, _ = extract_block_features(block)
        self.assertGreaterEqual(e, 0.0)

    def test_compute_feature_maps_shapes(self):
        from backend.core.dct_features import compute_feature_maps
        img = make_synthetic_image(64, 64)
        k, e, v = compute_feature_maps(img, block_size=8)
        self.assertEqual(k.shape, (8, 8))
        self.assertEqual(e.shape, (8, 8))
        self.assertEqual(v.shape, (8, 8))

    def test_normalize_map_range(self):
        from backend.core.dct_features import normalize_map
        fm = np.random.rand(10, 10) * 100
        norm = normalize_map(fm)
        self.assertGreaterEqual(float(norm.min()), 0.0)
        self.assertLessEqual(float(norm.max()), 1.0)

    def test_uniform_block_dct_ac_near_zero(self):
        """Uniform block: all AC DCT coefficients should be near zero (energy in DC only)."""
        from backend.core.dct_features import dct2d
        block = np.full((8, 8), 128.0)
        coeffs = dct2d(block)
        # AC coefficients (everything except [0,0]) must be near zero
        ac = coeffs.flatten()[1:]
        self.assertAlmostEqual(float(np.max(np.abs(ac))), 0.0, places=5)


# ─── Pyramid ──────────────────────────────────────────────────────────────────

class TestPyramid(unittest.TestCase):

    def test_pyramid_length(self):
        from backend.core.pyramid import build_gaussian_pyramid
        img = make_synthetic_image(128, 128)
        pyr = build_gaussian_pyramid(img.astype(np.float64), levels=4)
        self.assertEqual(len(pyr), 4)

    def test_pyramid_decreasing_size(self):
        from backend.core.pyramid import build_gaussian_pyramid
        img = make_synthetic_image(128, 128)
        pyr = build_gaussian_pyramid(img.astype(np.float64), levels=4)
        for i in range(1, len(pyr)):
            self.assertLessEqual(pyr[i].shape[0], pyr[i-1].shape[0])

    def test_analyze_pyramid_output_shapes(self):
        from backend.core.pyramid import analyze_pyramid
        img = make_synthetic_image(64, 64)
        k, e, v, stats = analyze_pyramid(img, levels=3, block_size=8)
        self.assertEqual(k.shape, (64, 64))
        self.assertEqual(e.shape, (64, 64))
        self.assertEqual(v.shape, (64, 64))

    def test_analyze_pyramid_level_stats_keys(self):
        from backend.core.pyramid import analyze_pyramid
        img = make_synthetic_image(64, 64)
        _, _, _, stats = analyze_pyramid(img, levels=3, block_size=8)
        self.assertGreaterEqual(len(stats), 1)
        for s in stats:
            self.assertIn("level", s)
            self.assertIn("resolution", s)
            self.assertIn("kurtosis_mean", s)
            self.assertIn("entropy_mean", s)
            self.assertIn("variance_mean", s)

    def test_analyze_pyramid_maps_in_range(self):
        from backend.core.pyramid import analyze_pyramid
        img = make_synthetic_image(64, 64)
        k, e, v, _ = analyze_pyramid(img, levels=3, block_size=8)
        for fm in (k, e, v):
            self.assertGreaterEqual(float(fm.min()), 0.0)
            self.assertLessEqual(float(fm.max()), 1.0 + 1e-6)


# ─── Statistics ───────────────────────────────────────────────────────────────

class TestStatistics(unittest.TestCase):

    def test_combine_anomaly_scores_range(self):
        from backend.core.statistics import combine_anomaly_scores
        k = np.random.rand(64, 64)
        e = np.random.rand(64, 64)
        v = np.random.rand(64, 64)
        score = combine_anomaly_scores(k, e, v)
        self.assertGreaterEqual(float(score.min()), 0.0)
        self.assertLessEqual(float(score.max()), 1.0)

    def test_combine_weights_sum(self):
        """Weights must produce values in [0,1] for unit inputs."""
        from backend.core.statistics import combine_anomaly_scores
        ones = np.ones((10, 10))
        score = combine_anomaly_scores(ones, ones, ones)
        self.assertAlmostEqual(float(score.max()), 1.0, places=5)

    def test_generate_binary_mask_values(self):
        from backend.core.statistics import generate_binary_mask
        score = np.random.rand(64, 64)
        mask, thresh = generate_binary_mask(score)
        unique = set(np.unique(mask).tolist())
        self.assertTrue(unique.issubset({0, 255}))
        self.assertGreater(thresh, 0.0)
        self.assertLess(thresh, 1.0)

    def test_generate_binary_mask_manual_threshold(self):
        from backend.core.statistics import generate_binary_mask
        score = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        mask, thresh = generate_binary_mask(score, threshold=0.5)
        self.assertAlmostEqual(thresh, 0.5, places=5)
        self.assertEqual(int(mask[0, 0]), 0)       # lowest score → clean
        self.assertEqual(int(mask[-1, -1]), 255)   # highest score → tampered

    def test_chi_square_anomaly_map_range(self):
        from backend.core.statistics import chi_square_anomaly_map
        fm = np.random.rand(64, 64)
        chi2 = chi_square_anomaly_map(fm, patch_size=8)
        self.assertGreaterEqual(float(chi2.min()), 0.0)
        self.assertLessEqual(float(chi2.max()), 1.0 + 1e-6)

    def test_chi_square_anomaly_map_shape(self):
        from backend.core.statistics import chi_square_anomaly_map
        fm = np.random.rand(64, 64)
        chi2 = chi_square_anomaly_map(fm, patch_size=8)
        self.assertEqual(chi2.shape, (64, 64))

    def test_compute_global_statistics_keys(self):
        from backend.core.statistics import compute_global_statistics
        score = np.random.rand(64, 64)
        mask  = (score > 0.5).astype(np.uint8) * 255
        result = compute_global_statistics(score, mask, [])
        for key in ["verdict", "confidence", "forgery_fraction",
                    "anomaly_score_mean", "anomaly_score_max",
                    "tampered_pixels", "total_pixels"]:
            self.assertIn(key, result)

    def test_verdict_authentic(self):
        from backend.core.statistics import compute_global_statistics
        score = np.full((100, 100), 0.1)
        mask  = np.zeros((100, 100), dtype=np.uint8)
        result = compute_global_statistics(score, mask, [])
        self.assertEqual(result["verdict"], "Likely Authentic")

    def test_verdict_tampered(self):
        from backend.core.statistics import compute_global_statistics
        score = np.random.rand(100, 100) * 0.5 + 0.3
        mask  = np.ones((100, 100), dtype=np.uint8) * 255  # all tampered
        result = compute_global_statistics(score, mask, [])
        self.assertIn("Tampered", result["verdict"])


# ─── Visualization ────────────────────────────────────────────────────────────

class TestVisualization(unittest.TestCase):

    def test_encode_image_to_bytes_png(self):
        from backend.core.visualization import encode_image_to_bytes
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        data = encode_image_to_bytes(img, '.png')
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
        # PNG magic bytes
        self.assertTrue(data[:4] == b'\x89PNG')

    def test_encode_image_to_bytes_jpeg(self):
        from backend.core.visualization import encode_image_to_bytes
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        data = encode_image_to_bytes(img, '.jpg')
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_heatmap_overlay_shape(self):
        from backend.core.visualization import score_map_to_heatmap
        score = np.random.rand(64, 64)
        bgr   = np.zeros((64, 64, 3), dtype=np.uint8)
        out   = score_map_to_heatmap(score, bgr)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_mask_overlay_shape(self):
        from backend.core.visualization import score_map_to_mask_overlay
        mask = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255
        bgr  = np.zeros((64, 64, 3), dtype=np.uint8)
        out  = score_map_to_mask_overlay(mask, bgr)
        self.assertEqual(out.shape, (64, 64, 3))

    def test_heatmap_overlay_different_size(self):
        """Score map resized to match image dimensions."""
        from backend.core.visualization import score_map_to_heatmap
        score = np.random.rand(32, 32)
        bgr   = np.zeros((128, 128, 3), dtype=np.uint8)
        out   = score_map_to_heatmap(score, bgr)
        self.assertEqual(out.shape, (128, 128, 3))

    def test_matplotlib_heatmap_returns_bgr(self):
        from backend.core.visualization import render_matplotlib_heatmap
        score = np.random.rand(64, 64)
        out   = render_matplotlib_heatmap(score)
        self.assertEqual(len(out.shape), 3)
        self.assertEqual(out.shape[2], 3)
        self.assertEqual(out.dtype, np.uint8)

    def test_side_by_side_width(self):
        from backend.core.visualization import generate_side_by_side
        orig = np.zeros((200, 300, 3), dtype=np.uint8)
        hm   = np.zeros((200, 300, 3), dtype=np.uint8)
        mk   = np.zeros((200, 300, 3), dtype=np.uint8)
        out  = generate_side_by_side(orig, hm, mk)
        self.assertEqual(len(out.shape), 3)
        self.assertEqual(out.shape[2], 3)


# ─── Full Pipeline Integration ────────────────────────────────────────────────

class TestPipelineIntegration(unittest.TestCase):

    def test_run_authentic_image(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128, tampered=False)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            self.assertIn("statistics", results)
            self.assertIn("heatmap_overlay_bytes", results)
            self.assertIn("mask_overlay_bytes", results)
            self.assertIn("heatmap_pure_bytes", results)
            self.assertIn("side_by_side_bytes", results)
            self.assertIsInstance(results["heatmap_overlay_bytes"], bytes)
            self.assertEqual(results["score_map"].shape, (128, 128))
        finally:
            os.unlink(path)

    def test_run_tampered_image(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128, tampered=True)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            stats = results["statistics"]
            self.assertIn("verdict", stats)
            self.assertGreaterEqual(stats["forgery_fraction"], 0.0)
            self.assertLessEqual(stats["forgery_fraction"], 100.0)
        finally:
            os.unlink(path)

    def test_invalid_path_raises_value_error(self):
        from backend.core.pipeline import run_forgery_detection
        with self.assertRaises(ValueError):
            run_forgery_detection("/nonexistent/fake/image.png")

    def test_jpeg_input(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img, suffix='.jpg')
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            self.assertGreater(results["statistics"]["total_pixels"], 0)
        finally:
            os.unlink(path)

    def test_custom_threshold(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3, threshold=0.6)
            self.assertAlmostEqual(results["threshold_used"], 0.6, places=5)
        finally:
            os.unlink(path)

    def test_pyramid_levels_3_to_5(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            for lvl in (3, 4, 5):
                r = run_forgery_detection(path, pyramid_levels=lvl)
                self.assertEqual(r["statistics"]["pyramid_levels_used"], lvl)
        finally:
            os.unlink(path)

    def test_score_map_values_in_range(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            sm = results["score_map"]
            self.assertGreaterEqual(float(sm.min()), 0.0)
            self.assertLessEqual(float(sm.max()), 1.0 + 1e-6)
        finally:
            os.unlink(path)

    def test_mask_binary_values(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            unique = set(np.unique(results["mask"]).tolist())
            self.assertTrue(unique.issubset({0, 255}))
        finally:
            os.unlink(path)

    def test_output_images_are_valid_png(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            for key in ("heatmap_overlay_bytes", "mask_overlay_bytes",
                        "heatmap_pure_bytes", "side_by_side_bytes"):
                data = results[key]
                self.assertTrue(data[:4] == b'\x89PNG',
                                f"{key} is not a valid PNG")
        finally:
            os.unlink(path)

    def test_statistics_pixel_counts_consistent(self):
        from backend.core.pipeline import run_forgery_detection
        img  = make_synthetic_image(128, 128)
        path = save_temp_image(img)
        try:
            results = run_forgery_detection(path, pyramid_levels=3)
            stats = results["statistics"]
            self.assertLessEqual(stats["tampered_pixels"], stats["total_pixels"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
