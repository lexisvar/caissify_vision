"""
tests/test_board_detector.py
=============================
Unit tests for board_detector.py:
  - order_points corner ordering
  - extract_squares grid extraction
  - BoardDetector with a synthetic warped image (no YOLO / calibration needed)
"""

import numpy as np
import pytest

from src.board_detector import (
    SQUARE_SIZE,
    WARP_SIZE,
    extract_squares,
    order_points,
)


# ── order_points ───────────────────────────────────────────────────────────────


class TestOrderPoints:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        result = order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])    # TL
        np.testing.assert_array_almost_equal(result[1], [100, 0])  # TR
        np.testing.assert_array_almost_equal(result[2], [100, 100])# BR
        np.testing.assert_array_almost_equal(result[3], [0, 100])  # BL

    def test_shuffled_order(self):
        pts = np.array([[100, 0], [0, 100], [0, 0], [100, 100]], dtype=np.float32)
        result = order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[2], [100, 100])

    def test_rotated_square(self):
        """Points that form a tilted quad should still order correctly."""
        pts = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32)
        result = order_points(pts)
        assert result.shape == (4, 2)


# ── extract_squares ────────────────────────────────────────────────────────────


class TestExtractSquares:
    @pytest.fixture
    def blank_board(self):
        """A 640×640 solid colour board image."""
        return np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)

    def test_returns_64_squares(self, blank_board):
        squares = extract_squares(blank_board)
        assert len(squares) == 64

    def test_all_algebraic_names_present(self, blank_board):
        squares = extract_squares(blank_board)
        expected = {f"{f}{r}" for f in "abcdefgh" for r in "12345678"}
        assert set(squares.keys()) == expected

    def test_roi_shape(self, blank_board):
        squares = extract_squares(blank_board)
        for sq in squares.values():
            assert sq.roi.shape == (SQUARE_SIZE, SQUARE_SIZE, 3), (
                f"Square {sq.name} ROI has wrong shape: {sq.roi.shape}"
            )

    def test_polygon_shape(self, blank_board):
        squares = extract_squares(blank_board)
        for sq in squares.values():
            assert sq.polygon.shape == (4, 2)

    def test_a8_is_top_left(self, blank_board):
        """a8 should be at pixel (0,0) in the warped image."""
        squares = extract_squares(blank_board)
        a8 = squares["a8"]
        assert a8.polygon[0, 0] == 0  # x1 = 0
        assert a8.polygon[0, 1] == 0  # y1 = 0

    def test_h1_is_bottom_right(self, blank_board):
        """h1 should end at pixel (640, 640)."""
        squares = extract_squares(blank_board)
        h1 = squares["h1"]
        assert h1.polygon[2, 0] == WARP_SIZE   # x2
        assert h1.polygon[2, 1] == WARP_SIZE   # y2

    def test_square_indices(self, blank_board):
        squares = extract_squares(blank_board)
        assert squares["a8"].file_idx == 0
        assert squares["a8"].rank_idx == 0
        assert squares["h1"].file_idx == 7
        assert squares["h1"].rank_idx == 7


# ── BoardDetector (no-model mode) ─────────────────────────────────────────────


class TestBoardDetectorNoModel:
    def test_init_without_models(self):
        """BoardDetector should initialise without raising even if no models present."""
        from src.board_detector import BoardDetector
        detector = BoardDetector(model_path=None, calibration_path="nonexistent.json")
        assert detector._yolo is None
        assert detector._calibration is None

    def test_detect_raises_without_models(self):
        """detect() should raise RuntimeError if no detection method is available."""
        from src.board_detector import BoardDetector
        detector = BoardDetector(model_path=None, calibration_path="nonexistent.json")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            detector.detect(dummy_frame)

    def test_draw_grid_returns_image(self):
        """draw_grid should return a same-size image."""
        from src.board_detector import BoardDetector, BoardDetectionResult
        import numpy as np

        warped = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)
        squares = extract_squares(warped)
        H = np.eye(3, dtype=np.float64)
        result = BoardDetectionResult(
            warped=warped,
            squares=squares,
            corners=np.zeros((4, 2), dtype=np.float32),
            H=H,
            source_bgr=warped,
        )
        detector = BoardDetector(model_path=None, calibration_path="nonexistent.json")
        grid = detector.draw_grid(result)
        assert grid.shape == warped.shape
