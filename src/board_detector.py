"""
src/board_detector.py
=====================
Phase 1 — Chessboard localisation pipeline.

Steps:
  1. Load a raw frame (from camera or file)
  2. Detect the 4 outer corners via YOLO (or fall back to calibration matrix)
  3. Order corners consistently (TL, TR, BR, BL)
  4. Compute + apply perspective transform → 640×640 bird's-eye board
  5. Extract the 64 individual square ROIs with algebraic labels

The key export is `BoardDetector` which produces a `BoardDetectionResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.calibration import Calibration, WARP_SIZE

logger = logging.getLogger(__name__)

# Each warped square is 80×80 px  (640 / 8)
SQUARE_SIZE = WARP_SIZE // 8

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 at top of image (row 0)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Square:
    """One of the 64 board squares with its image ROI and geometry."""

    name: str           # e.g. "e4"
    file_idx: int       # 0–7  (a=0 … h=7)
    rank_idx: int       # 0–7  (rank8=0 … rank1=7)
    roi: np.ndarray     # 80×80 BGR image crop from the warped board
    polygon: np.ndarray # (4,2) float32 corners in warped-board pixel space


@dataclass
class BoardDetectionResult:
    """
    Everything produced by the board detector for one frame.

    Attributes:
        warped:     640×640 BGR bird's-eye board image.
        squares:    Dict of algebraic-name → Square.
        corners:    (4,2) float32 corners in the *original* (pre-warp) image.
        H:          3×3 homography matrix used for this frame.
        source_bgr: The original (pre-warp) frame.
        corner_source: "yolo" | "calibration" indicating how corners were found.
    """

    warped: np.ndarray
    squares: dict[str, Square]
    corners: np.ndarray
    H: np.ndarray
    source_bgr: np.ndarray
    corner_source: str = "calibration"


# ── Helper: corner ordering ───────────────────────────────────────────────────


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 corner points into [top-left, top-right, bottom-right, bottom-left].

    Uses the classic sum/diff trick:
      - TL has the smallest (x+y)
      - BR has the largest (x+y)
      - TR has the smallest (x-y)
      - BL has the largest (x-y)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


# ── Helper: square grid extraction ───────────────────────────────────────────


def extract_squares(warped: np.ndarray) -> dict[str, Square]:
    """
    Divide the 640×640 warped board into 64 labelled Square objects.

    Coordinate convention:
      - File a is left column  (x = 0)
      - Rank 8 is top row      (y = 0)
    """
    squares: dict[str, Square] = {}
    sq = SQUARE_SIZE

    for r_idx, rank in enumerate(RANKS):
        for f_idx, file in enumerate(FILES):
            name = f"{file}{rank}"
            x1, y1 = f_idx * sq, r_idx * sq
            x2, y2 = x1 + sq, y1 + sq

            roi = warped[y1:y2, x1:x2].copy()
            polygon = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
            squares[name] = Square(
                name=name,
                file_idx=f_idx,
                rank_idx=r_idx,
                roi=roi,
                polygon=polygon,
            )

    return squares


# ── Main class ────────────────────────────────────────────────────────────────


class BoardDetector:
    """
    Detects the chessboard in a raw frame and returns a BoardDetectionResult.

    Corner detection strategy (in priority order):
      1. YOLO corner model (if model_path is provided and file exists)
      2. Pre-calibrated homography (if calibration_path exists)
      3. Raises RuntimeError if neither is available

    Args:
        model_path:       Path to YOLOv8 corner detector weights (.pt).
                          Pass None to skip YOLO and use calibration only.
        calibration_path: Path to calibration.json produced by calibration.py.
        conf_threshold:   YOLO confidence threshold for corner detections.
        device:           "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        calibration_path: str = "data/calibration.json",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self._yolo = None
        self._calibration: Optional[Calibration] = None

        # Try to load YOLO corner detector
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO  # type: ignore
                self._yolo = YOLO(model_path)
                self._yolo.to(device)
                logger.info("Corner detector loaded from %s (%s)", model_path, device)
            except Exception as exc:
                logger.warning("Failed to load YOLO corner model: %s", exc)

        # Load static calibration as fallback
        cal_path = Path(calibration_path)
        if cal_path.exists():
            try:
                self._calibration = Calibration.from_file(cal_path)
                logger.info("Calibration loaded from %s", cal_path)
            except Exception as exc:
                logger.warning("Failed to load calibration: %s", exc)

        if self._yolo is None and self._calibration is None:
            logger.warning(
                "BoardDetector has no YOLO model and no calibration file. "
                "Call detect() only after providing at least one."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> BoardDetectionResult:
        """
        Run the full board detection pipeline on a BGR frame.

        Returns:
            BoardDetectionResult

        Raises:
            RuntimeError: If no detection method is available or corners are not found.
        """
        corners, source = self._find_corners(frame)

        # Order corners consistently
        ordered = order_points(corners)

        # Compute homography
        dst_pts = np.float32(
            [
                [0, 0],
                [WARP_SIZE - 1, 0],
                [WARP_SIZE - 1, WARP_SIZE - 1],
                [0, WARP_SIZE - 1],
            ]
        )
        H, _ = cv2.findHomography(ordered, dst_pts)
        warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))

        squares = extract_squares(warped)

        return BoardDetectionResult(
            warped=warped,
            squares=squares,
            corners=ordered,
            H=H,
            source_bgr=frame,
            corner_source=source,
        )

    def detect_from_file(self, path: str) -> BoardDetectionResult:
        """Convenience wrapper: load a BGR image from disk and run detect()."""
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.detect(frame)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_corners(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Try YOLO first, then calibration.
        Returns (corners_array, source_label).
        corners_array shape: (4, 2) float32
        """
        if self._yolo is not None:
            corners = self._yolo_corners(frame)
            if corners is not None:
                return corners, "yolo"
            logger.debug("YOLO corner detection failed/low-conf; falling back to calibration.")

        if self._calibration is not None:
            # With a static calibration, the "corners" are the 4 image corners
            # mapped through the inverse homography
            corners = self._calibration_corners(frame)
            return corners, "calibration"

        raise RuntimeError(
            "No detection method available. Provide a YOLO model path or a calibration.json."
        )

    def _yolo_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLO on the frame and extract exactly 4 corner detections."""
        results = self._yolo.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes
        if len(boxes) < 4:
            logger.debug("YOLO found %d corners (need 4)", len(boxes))
            return None

        # Use the 4 highest-confidence detections
        confs = boxes.conf.cpu().numpy()
        top4 = np.argsort(confs)[-4:]
        xyxy = boxes.xyxy.cpu().numpy()[top4]
        # Use centre of each bbox as the corner point
        centres = np.column_stack(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]
        )
        return centres.astype(np.float32)

    def _calibration_corners(self, frame: np.ndarray) -> np.ndarray:
        """
        Derive source corners by back-projecting the destination square corners
        through the inverse homography stored in calibration.
        """
        H_inv = np.linalg.inv(self._calibration.H)
        dst_pts = np.float32(
            [
                [0, 0, 1],
                [WARP_SIZE - 1, 0, 1],
                [WARP_SIZE - 1, WARP_SIZE - 1, 1],
                [0, WARP_SIZE - 1, 1],
            ]
        ).T  # (3, 4)
        src_h = H_inv @ dst_pts        # (3, 4)
        src = (src_h[:2] / src_h[2]).T  # (4, 2)
        return src.astype(np.float32)

    # ── Debug visualisation ───────────────────────────────────────────────────

    def draw_grid(self, result: BoardDetectionResult) -> np.ndarray:
        """Draw the 8×8 grid on the warped image (useful for debugging)."""
        vis = result.warped.copy()
        sq = SQUARE_SIZE
        for i in range(9):
            cv2.line(vis, (i * sq, 0), (i * sq, WARP_SIZE), (0, 255, 0), 1)
            cv2.line(vis, (0, i * sq), (WARP_SIZE, i * sq), (0, 255, 0), 1)
        # Label a1
        cv2.putText(vis, "a8", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(
            vis, "h1", (WARP_SIZE - 26, WARP_SIZE - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        return vis
