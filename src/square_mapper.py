"""
src/square_mapper.py
====================
Maps YOLO bounding boxes to algebraic square names using IoU geometry.

For each piece detection, find the board square whose polygon has the
highest intersection-over-union with the detection's bounding box.

Special rule for tall pieces (King/Queen): only the *lower half* of their
bounding box is used for matching, since the base of the piece indicates
the occupied square.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from shapely.geometry import Polygon  # type: ignore

from src.board_detector import Square, SQUARE_SIZE
from src.piece_detector import PieceDetection

logger = logging.getLogger(__name__)

# Pieces for which we use only the lower 50 % of the bbox
TALL_PIECES = {"wK", "wQ", "bK", "bQ"}

# Minimum IoU required to commit a piece to a square
MIN_IOU = 0.10


def bbox_to_polygon(
    bbox_xyxy: np.ndarray, lower_half_only: bool = False
) -> Polygon:
    """Convert a [x1, y1, x2, y2] array to a Shapely Polygon."""
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    if lower_half_only:
        y1 = y1 + (y2 - y1) * 0.5  # use bottom 50 %
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def square_to_polygon(sq: Square) -> Polygon:
    pts = sq.polygon.tolist()
    return Polygon(pts)


def calculate_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Intersection over Union for two Shapely polygons."""
    try:
        intersection = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def map_detections_to_squares(
    detections: list[PieceDetection],
    squares: dict[str, Square],
) -> dict[str, Optional[PieceDetection]]:
    """
    Assign each detection to its best-matching square.

    Returns:
        dict mapping algebraic square name → PieceDetection (or None if empty)

    Algorithm:
      - For every detection, compute IoU against all 64 square polygons.
      - Assign the detection to the square with the highest IoU (if > MIN_IOU).
      - If two detections compete for the same square, keep the higher-confidence one.
    """
    # Initialise all squares as None (empty)
    board: dict[str, Optional[PieceDetection]] = {name: None for name in squares}

    for det in detections:
        lower_half = det.class_name in TALL_PIECES
        det_poly = bbox_to_polygon(det.bbox_xyxy, lower_half_only=lower_half)

        best_sq: Optional[str] = None
        best_iou: float = MIN_IOU  # threshold floor

        for sq_name, sq in squares.items():
            sq_poly = square_to_polygon(sq)
            iou = calculate_iou(det_poly, sq_poly)
            if iou > best_iou:
                best_iou = iou
                best_sq = sq_name

        if best_sq is None:
            logger.debug(
                "Detection %s (conf=%.2f) could not be mapped to any square.",
                det.class_name, det.confidence,
            )
            continue

        # Conflict resolution: keep higher confidence
        existing = board[best_sq]
        if existing is None or det.confidence > existing.confidence:
            board[best_sq] = det

    return board
