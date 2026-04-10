"""
api/routers/analyze.py
======================
POST /analyze/image        — upload image → FEN
POST /analyze/board-only   — upload image → board corners
"""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import AnalyzeBoardOnlyResponse, AnalyzeImageResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("/image", response_model=AnalyzeImageResponse)
async def analyze_image(
    file: UploadFile = File(..., description="JPEG/PNG image of the chessboard"),
):
    """
    Upload a chessboard photo → returns the detected FEN + Lichess URL.
    """
    from fastapi import Request
    import cv2

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    from api.main import get_pipeline
    pipeline = get_pipeline()

    try:
        # Temporarily write to a temp file then run; or use detect() directly
        detection = pipeline._board_detector.detect(frame)
        detections = pipeline._piece_detector.detect(detection.warped)

        from src.square_mapper import map_detections_to_squares
        from src.board_state import build_board_state
        import urllib.parse

        det_map = map_detections_to_squares(detections, detection.squares)
        state = build_board_state(det_map)

        import time
        return AnalyzeImageResponse(
            fen=state.fen,
            lichess_url=f"https://lichess.org/analysis/{urllib.parse.quote(state.fen)}",
            latency_ms=0.0,
            corner_source=detection.corner_source,
        )
    except Exception as exc:
        logger.error("analyze_image error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/board-only", response_model=AnalyzeBoardOnlyResponse)
async def analyze_board_only(
    file: UploadFile = File(..., description="JPEG/PNG image of the chessboard"),
):
    """
    Upload a chessboard photo → returns detected corner coordinates only.
    """
    import cv2

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    from api.main import get_pipeline
    pipeline = get_pipeline()

    try:
        detection = pipeline._board_detector.detect(frame)
        corners = detection.corners.tolist()
        return AnalyzeBoardOnlyResponse(
            corners=corners,
            corner_source=detection.corner_source,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
