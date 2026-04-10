"""
api/routers/calibration.py
==========================
POST /calibration/compute   — upload image with 4 corner coordinates → save H
GET  /calibration/current   — return loaded calibration matrix
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from api.schemas import CalibrationComputeResponse, CalibrationStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calibration", tags=["Calibration"])


@router.post("/compute", response_model=CalibrationComputeResponse)
async def compute_calibration(
    file: UploadFile = File(..., description="Board image for calibration"),
    corners: str = Body(
        ...,
        description=(
            "JSON array of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] "
            "ordered TL, TR, BR, BL in original image space"
        ),
    ),
    output_path: str = Body("data/calibration.json"),
):
    """
    Compute and save a homography matrix from a board image + 4 known corners.
    """
    import json
    import cv2

    try:
        pts = np.array(json.loads(corners), dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=422, detail="corners must be a JSON array of 4 [x,y] pairs.")

    if pts.shape != (4, 2):
        raise HTTPException(status_code=422, detail=f"Expected shape (4,2), got {pts.shape}.")

    from src.calibration import Calibration, order_points

    ordered = order_points(pts)
    cal = Calibration.from_points(ordered)
    cal.save(output_path)

    # Reload into the pipeline's board detector
    from api.main import get_pipeline
    pipeline = get_pipeline()
    pipeline._board_detector._calibration = cal

    return CalibrationComputeResponse(
        homography=cal.H.tolist(),
        saved=True,
    )


@router.get("/current", response_model=CalibrationStatusResponse)
async def get_calibration():
    """Return the currently loaded calibration matrix."""
    from api.main import get_pipeline
    pipeline = get_pipeline()
    cal = pipeline._board_detector._calibration

    if cal is None:
        return CalibrationStatusResponse(calibration_loaded=False, warp_size=640)

    return CalibrationStatusResponse(
        calibration_loaded=True,
        warp_size=640,
        homography=cal.H.tolist(),
    )
