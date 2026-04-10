"""
api/schemas.py
==============
Pydantic request and response models for all ChessVision API endpoints.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ModelsHealthResponse(BaseModel):
    corner_detector_loaded: bool
    piece_detector_loaded: bool
    square_classifier_loaded: bool
    device: str
    version: str


# ── Analysis ───────────────────────────────────────────────────────────────────


class AnalyzeImageResponse(BaseModel):
    fen: str
    lichess_url: str
    latency_ms: float
    corner_source: str = Field(
        description="How corners were found: 'yolo' or 'calibration'"
    )


class AnalyzeBoardOnlyResponse(BaseModel):
    corners: list[list[float]] = Field(
        description="4 corners [[x,y],...] in original image pixel space"
    )
    corner_source: str
    warp_size: int = 640


# ── Moves ──────────────────────────────────────────────────────────────────────


class DetectMoveRequest(BaseModel):
    prev_fen: str = Field(description="FEN string of the previous board state")
    curr_fen: str = Field(description="FEN string of the current board state")


class DetectMoveResponse(BaseModel):
    move_uci: Optional[str] = Field(None, description="Detected move in UCI notation")
    move_san: Optional[str] = Field(None, description="Detected move in SAN notation")
    ambiguous: bool = False
    error: Optional[str] = None


class ValidateMoveRequest(BaseModel):
    fen: str
    move_uci: str


class ValidateMoveResponse(BaseModel):
    is_legal: bool
    message: str


# ── Game session ───────────────────────────────────────────────────────────────


class StartGameRequest(BaseModel):
    event: str = "ChessVision Game"
    white: str = "?"
    black: str = "?"


class StartGameResponse(BaseModel):
    session_id: str


class GameFENResponse(BaseModel):
    session_id: str
    fen: str
    move_count: int


class GamePGNResponse(BaseModel):
    session_id: str
    pgn: str
    move_count: int


class GameLichessResponse(BaseModel):
    session_id: str
    lichess_url: str
    fen: str


class FinishGameRequest(BaseModel):
    result: str = Field("*", description="PGN result: 1-0 | 0-1 | 1/2-1/2 | *")


# ── Calibration ────────────────────────────────────────────────────────────────


class CalibrationComputeResponse(BaseModel):
    homography: list[list[float]] = Field(description="3×3 homography matrix")
    warp_size: int = 640
    saved: bool


class CalibrationStatusResponse(BaseModel):
    calibration_loaded: bool
    warp_size: int
    homography: Optional[list[list[float]]] = None
