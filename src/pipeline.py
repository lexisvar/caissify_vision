"""
src/pipeline.py
===============
End-to-end inference pipeline: raw frame → board state → move detection.

This ties together all Phase 1–3 modules:
  BoardDetector → PieceDetector → SquareMapper → BoardState → MoveDetector

Usage (single image):
    from src.pipeline import ChessVisionPipeline
    pipeline = ChessVisionPipeline()
    result = pipeline.run_image("photo.jpg")
    print(result.fen)

Usage (video / live stream):
    pipeline = ChessVisionPipeline()
    pipeline.start_game()
    for frame in camera_frames:
        result = pipeline.process_frame(frame)
        if result.move:
            print("Move played:", result.move.uci())
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess  # type: ignore
import numpy as np

from src.board_detector import BoardDetector, BoardDetectionResult
from src.board_state import BoardState, build_board_state
from src.game_recorder import GameRecorder
from src.move_detector import MoveDetector, MoveResult
from src.piece_detector import PieceDetector
from src.square_mapper import map_detections_to_squares

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────────────────


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    fen: str
    move: Optional[chess.Move]
    move_uci: Optional[str]
    board_state: Optional[BoardState]
    detection: Optional[BoardDetectionResult]
    move_result: Optional[MoveResult]
    latency_ms: float


@dataclass
class ImageResult:
    """Result of analysing a single still image (no move context)."""

    fen: str
    board_state: Optional[BoardState]
    detection: Optional[BoardDetectionResult]
    lichess_url: str
    latency_ms: float


# ── Main pipeline class ────────────────────────────────────────────────────────


class ChessVisionPipeline:
    """
    Full ChessVision inference pipeline.

    Args:
        corner_model_path:     Path to YOLOv8 corner detector weights.
        piece_model_path:      Path to YOLO11 piece detector weights.
        classifier_model_path: Path to EfficientNet square classifier weights.
        calibration_path:      Path to calibration.json.
        conf_corner:           Corner detection confidence threshold.
        conf_piece:            Piece detection confidence threshold.
        consensus_count:       Frames required to commit a state change.
        device:                "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        corner_model_path: Optional[str] = "models/corner_detector/best.pt",
        piece_model_path: Optional[str] = "models/piece_detector/best.pt",
        classifier_model_path: Optional[str] = "models/square_classifier/best.pt",
        calibration_path: str = "data/calibration.json",
        conf_corner: float = 0.4,
        conf_piece: float = 0.6,
        consensus_count: int = 3,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self._board_detector = BoardDetector(
            model_path=corner_model_path,
            calibration_path=calibration_path,
            conf_threshold=conf_corner,
            device=device,
        )
        self._piece_detector = PieceDetector(
            yolo_model_path=piece_model_path,
            classifier_model_path=classifier_model_path,
            conf_threshold=conf_piece,
            device=device,
        )
        self._move_detector = MoveDetector(consensus_count=consensus_count)
        self._recorder: Optional[GameRecorder] = None

    # ── Session management ─────────────────────────────────────────────────────

    def start_game(
        self,
        session_id: Optional[str] = None,
        event: str = "ChessVision Game",
        white: str = "?",
        black: str = "?",
    ) -> str:
        """Start a new game session. Returns the session_id."""
        sid = session_id or str(uuid.uuid4())[:8]
        self._recorder = GameRecorder(sid, event=event, white=white, black=black)
        self._move_detector.reset()
        logger.info("Game session started: %s", sid)
        return sid

    def finish_game(self, result: str = "*") -> Optional[str]:
        """
        Finish the current game session.

        Returns:
            PGN string of the completed game.
        """
        if self._recorder is None:
            return None
        self._recorder.finish(result)
        pgn = self._recorder.export_pgn()
        logger.info("Game finished. PGN:\n%s", pgn)
        return pgn

    # ── Main processing methods ────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single BGR video frame in the context of a running game.

        Returns FrameResult with .move populated when a new move is committed.
        """
        t0 = time.perf_counter()

        try:
            # Phase 1: detect board
            detection = self._board_detector.detect(frame)

            # Phase 2: detect pieces
            detections = self._piece_detector.detect(detection.warped)

            # Map detections to squares
            det_map = map_detections_to_squares(detections, detection.squares)

            # Build board state
            side = (
                self._move_detector.committed_state.board.turn
                if self._move_detector.committed_state
                else chess.WHITE
            )
            state = build_board_state(det_map, side_to_move=side)

            # Phase 3: detect move
            move_result = self._move_detector.process_frame(state)

            # Record if a move was committed
            if move_result.move is not None and self._recorder is not None:
                self._recorder.record_move(move_result.move, state.fen)

            latency = (time.perf_counter() - t0) * 1000
            return FrameResult(
                fen=state.fen,
                move=move_result.move,
                move_uci=move_result.move.uci() if move_result.move else None,
                board_state=state,
                detection=detection,
                move_result=move_result,
                latency_ms=round(latency, 2),
            )

        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("Pipeline error on frame: %s", exc, exc_info=True)
            return FrameResult(
                fen="",
                move=None,
                move_uci=None,
                board_state=None,
                detection=None,
                move_result=None,
                latency_ms=round(latency, 2),
            )

    def run_image(self, image_path: str) -> ImageResult:
        """
        Analyse a single still image (no game context, no move detection).

        Returns:
            ImageResult with the detected FEN and a Lichess URL.
        """
        import cv2
        import urllib.parse

        t0 = time.perf_counter()

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        detection = self._board_detector.detect(frame)
        detections = self._piece_detector.detect(detection.warped)
        det_map = map_detections_to_squares(detections, detection.squares)
        state = build_board_state(det_map)

        latency = (time.perf_counter() - t0) * 1000

        return ImageResult(
            fen=state.fen,
            board_state=state,
            detection=detection,
            lichess_url=f"https://lichess.org/analysis/{urllib.parse.quote(state.fen)}",
            latency_ms=round(latency, 2),
        )

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def current_fen(self) -> Optional[str]:
        state = self._move_detector.committed_state
        return state.fen if state else None

    @property
    def recorder(self) -> Optional[GameRecorder]:
        return self._recorder

    def info(self) -> dict:
        return {
            "device": self.device,
            "piece_detector": self._piece_detector.info(),
            "session_active": self._recorder is not None,
            "moves_recorded": len(self._recorder) if self._recorder else 0,
        }
