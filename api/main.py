"""
api/main.py
===========
FastAPI application factory for ChessVision.

Endpoints:
  GET  /health
  GET  /health/models
  POST /analyze/image
  POST /analyze/board-only
  POST /moves/detect
  POST /moves/validate
  POST /game/start
  POST /game/{session_id}/finish
  GET  /game/{session_id}/fen
  GET  /game/{session_id}/pgn
  GET  /game/{session_id}/lichess
  POST /calibration/compute
  GET  /calibration/current
  WS   /ws/live
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, ModelsHealthResponse
from src import __version__

logger = logging.getLogger(__name__)

# ── Global pipeline instance ──────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    """Return the singleton ChessVisionPipeline (lazy-loaded)."""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised. Server did not start correctly.")
    return _pipeline


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("ChessVision API starting up (v%s) ...", __version__)

    from src.pipeline import ChessVisionPipeline

    _pipeline = ChessVisionPipeline(
        corner_model_path=os.getenv("MODEL_CORNER_PATH", "models/corner_detector/best.pt"),
        piece_model_path=os.getenv("MODEL_PIECE_PATH", "models/piece_detector/best.pt"),
        classifier_model_path=os.getenv(
            "MODEL_SQUARE_CLASSIFIER_PATH", "models/square_classifier/best.pt"
        ),
        calibration_path=os.getenv("CALIBRATION_PATH", "data/calibration.json"),
        conf_corner=float(os.getenv("CORNER_CONF_THRESHOLD", "0.4")),
        conf_piece=float(os.getenv("PIECE_CONF_THRESHOLD", "0.6")),
        consensus_count=int(os.getenv("FRAME_CONSENSUS_COUNT", "3")),
        device=os.getenv("INFERENCE_DEVICE", "cpu"),
    )

    logger.info("Pipeline initialised. Device: %s", _pipeline.device)
    yield

    logger.info("ChessVision API shutting down.")
    _pipeline = None


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChessVision API",
        description=(
            "Real-time chess move detection via computer vision. "
            "Detects board position from camera images and records chess games as FEN/PGN."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health endpoints ──────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        return HealthResponse(status="ok", version=__version__)

    @app.get("/health/models", response_model=ModelsHealthResponse, tags=["Health"])
    async def health_models():
        p = get_pipeline()
        info = p.info()
        pd_info = info["piece_detector"]
        return ModelsHealthResponse(
            corner_detector_loaded=p._board_detector._yolo is not None
                                   or p._board_detector._calibration is not None,
            piece_detector_loaded=pd_info["yolo_loaded"],
            square_classifier_loaded=pd_info["classifier_loaded"],
            device=info["device"],
            version=__version__,
        )

    # ── Register routers ──────────────────────────────────────────────────────

    from api.routers import analyze, calibration, game
    from api.websocket_handler import router as ws_router

    app.include_router(analyze.router)
    app.include_router(game.router)
    app.include_router(calibration.router)
    app.include_router(ws_router)

    return app


# Singleton app instance used by uvicorn
app = create_app()
