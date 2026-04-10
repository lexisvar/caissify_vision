"""
api/websocket_handler.py
========================
WebSocket endpoint for live-stream frame processing.

Protocol:
  Client → Server: raw JPEG/PNG bytes  (one frame per message)
  Server → Client: JSON with move and FEN data

  {
    "fen": "rnbqkbnr/pppp...",
    "move_uci": "e2e4",        # null if no new move
    "move_san": "e4",          # null if no new move
    "latency_ms": 45.2
  }
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Live Stream"])


@router.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chess game recording.

    Send JPEG frame bytes → receive JSON board state + move events.
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    from api.main import get_pipeline
    import cv2
    import json

    pipeline = get_pipeline()
    session_id = pipeline.start_game()
    logger.info("Live session started: %s", session_id)

    try:
        while True:
            # Receive a raw image frame
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame."})
                continue

            result = pipeline.process_frame(frame)

            # Build SAN if a move was detected
            move_san = None
            if result.move is not None and result.move_result is not None:
                prev = result.move_result.prev_state
                if prev is not None:
                    try:
                        move_san = prev.board.san(result.move)
                    except Exception:
                        pass

            await websocket.send_json({
                "session_id": session_id,
                "fen": result.fen,
                "move_uci": result.move_uci,
                "move_san": move_san,
                "latency_ms": result.latency_ms,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
        pgn = pipeline.finish_game()
        logger.info("Session %s finished:\n%s", session_id, pgn)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        await websocket.close(code=1011)
