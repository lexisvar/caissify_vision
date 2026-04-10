"""
api/routers/game.py
===================
POST  /game/start             — start a game session
POST  /game/{session_id}/finish
GET   /game/{session_id}/fen
GET   /game/{session_id}/pgn
GET   /game/{session_id}/lichess
POST  /moves/detect           — diff two FENs → detected move
POST  /moves/validate         — check if a move is legal in a position
"""

from __future__ import annotations

import logging

import chess
from fastapi import APIRouter, HTTPException

from api.schemas import (
    DetectMoveRequest,
    DetectMoveResponse,
    FinishGameRequest,
    GameFENResponse,
    GameLichessResponse,
    GamePGNResponse,
    StartGameRequest,
    StartGameResponse,
    ValidateMoveRequest,
    ValidateMoveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Game"])

# In-memory session store: session_id → GameRecorder
_sessions: dict = {}


# ── Game session endpoints ─────────────────────────────────────────────────────


@router.post("/game/start", response_model=StartGameResponse)
async def start_game(body: StartGameRequest):
    """Start a new game recording session."""
    from api.main import get_pipeline
    pipeline = get_pipeline()
    sid = pipeline.start_game(event=body.event, white=body.white, black=body.black)
    _sessions[sid] = pipeline.recorder
    return StartGameResponse(session_id=sid)


@router.post("/game/{session_id}/finish")
async def finish_game(session_id: str, body: FinishGameRequest):
    """Finish a game session and return the PGN."""
    recorder = _sessions.get(session_id)
    if recorder is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    recorder.finish(body.result)
    pgn = recorder.export_pgn()
    return {"session_id": session_id, "pgn": pgn, "result": body.result}


@router.get("/game/{session_id}/fen", response_model=GameFENResponse)
async def get_game_fen(session_id: str):
    """Return the current FEN of a live session."""
    recorder = _sessions.get(session_id)
    if recorder is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return GameFENResponse(
        session_id=session_id,
        fen=recorder.current_fen(),
        move_count=len(recorder),
    )


@router.get("/game/{session_id}/pgn", response_model=GamePGNResponse)
async def get_game_pgn(session_id: str):
    """Return the PGN of all moves recorded so far."""
    recorder = _sessions.get(session_id)
    if recorder is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return GamePGNResponse(
        session_id=session_id,
        pgn=recorder.export_pgn(),
        move_count=len(recorder),
    )


@router.get("/game/{session_id}/lichess", response_model=GameLichessResponse)
async def get_lichess_url(session_id: str):
    """Return a Lichess analysis URL for the current position."""
    recorder = _sessions.get(session_id)
    if recorder is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    fen = recorder.current_fen()
    return GameLichessResponse(
        session_id=session_id,
        fen=fen,
        lichess_url=recorder.lichess_url(),
    )


# ── Move endpoints ─────────────────────────────────────────────────────────────


@router.post("/moves/detect", response_model=DetectMoveResponse)
async def detect_move(body: DetectMoveRequest):
    """
    Diff two FEN positions to determine which move was played.
    Uses python-chess legal move enumeration.
    """
    try:
        prev_board = chess.Board(body.prev_fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prev_fen: {exc}")

    curr_fen_board = body.curr_fen.split()[0]
    matching: list[chess.Move] = []

    for move in prev_board.legal_moves:
        test = prev_board.copy()
        test.push(move)
        if test.board_fen() == curr_fen_board:
            matching.append(move)

    if len(matching) == 1:
        move = matching[0]
        san = prev_board.san(move)
        return DetectMoveResponse(move_uci=move.uci(), move_san=san)
    elif len(matching) > 1:
        return DetectMoveResponse(
            ambiguous=True,
            error=f"Multiple legal moves match: {[m.uci() for m in matching]}",
        )
    else:
        return DetectMoveResponse(
            error="No legal move found between these two positions."
        )


@router.post("/moves/validate", response_model=ValidateMoveResponse)
async def validate_move(body: ValidateMoveRequest):
    """Check whether a UCI move is legal in a given FEN position."""
    try:
        board = chess.Board(body.fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid FEN: {exc}")

    try:
        move = chess.Move.from_uci(body.move_uci)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid UCI move: {exc}")

    is_legal = move in board.legal_moves
    return ValidateMoveResponse(
        is_legal=is_legal,
        message="Legal move." if is_legal else "Illegal move in this position.",
    )
