"""
src/board_state.py
==================
Converts a per-square piece map into a FEN string and wraps it in a
python-chess Board for legality validation.

FEN format (Forsyth–Edwards Notation):
  - 8 ranks separated by "/"
  - Uppercase = white, lowercase = black
  - Numbers represent consecutive empty squares
  - Example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1

Reference: https://python-chess.readthedocs.io/en/latest/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chess  # type: ignore

from src.piece_detector import CLASS_TO_FEN, PieceDetection

logger = logging.getLogger(__name__)

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 is image row 0


# ── Data class ─────────────────────────────────────────────────────────────────


@dataclass
class BoardState:
    """
    A snapshot of the board at one moment in time.

    Attributes:
        fen:        Full FEN string (board part only — no side-to-move / castling).
        board:      python-chess Board object (for legality checking).
        piece_map:  Dict of algebraic square name → FEN character.
        is_valid:   Whether the FEN represents a legal position.
        raw_map:    The original detection dict before FEN conversion.
    """

    fen: str
    board: chess.Board
    piece_map: dict[str, str]
    is_valid: bool
    raw_map: dict[str, Optional[PieceDetection]] = field(default_factory=dict)


# ── Builder ────────────────────────────────────────────────────────────────────


def build_board_state(
    detection_map: dict[str, Optional[PieceDetection]],
    side_to_move: chess.Color = chess.WHITE,
) -> BoardState:
    """
    Convert a square→PieceDetection map (from square_mapper) into a BoardState.

    Args:
        detection_map:  Output of square_mapper.map_detections_to_squares()
        side_to_move:   chess.WHITE or chess.BLACK (default WHITE)

    Returns:
        BoardState with a fully constructed python-chess Board.
    """
    # Build rank strings top-down (rank 8 → rank 1)
    piece_map: dict[str, str] = {}
    for sq_name, det in detection_map.items():
        piece_map[sq_name] = det.fen_char if det is not None else "1"

    fen_rows: list[str] = []
    for rank in RANKS:
        row_chars = [piece_map.get(f"{f}{rank}", "1") for f in FILES]
        fen_rows.append(_compress_row(row_chars))

    board_fen = "/".join(fen_rows)

    # Attempt to build a python-chess board
    # Use a generic FEN with unknown castling / en-passant for now
    side_char = "w" if side_to_move == chess.WHITE else "b"
    full_fen = f"{board_fen} {side_char} KQkq - 0 1"

    valid = False
    board = chess.Board()
    try:
        board = chess.Board(full_fen)
        valid = True
    except ValueError as exc:
        logger.debug("FEN is not strictly legal: %s  |  %s", full_fen, exc)
        # Still store what we have — the validator in move_detector will filter it
        try:
            board.set_fen(full_fen)
        except Exception:
            pass

    return BoardState(
        fen=full_fen,
        board=board,
        piece_map=piece_map,
        is_valid=valid,
        raw_map=detection_map,
    )


def _compress_row(chars: list[str]) -> str:
    """
    Compress a row of 8 FEN characters by merging consecutive '1's into a count.
    e.g. ['1','1','r','1','1','1','1','1'] → '2r5'
    """
    result = ""
    empty_count = 0
    for c in chars:
        if c == "1":
            empty_count += 1
        else:
            if empty_count:
                result += str(empty_count)
                empty_count = 0
            result += c
    if empty_count:
        result += str(empty_count)
    return result


def fen_to_lichess_url(fen: str) -> str:
    """Return a Lichess analysis URL for a given FEN."""
    import urllib.parse
    return f"https://lichess.org/analysis/{urllib.parse.quote(fen)}"
