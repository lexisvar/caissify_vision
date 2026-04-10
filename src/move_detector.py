"""
src/move_detector.py
====================
Phase 3 — Detects which chess move was played by comparing consecutive
board states, validates it against python-chess legal moves, and handles
temporal smoothing to avoid spurious detections.

Algorithm:
  1. Maintain a rolling buffer of N consecutive identical FEN readings.
  2. Once the buffer is full and all N FENs agree, commit the new state.
  3. Diff the committed state against the previous committed state to find
     the move that was played.
  4. Validate the move is legal; if ambiguous, wait for more frames.
  5. Handle special moves: castling, en passant, promotion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import chess  # type: ignore

from src.board_state import BoardState

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class MoveResult:
    """The outcome of processing a new board state frame."""

    move: Optional[chess.Move]          # None if no new move was committed
    committed: bool                     # True when a new state was locked in
    prev_state: Optional[BoardState]
    curr_state: Optional[BoardState]
    ambiguous: bool = False             # True if multiple legal moves matched
    error: Optional[str] = None        # Description of any detection error


# ── Main class ─────────────────────────────────────────────────────────────────


class MoveDetector:
    """
    Stateful move detector that processes a sequence of BoardState frames.

    Args:
        consensus_count: Number of identical consecutive FENs required to
                         commit a new board state (default 3).
    """

    def __init__(self, consensus_count: int = 3) -> None:
        self.consensus_count = consensus_count
        self._buffer: deque[BoardState] = deque(maxlen=consensus_count)
        self._committed: Optional[BoardState] = None  # last locked-in state
        self._side_to_move: chess.Color = chess.WHITE

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_frame(self, state: BoardState) -> MoveResult:
        """
        Feed one new BoardState into the detector.

        Returns:
            MoveResult describing whether a move was detected this frame.
        """
        # Add to rolling consensus buffer
        self._buffer.append(state)

        # Check if all buffered FENs agree (board part only)
        if not self._buffer_consensus():
            return MoveResult(move=None, committed=False,
                              prev_state=self._committed, curr_state=state)

        # A stable new state has been observed
        stable_state = self._buffer[-1]

        # No committed state yet — this is the initial position
        if self._committed is None:
            self._committed = stable_state
            logger.info("Initial board state committed: %s", stable_state.fen)
            return MoveResult(move=None, committed=True,
                              prev_state=None, curr_state=stable_state)

        # Board didn't change → nothing to do
        if self._board_fen(stable_state) == self._board_fen(self._committed):
            return MoveResult(move=None, committed=False,
                              prev_state=self._committed, curr_state=stable_state)

        # Try to find the legal move that maps prev → curr
        result = self._detect_move(self._committed, stable_state)

        if result.move is not None:
            self._committed = stable_state
            self._side_to_move = not self._side_to_move
            logger.info("Move committed: %s  FEN: %s", result.move.uci(), stable_state.fen)
        elif result.error:
            logger.warning("Move detection error: %s", result.error)

        return result

    def reset(self) -> None:
        """Reset detector state (call at the start of a new game)."""
        self._buffer.clear()
        self._committed = None
        self._side_to_move = chess.WHITE
        logger.info("MoveDetector reset.")

    @property
    def committed_state(self) -> Optional[BoardState]:
        return self._committed

    # ── Internal ───────────────────────────────────────────────────────────────

    def _buffer_consensus(self) -> bool:
        """True if the buffer is full and all board-FENs are identical."""
        if len(self._buffer) < self.consensus_count:
            return False
        fens = [self._board_fen(s) for s in self._buffer]
        return len(set(fens)) == 1

    @staticmethod
    def _board_fen(state: BoardState) -> str:
        """Extract only the board part of the FEN (ignores side/castling/clocks)."""
        return state.fen.split()[0]

    def _detect_move(
        self, prev: BoardState, curr: BoardState
    ) -> MoveResult:
        """
        Find which legal move transforms prev.board into curr.board.

        Returns MoveResult with .move set if exactly one match is found,
        .ambiguous=True if multiple matches, or .error set if none.
        """
        prev_board = prev.board.copy()
        curr_board_fen = self._board_fen(curr)

        matching_moves: list[chess.Move] = []

        for move in prev_board.legal_moves:
            test = prev_board.copy()
            test.push(move)
            if test.board_fen() == curr_board_fen:
                matching_moves.append(move)

        if len(matching_moves) == 1:
            return MoveResult(
                move=matching_moves[0],
                committed=True,
                prev_state=prev,
                curr_state=curr,
            )
        elif len(matching_moves) > 1:
            logger.warning(
                "Ambiguous move: %d candidates %s",
                len(matching_moves),
                [m.uci() for m in matching_moves],
            )
            return MoveResult(
                move=None,
                committed=False,
                prev_state=prev,
                curr_state=curr,
                ambiguous=True,
            )
        else:
            # No legal move produces this board — detection error
            changed = self._changed_squares(prev, curr)
            return MoveResult(
                move=None,
                committed=False,
                prev_state=prev,
                curr_state=curr,
                error=(
                    f"No legal move found. "
                    f"Changed squares: {changed}. "
                    f"Prev FEN: {prev.fen}  Curr FEN: {curr.fen}"
                ),
            )

    @staticmethod
    def _changed_squares(prev: BoardState, curr: BoardState) -> list[str]:
        """Return list of squares where prev and curr differ."""
        return [
            sq for sq in prev.piece_map
            if prev.piece_map.get(sq) != curr.piece_map.get(sq)
        ]
