"""
src/game_recorder.py
====================
Maintains a move list throughout a live game and exports to PGN / FEN.

Integrations:
  - python-chess for PGN generation
  - Lichess analysis URL builder
  - Stockfish analysis (optional)
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess  # type: ignore
import chess.pgn  # type: ignore

logger = logging.getLogger(__name__)


# ── Data class ─────────────────────────────────────────────────────────────────


@dataclass
class RecordedGame:
    """A fully recorded game session."""

    session_id: str
    moves: list[chess.Move] = field(default_factory=list)
    move_fens: list[str] = field(default_factory=list)  # FEN after each move
    started_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    finished_at: Optional[str] = None
    event: str = "ChessVision Game"
    white: str = "?"
    black: str = "?"
    result: str = "*"  # PGN result: "1-0" | "0-1" | "1/2-1/2" | "*"


# ── Main class ─────────────────────────────────────────────────────────────────


class GameRecorder:
    """
    Records moves during a live game and provides export utilities.

    Usage:
        recorder = GameRecorder(session_id="game_001")
        recorder.record_move(move, fen_after)
        pgn = recorder.export_pgn()
        recorder.save("data/games/game_001.json")
    """

    def __init__(
        self,
        session_id: str,
        event: str = "ChessVision Game",
        white: str = "?",
        black: str = "?",
    ) -> None:
        self.game = RecordedGame(
            session_id=session_id,
            event=event,
            white=white,
            black=black,
        )
        self._board = chess.Board()

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_move(self, move: chess.Move, fen_after: Optional[str] = None) -> None:
        """Append a move to the game record."""
        self.game.moves.append(move)
        self._board.push(move)
        self.game.move_fens.append(fen_after or self._board.fen())
        logger.debug("Recorded move %d: %s", len(self.game.moves), move.uci())

    def finish(self, result: str = "*") -> None:
        """Mark the game as finished."""
        self.game.result = result
        self.game.finished_at = datetime.datetime.utcnow().isoformat()
        logger.info("Game %s finished: %s (%d moves)", self.game.session_id, result, len(self.game.moves))

    def reset(self) -> None:
        """Start a new game (clear all moves)."""
        self.game.moves.clear()
        self.game.move_fens.clear()
        self.game.result = "*"
        self._board = chess.Board()

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_pgn(self) -> str:
        """Return the game as a PGN string."""
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = self.game.event
        pgn_game.headers["Date"] = self.game.started_at[:10]
        pgn_game.headers["White"] = self.game.white
        pgn_game.headers["Black"] = self.game.black
        pgn_game.headers["Result"] = self.game.result

        node = pgn_game
        board = chess.Board()
        for move in self.game.moves:
            node = node.add_variation(move)
            board.push(move)

        return str(pgn_game)

    def current_fen(self) -> str:
        """Return the FEN of the current board position."""
        return self._board.fen()

    def lichess_url(self) -> str:
        """Return a Lichess analysis URL for the current position."""
        import urllib.parse
        return f"https://lichess.org/analysis/{urllib.parse.quote(self.current_fen())}"

    def analyse_with_stockfish(
        self,
        stockfish_path: str = "/usr/local/bin/stockfish",
        depth: int = 15,
    ) -> Optional[dict]:
        """
        Run Stockfish on the current position and return analysis.

        Returns:
            Dict with keys: best_move, score, mate_in (or None on error).
        """
        try:
            import stockfish  # type: ignore
            sf = stockfish.Stockfish(path=stockfish_path, depth=depth)
            sf.set_fen_position(self.current_fen())
            best = sf.get_best_move()
            eval_data = sf.get_evaluation()
            return {
                "best_move": best,
                "score": eval_data.get("value"),
                "type": eval_data.get("type"),
                "fen": self.current_fen(),
            }
        except Exception as exc:
            logger.warning("Stockfish analysis failed: %s", exc)
            return None

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the game session to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.game.session_id,
            "event": self.game.event,
            "white": self.game.white,
            "black": self.game.black,
            "result": self.game.result,
            "started_at": self.game.started_at,
            "finished_at": self.game.finished_at,
            "moves_uci": [m.uci() for m in self.game.moves],
            "move_fens": self.game.move_fens,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Game saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GameRecorder":
        """Load a previously saved game session from JSON."""
        data = json.loads(Path(path).read_text())
        recorder = cls(
            session_id=data["session_id"],
            event=data.get("event", "ChessVision Game"),
            white=data.get("white", "?"),
            black=data.get("black", "?"),
        )
        recorder.game.started_at = data.get("started_at", recorder.game.started_at)
        recorder.game.finished_at = data.get("finished_at")
        recorder.game.result = data.get("result", "*")

        for uci in data.get("moves_uci", []):
            move = chess.Move.from_uci(uci)
            recorder.record_move(move)

        recorder.game.move_fens = data.get("move_fens", recorder.game.move_fens)
        return recorder

    def __len__(self) -> int:
        return len(self.game.moves)

    def __repr__(self) -> str:
        return (
            f"GameRecorder(session={self.game.session_id!r}, "
            f"moves={len(self.game.moves)}, result={self.game.result!r})"
        )
