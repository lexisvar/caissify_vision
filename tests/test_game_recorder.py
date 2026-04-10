"""
tests/test_game_recorder.py
============================
Unit tests for game_recorder.py — move recording, PGN export, persistence.
"""

import json
import tempfile
from pathlib import Path

import chess
import pytest

from src.game_recorder import GameRecorder


class TestGameRecorder:
    def test_initial_state(self):
        r = GameRecorder("test_001")
        assert len(r) == 0
        assert r.game.result == "*"

    def test_record_move_increments_count(self):
        r = GameRecorder("test_001")
        r.record_move(chess.Move.from_uci("e2e4"))
        assert len(r) == 1

    def test_current_fen_updates(self):
        r = GameRecorder("test_001")
        initial_fen = r.current_fen()
        r.record_move(chess.Move.from_uci("e2e4"))
        assert r.current_fen() != initial_fen

    def test_export_pgn_contains_moves(self):
        r = GameRecorder("test_001", event="Test Event")
        r.record_move(chess.Move.from_uci("e2e4"))
        r.record_move(chess.Move.from_uci("e7e5"))
        r.record_move(chess.Move.from_uci("g1f3"))
        pgn = r.export_pgn()
        assert "e4" in pgn
        assert "e5" in pgn
        assert "Nf3" in pgn
        assert "Test Event" in pgn

    def test_finish_sets_result(self):
        r = GameRecorder("test_001")
        r.record_move(chess.Move.from_uci("e2e4"))
        r.finish("1-0")
        assert r.game.result == "1-0"
        assert r.game.finished_at is not None

    def test_lichess_url(self):
        r = GameRecorder("test_001")
        url = r.lichess_url()
        assert url.startswith("https://lichess.org/analysis/")

    def test_save_and_load(self):
        r = GameRecorder("test_save", event="Save Test", white="Alice", black="Bob")
        r.record_move(chess.Move.from_uci("d2d4"))
        r.record_move(chess.Move.from_uci("d7d5"))
        r.finish("1/2-1/2")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "game.json"
            r.save(path)

            loaded = GameRecorder.load(path)
            assert len(loaded) == 2
            assert loaded.game.result == "1/2-1/2"
            assert loaded.game.white == "Alice"
            assert loaded.game.black == "Bob"

    def test_reset_clears_moves(self):
        r = GameRecorder("test_reset")
        r.record_move(chess.Move.from_uci("e2e4"))
        r.reset()
        assert len(r) == 0
        assert r.current_fen() == chess.Board().fen()

    def test_full_game_pgn(self):
        """Scholars mate: 4 moves, should produce valid PGN."""
        r = GameRecorder("scholars_mate")
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "a7a6", "h5f7"]
        for uci in moves:
            r.record_move(chess.Move.from_uci(uci))
        pgn = r.export_pgn()
        assert "1." in pgn
        assert len(pgn) > 10
