"""
tests/test_board_state.py
==========================
Unit tests for board_state.py — FEN generation and python-chess integration.
"""

import pytest
import chess

from src.board_state import _compress_row, build_board_state, fen_to_lichess_url


class TestCompressRow:
    def test_full_empty_row(self):
        assert _compress_row(["1"] * 8) == "8"

    def test_no_empty(self):
        chars = ["R", "N", "B", "Q", "K", "B", "N", "R"]
        assert _compress_row(chars) == "RNBQKBNR"

    def test_mixed(self):
        assert _compress_row(["1", "1", "r", "1", "1", "1", "1", "1"]) == "2r5"

    def test_pieces_adjacent(self):
        assert _compress_row(["p", "p", "p", "p", "p", "p", "p", "p"]) == "pppppppp"


class TestBuildBoardState:
    def _make_det_map(self, mapping: dict):
        """Build a minimal detection_map (all None except provided squares)."""
        all_squares = {f"{f}{r}" for f in "abcdefgh" for r in "12345678"}
        det_map = {sq: None for sq in all_squares}
        for sq_name, piece_class in mapping.items():
            # Create a mock PieceDetection
            from src.piece_detector import PieceDetection, CLASS_TO_FEN
            import numpy as np
            det_map[sq_name] = PieceDetection(
                class_name=piece_class,
                fen_char=CLASS_TO_FEN[piece_class],
                confidence=0.95,
                bbox_xyxy=np.array([0, 0, 80, 80], dtype=np.float32),
                source="test",
            )
        return det_map

    def test_empty_board_produces_starting_fen_structure(self):
        """An empty board should produce '8/8/8/8/8/8/8/8' board-FEN."""
        det_map = self._make_det_map({})
        state = build_board_state(det_map)
        board_fen = state.fen.split()[0]
        assert board_fen == "8/8/8/8/8/8/8/8"

    def test_single_white_pawn_on_e4(self):
        det_map = self._make_det_map({"e4": "wP"})
        state = build_board_state(det_map)
        board_fen = state.fen.split()[0]
        # e4 is file e (index 4), rank 4 (row index 4 from top = rank 4)
        assert "P" in board_fen

    def test_starting_position(self):
        """Full starting position should produce a valid python-chess board."""
        mapping = {}
        # Rank 8 — black pieces
        for f, p in zip("abcdefgh", ["bR","bN","bB","bQ","bK","bB","bN","bR"]):
            mapping[f"{f}8"] = p
        # Rank 7 — black pawns
        for f in "abcdefgh":
            mapping[f"{f}7"] = "bP"
        # Rank 2 — white pawns
        for f in "abcdefgh":
            mapping[f"{f}2"] = "wP"
        # Rank 1 — white pieces
        for f, p in zip("abcdefgh", ["wR","wN","wB","wQ","wK","wB","wN","wR"]):
            mapping[f"{f}1"] = p

        det_map = self._make_det_map(mapping)
        state = build_board_state(det_map)
        assert state.is_valid
        assert state.board.board_fen() == chess.Board().board_fen()

    def test_piece_map_populated(self):
        det_map = self._make_det_map({"d4": "bQ"})
        state = build_board_state(det_map)
        assert state.piece_map["d4"] == "q"  # FEN char for black queen


class TestLichessUrl:
    def test_url_contains_fen(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        url = fen_to_lichess_url(fen)
        assert url.startswith("https://lichess.org/analysis/")
        assert "rnbqkbnr" in url
