"""
tests/test_move_detector.py
============================
Unit tests for move_detector.py — state delta, legal move detection,
special moves (castling, en passant, promotion), temporal consensus.
"""

import chess
import pytest

from src.move_detector import MoveDetector


def make_state(fen: str):
    """Helper: create a minimal BoardState from a FEN string."""
    from src.board_state import BoardState
    board = chess.Board(fen)
    # Minimal piece_map from board
    piece_map = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        name = chess.square_name(sq)
        if piece:
            color = "w" if piece.color == chess.WHITE else "b"
            symbol = piece.symbol().upper()
            piece_map[name] = piece.symbol()
        else:
            piece_map[name] = "1"
    return BoardState(
        fen=fen,
        board=board,
        piece_map=piece_map,
        is_valid=True,
    )


STARTING_FEN = chess.STARTING_FEN


class TestMoveDetectorConsensus:
    def test_no_commit_before_consensus(self):
        detector = MoveDetector(consensus_count=3)
        state = make_state(STARTING_FEN)
        result1 = detector.process_frame(state)
        result2 = detector.process_frame(state)
        # Not committed yet (need 3)
        assert result1.committed is False
        assert result2.committed is False

    def test_commits_after_n_identical_frames(self):
        detector = MoveDetector(consensus_count=3)
        state = make_state(STARTING_FEN)
        for _ in range(3):
            result = detector.process_frame(state)
        assert result.committed is True
        assert detector.committed_state is not None

    def test_resets_buffer_on_different_frame(self):
        detector = MoveDetector(consensus_count=3)
        s1 = make_state(STARTING_FEN)
        s2 = make_state("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

        detector.process_frame(s1)
        detector.process_frame(s1)
        # Inject a different frame — buffer should not give consensus on next s1
        detector.process_frame(s2)
        result = detector.process_frame(s1)
        # Buffer is now [s1, s2, s1] — not all same
        assert result.committed is False


class TestMoveDetection:
    def _commit_initial(self, detector: MoveDetector, fen: str):
        state = make_state(fen)
        for _ in range(detector.consensus_count):
            detector.process_frame(state)

    def test_detects_e2e4(self):
        detector = MoveDetector(consensus_count=1)
        self._commit_initial(detector, STARTING_FEN)

        after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        new_state = make_state(after_e4)
        result = detector.process_frame(new_state)

        assert result.move is not None
        assert result.move.uci() == "e2e4"

    def test_detects_kingside_castling(self):
        """White kingside castling: e1g1"""
        # Position where white can castle kingside
        fen_before = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        fen_after  = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"

        detector = MoveDetector(consensus_count=1)
        self._commit_initial(detector, fen_before)
        result = detector.process_frame(make_state(fen_after))

        assert result.move is not None
        assert result.move.uci() == "e1g1"

    def test_detects_promotion(self):
        """White pawn promotes to queen: a7a8q"""
        fen_before = "8/P7/8/8/8/8/8/4K1k1 w - - 0 1"
        fen_after  = "Q7/8/8/8/8/8/8/4K1k1 b - - 0 1"

        detector = MoveDetector(consensus_count=1)
        self._commit_initial(detector, fen_before)
        result = detector.process_frame(make_state(fen_after))

        assert result.move is not None
        assert result.move.uci() == "a7a8q"

    def test_error_on_impossible_transition(self):
        """A transition that matches no legal move returns an error."""
        detector = MoveDetector(consensus_count=1)
        self._commit_initial(detector, STARTING_FEN)

        # Create a nonsense FEN (teleported pieces)
        nonsense_fen = "rnbqkbnr/8/8/8/8/8/8/RNBQKBNR b - - 0 1"
        result = detector.process_frame(make_state(nonsense_fen))

        assert result.move is None
        assert result.error is not None

    def test_reset_clears_state(self):
        detector = MoveDetector(consensus_count=1)
        self._commit_initial(detector, STARTING_FEN)
        assert detector.committed_state is not None

        detector.reset()
        assert detector.committed_state is None
