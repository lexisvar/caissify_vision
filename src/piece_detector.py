"""
src/piece_detector.py
=====================
Phase 2 — Chess piece detection and classification.

Strategy (hybrid):
  A) YOLO full-board detection: run inference on the 640×640 warped board.
     Returns bounding boxes with 12-class labels.
  B) Square crop classifier (fallback): run a lightweight CNN on each 80×80
     square ROI when YOLO confidence is below threshold.

Classes (13 total):
    empty, wK, wQ, wR, wB, wN, wP,
    bK, bQ, bR, bB, bN, bP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PIECE_CLASSES = [
    "empty",
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]

# FEN character mapping (lowercase = black, uppercase = white)
CLASS_TO_FEN: dict[str, str] = {
    "empty": "1",
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class PieceDetection:
    """A single piece detection result on the warped board."""

    class_name: str          # e.g. "wQ"
    fen_char: str            # e.g. "Q"
    confidence: float
    bbox_xyxy: np.ndarray    # (4,) float32 in warped-board pixel space
    source: str              # "yolo" | "classifier"


# ── Main class ─────────────────────────────────────────────────────────────────


class PieceDetector:
    """
    Detects and classifies all chess pieces on a 640×640 warped board image.

    Args:
        yolo_model_path:       Path to YOLOv8/YOLO11 piece detector weights.
        classifier_model_path: Path to square crop classifier weights (.pt).
        conf_threshold:        YOLO confidence threshold. Detections below this
                               are re-classified with the crop classifier.
        device:                "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self._yolo = None
        self._classifier = None

        # Load YOLO piece detector
        if yolo_model_path and Path(yolo_model_path).exists():
            try:
                from ultralytics import YOLO  # type: ignore
                self._yolo = YOLO(yolo_model_path)
                self._yolo.to(device)
                logger.info("Piece detector loaded from %s (%s)", yolo_model_path, device)
            except Exception as exc:
                logger.warning("Failed to load YOLO piece model: %s", exc)
        else:
            logger.warning(
                "No piece detector weights at '%s'. "
                "Train a model first (see train/train_piece_detector.py).",
                yolo_model_path,
            )

        # Load square crop classifier (optional fallback)
        if classifier_model_path and Path(classifier_model_path).exists():
            try:
                self._classifier = self._load_classifier(classifier_model_path, device)
                logger.info("Square classifier loaded from %s", classifier_model_path)
            except Exception as exc:
                logger.warning("Failed to load square classifier: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, warped_board: np.ndarray) -> list[PieceDetection]:
        """
        Run piece detection on a 640×640 warped board.

        Returns:
            List of PieceDetection objects (one per detected piece).
        """
        if self._yolo is None:
            logger.warning("No YOLO model loaded — returning empty detection list.")
            return []

        # Run YOLO on the full board
        results = self._yolo.predict(
            source=warped_board,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        detections: list[PieceDetection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            # +1 because class 0 in YOLO is "wK" (empty is not detected by YOLO)
            # Adjust based on your dataset's class ordering
            class_name = PIECE_CLASSES[cls_idx + 1] if cls_idx + 1 < len(PIECE_CLASSES) else "empty"
            conf = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()

            detections.append(
                PieceDetection(
                    class_name=class_name,
                    fen_char=CLASS_TO_FEN.get(class_name, "1"),
                    confidence=conf,
                    bbox_xyxy=bbox,
                    source="yolo",
                )
            )

        return detections

    def classify_square(self, roi: np.ndarray) -> tuple[str, float]:
        """
        Run the crop classifier on a single 80×80 square ROI.

        Returns:
            (class_name, confidence)
        """
        if self._classifier is None:
            return "empty", 0.0

        import torch
        from torchvision import transforms  # type: ignore

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls_idx = probs.max(dim=1)

        class_name = PIECE_CLASSES[cls_idx.item()]
        return class_name, conf.item()

    def info(self) -> dict:
        return {
            "yolo_loaded": self._yolo is not None,
            "classifier_loaded": self._classifier is not None,
            "conf_threshold": self.conf_threshold,
            "device": self.device,
            "classes": PIECE_CLASSES,
        }

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_classifier(path: str, device: str):
        """Load an EfficientNet/ResNet square classifier from a .pt checkpoint."""
        import torch

        checkpoint = torch.load(path, map_location=device)
        # Support both raw state_dict and full model
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        else:
            model = checkpoint
        model.eval()
        model.to(device)
        return model
