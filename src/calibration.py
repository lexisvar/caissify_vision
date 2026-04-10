"""
src/calibration.py
==================
Camera calibration tool.

Two modes:
  1. Interactive: click 4 corners on a raw image → saves homography matrix
  2. Auto: provide known 4 corner coordinates programmatically

The homography matrix H transforms a raw camera frame so the chessboard
appears as a perfect 640×640 bird's-eye square.

Usage (interactive CLI):
    python src/calibration.py --image path/to/board.jpg --output data/calibration.json

Usage (from code):
    from src.calibration import Calibration
    cal = Calibration.from_file("data/calibration.json")
    warped = cal.warp(frame)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Output size for the warped board (pixels)
WARP_SIZE = 640


class Calibration:
    """Holds the homography matrix and applies it to raw frames."""

    def __init__(self, H: np.ndarray) -> None:
        self.H = H  # 3×3 float32 homography matrix

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_points(
        cls,
        src_pts: np.ndarray,
        dst_size: int = WARP_SIZE,
    ) -> "Calibration":
        """
        Compute H from four source corner points.

        Args:
            src_pts: (4, 2) float32 array — corners in raw image space,
                     ordered: [top-left, top-right, bottom-right, bottom-left]
            dst_size: Side length of the output square (default 640)
        """
        dst_pts = np.float32(
            [
                [0, 0],
                [dst_size - 1, 0],
                [dst_size - 1, dst_size - 1],
                [0, dst_size - 1],
            ]
        )
        H, _ = cv2.findHomography(src_pts.astype(np.float32), dst_pts)
        return cls(H)

    @classmethod
    def from_file(cls, path: str | Path) -> "Calibration":
        """Load a previously saved calibration JSON."""
        data = json.loads(Path(path).read_text())
        H = np.array(data["H"], dtype=np.float64)
        logger.info("Calibration loaded from %s", path)
        return cls(H)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the homography matrix to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"H": self.H.tolist(), "warp_size": WARP_SIZE}
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Calibration saved to %s", path)

    # ── Transform ─────────────────────────────────────────────────────────────

    def warp(self, image: np.ndarray, size: int = WARP_SIZE) -> np.ndarray:
        """
        Apply the homography to produce a bird's-eye square board image.

        Returns:
            (size × size) uint8 RGB image.
        """
        return cv2.warpPerspective(image, self.H, (size, size))

    def is_valid(self) -> bool:
        return self.H is not None and self.H.shape == (3, 3)

    def __repr__(self) -> str:
        return f"Calibration(H={self.H.tolist()})"


# ── Interactive calibration tool ──────────────────────────────────────────────

class _CornerSelector:
    """OpenCV mouse handler that collects exactly 4 corner clicks."""

    INSTRUCTIONS = (
        "Click the 4 board corners in order:\n"
        "  1) Top-left  2) Top-right\n"
        "  3) Bottom-right  4) Bottom-left\n"
        "Press R to reset, Q/Enter to confirm."
    )

    def __init__(self, image: np.ndarray) -> None:
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[int, int]] = []

    def _mouse_callback(self, event: int, x: int, y: int, *_) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 4:
            return
        self.points.append((x, y))
        self.display = self.image.copy()
        colors = [
            (0, 255, 0),
            (0, 200, 255),
            (0, 100, 255),
            (255, 0, 150),
        ]
        for i, pt in enumerate(self.points):
            cv2.circle(self.display, pt, 8, colors[i], -1)
            cv2.putText(
                self.display,
                str(i + 1),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colors[i],
                2,
            )

    def run(self) -> Optional[np.ndarray]:
        win = "ChessVision — Calibration (R=reset, Enter/Q=confirm)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_callback)

        for line in self.INSTRUCTIONS.split("\n"):
            logger.info(line)

        while True:
            cv2.imshow(win, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                self.points.clear()
                self.display = self.image.copy()
            elif key in (13, ord("q")) and len(self.points) == 4:
                break

        cv2.destroyAllWindows()
        if len(self.points) == 4:
            return np.array(self.points, dtype=np.float32)
        return None


def run_interactive(image_path: str, output_path: str) -> None:
    """
    Open an image for interactive corner selection and save the calibration.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Resize for display if very large
    h, w = image.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        scale_factor = 1.0 / scale
    else:
        display_img = image
        scale_factor = 1.0

    selector = _CornerSelector(display_img)
    pts_display = selector.run()

    if pts_display is None:
        logger.error("Calibration cancelled — not enough points selected.")
        return

    # Scale points back to original image space
    pts_original = pts_display * scale_factor

    cal = Calibration.from_points(pts_original)
    cal.save(output_path)

    # Show preview of the warped result
    warped = cal.warp(image)
    cv2.imshow("Warped board preview (press any key to close)", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info("Calibration complete. Saved to %s", output_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="ChessVision camera calibration tool"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a raw board photo for calibration",
    )
    parser.add_argument(
        "--output",
        default="data/calibration.json",
        help="Where to save calibration.json (default: data/calibration.json)",
    )
    args = parser.parse_args()
    run_interactive(args.image, args.output)
