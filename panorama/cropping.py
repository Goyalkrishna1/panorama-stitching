"""
Automatic Canvas Cropping and Boundary Optimization
===================================================
Removes black warped canvas borders:
- Largest Interior Rectangle (LIR): O(W * H) stack-based histogram dynamic programming
  finding the maximum area rectangle containing 100% valid image pixels.
- Tight Bounding Box: Strips empty border frames.
- None (Raw): Leaves canvas uncropped.
"""

from typing import Tuple
import cv2
import numpy as np


class CanvasCropper:
    """Smart cropping algorithms for stitched panoramas."""

    @staticmethod
    def crop_tight_bbox(image: np.ndarray, threshold: int = 2) -> np.ndarray:
        """Crop outer empty border margins."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Union bounding box of all valid contours
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        if x_max > x_min and y_max > y_min:
            return image[y_min:y_max, x_min:x_max]
        return image

    @classmethod
    def crop_largest_interior_rectangle(
        cls,
        image: np.ndarray,
        threshold: int = 2,
    ) -> np.ndarray:
        """
        Finds and extracts the largest axis-aligned rectangle completely inside
        the non-black region of the panorama using histogram stack analysis.

        Time complexity: O(H * W)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = (gray > threshold).astype(np.uint8)

        h, w = binary.shape
        heights = np.zeros(w, dtype=np.int32)
        max_area = 0
        best_rect = (0, 0, w, h)  # (x, y, w, h)

        for row in range(h):
            # Update heights: if pixel is valid, increment; else reset to 0
            heights = np.where(binary[row] == 1, heights + 1, 0)

            # Largest rectangle under histogram using stack
            stack = []
            for col, height in enumerate(heights):
                start = col
                while stack and stack[-1][1] > height:
                    s_idx, s_h = stack.pop()
                    area = s_h * (col - s_idx)
                    if area > max_area:
                        max_area = area
                        best_rect = (s_idx, row - s_h + 1, col - s_idx, s_h)
                    start = s_idx
                stack.append((start, height))

            while stack:
                s_idx, s_h = stack.pop()
                area = s_h * (w - s_idx)
                if area > max_area:
                    max_area = area
                    best_rect = (s_idx, row - s_h + 1, w - s_idx, s_h)

        rx, ry, rw, rh = best_rect
        if rw > 20 and rh > 20:
            return image[ry : ry + rh, rx : rx + rw]

        return cls.crop_tight_bbox(image, threshold)
