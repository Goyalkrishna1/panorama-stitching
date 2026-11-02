"""
Geometric Surface Projection Module
====================================
Projects planar perspective images onto non-planar surfaces (Cylindrical and Spherical)
prior to feature matching and stitching.

Mathematical Formulation (Cylindrical):
----------------------------------------
When camera rotates around vertical axis, points on cylinder of radius f satisfy:
    theta = (x_cyl - cx) / f
    h_cyl = (y_cyl - cy) / f

Inverse mapping for backward warping (remap):
    x_src = f * tan(theta) + cx
    y_src = (y_cyl - cy) / cos(theta) + cy
"""

from typing import Tuple
import cv2
import numpy as np


class SurfaceProjector:
    """Handles cylindrical and spherical projections of camera images."""

    @staticmethod
    def calculate_focal_length(image_width: int, fov_degrees: float = 60.0) -> float:
        """
        Estimate focal length in pixel units given image width and horizontal FOV.

        f = W / (2 * tan(FOV / 2))
        """
        fov_rad = np.radians(fov_degrees)
        focal_pixels = image_width / (2.0 * np.tan(fov_rad / 2.0))
        return float(focal_pixels)

    @classmethod
    def warp_cylindrical(
        cls,
        image: np.ndarray,
        focal_length: float,
    ) -> np.ndarray:
        """
        Project image onto a cylindrical surface using backward bilinear interpolation.

        Args:
            image: BGR uint8 input image.
            focal_length: Focal length in pixel coordinates.

        Returns:
            Cylindrically warped BGR image.
        """
        h, w = image.shape[:2]
        cx = w / 2.0
        cy = h / 2.0

        # Vectorized coordinate grid
        x_indices = np.arange(w, dtype=np.float64) - cx
        y_indices = np.arange(h, dtype=np.float64) - cy
        X, Y = np.meshgrid(x_indices, y_indices)

        theta = X / focal_length
        cos_theta = np.cos(theta)

        # Backward map: for each cylindrical (X, Y), find source (map_x, map_y)
        map_x = (focal_length * np.tan(theta) + cx).astype(np.float32)
        map_y = (Y / cos_theta + cy).astype(np.float32)

        # Interpolate source image onto target cylinder
        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped

    @classmethod
    def warp_spherical(
        cls,
        image: np.ndarray,
        focal_length: float,
    ) -> np.ndarray:
        """
        Project image onto a spherical surface.

        Args:
            image: BGR input image.
            focal_length: Focal length in pixels.

        Returns:
            Spherically warped BGR image.
        """
        h, w = image.shape[:2]
        cx = w / 2.0
        cy = h / 2.0

        x_indices = np.arange(w, dtype=np.float64) - cx
        y_indices = np.arange(h, dtype=np.float64) - cy
        X, Y = np.meshgrid(x_indices, y_indices)

        theta = X / focal_length
        phi = Y / focal_length

        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)

        map_x = (focal_length * np.tan(theta) + cx).astype(np.float32)
        map_y = (focal_length * (np.tan(phi) / cos_theta) + cy).astype(np.float32)

        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped

    @staticmethod
    def trim_curved_black_borders(image: np.ndarray, threshold: int = 5) -> np.ndarray:
        """
        Crop unmapped black vertical columns on left/right edges produced by projection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        column_brightness = np.max(gray, axis=0)
        valid_columns = np.where(column_brightness > threshold)[0]

        if len(valid_columns) == 0:
            return image

        x_start = valid_columns[0]
        x_end = valid_columns[-1]
        return image[:, x_start : x_end + 1]
