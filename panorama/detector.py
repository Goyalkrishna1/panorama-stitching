"""
Feature Detection and Description Module
========================================
Supports multiple keypoint detectors and descriptor extractors:
- SIFT (Scale-Invariant Feature Transform) - DoG extrema, scale/rotation invariant (Unit IV standard)
- ORB (Oriented FAST and Rotated BRIEF) - fast, binary descriptors
- AKAZE (Accelerated KAZE) - nonlinear scale spaces
- BRISK (Binary Robust Invariant Scalable Keypoints)
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


class FeatureDetector:
    """Configurable multi-algorithm keypoint detector and descriptor extractor."""

    SUPPORTED_ALGORITHMS = ["SIFT", "ORB", "AKAZE", "BRISK"]

    def __init__(
        self,
        algorithm: str = "SIFT",
        max_features: int = 5000,
        **kwargs: Any,
    ):
        """
        Initialize detector.

        Args:
            algorithm: Algorithm name ('SIFT', 'ORB', 'AKAZE', 'BRISK').
            max_features: Upper bound on keypoints to detect.
            **kwargs: Extra parameters passed to the underlying OpenCV constructor.
        """
        self.algorithm = algorithm.upper()
        self.max_features = max_features
        self.kwargs = kwargs

        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. Choose from: {self.SUPPORTED_ALGORITHMS}"
            )

        self._detector = self._build_detector()

    def _build_detector(self):
        algo = self.algorithm
        if algo == "SIFT":
            # SIFT uses DoG across scale octaves
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                contrastThreshold=self.kwargs.get("contrast_threshold", 0.04),
                edgeThreshold=self.kwargs.get("edge_threshold", 10.0),
                sigma=self.kwargs.get("sigma", 1.6),
            )
        elif algo == "ORB":
            # ORB uses FAST corners + Harris response + steered BRIEF descriptors
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=self.kwargs.get("scale_factor", 1.2),
                nlevels=self.kwargs.get("n_levels", 8),
                edgeThreshold=self.kwargs.get("edge_threshold", 31),
                fastThreshold=self.kwargs.get("fast_threshold", 20),
            )
        elif algo == "AKAZE":
            return cv2.AKAZE_create(
                threshold=self.kwargs.get("threshold", 0.001),
                nOctaves=self.kwargs.get("n_octaves", 4),
            )
        elif algo == "BRISK":
            return cv2.BRISK_create(
                thresh=self.kwargs.get("thresh", 30),
                octaves=self.kwargs.get("octaves", 3),
            )
        else:
            raise NotImplementedError(f"Algorithm {algo} not implemented.")

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to single-channel grayscale if needed."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract keypoints and compute descriptors.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Tuple of (list_of_KeyPoints, descriptors_array)
        """
        gray = self.to_grayscale(image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        if keypoints is None:
            keypoints = []

        return keypoints, descriptors

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        rich: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw keypoints over image.

        Args:
            image: Source BGR image.
            keypoints: List of cv2.KeyPoint.
            rich: If True, draws scale circles and orientation rays.
            color: Keypoint color (B, G, R).

        Returns:
            Annotated BGR image.
        """
        flags = (
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            if rich
            else cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        return cv2.drawKeypoints(image, keypoints, None, color=color, flags=flags)

    def get_statistics(self, keypoints: List[cv2.KeyPoint]) -> Dict[str, Any]:
        """Compute summary statistics for detected keypoints."""
        if not keypoints:
            return {
                "count": 0,
                "avg_response": 0.0,
                "avg_size": 0.0,
                "algorithm": self.algorithm,
            }

        responses = [kp.response for kp in keypoints]
        sizes = [kp.size for kp in keypoints]
        return {
            "count": len(keypoints),
            "avg_response": float(np.mean(responses)),
            "avg_size": float(np.mean(sizes)),
            "max_response": float(np.max(responses)),
            "algorithm": self.algorithm,
        }
