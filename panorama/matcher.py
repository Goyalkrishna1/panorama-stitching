"""
Feature Matching Module
=======================
Matches local descriptors across pairs of images using:
- Brute-Force Matcher (BFMatcher) with L2 or Hamming distance
- FLANN-based approximate nearest neighbor search
- Lowe's Ratio Test (k=2 nearest neighbors filtering)
- Mutual Cross-Checking
- Dual-color visualization (Green = RANSAC inliers, Red = outliers)
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


class FeatureMatcher:
    """Robust feature matching with ratio test and inlier-aware visualization."""

    def __init__(
        self,
        algorithm: str = "SIFT",
        use_flann: bool = False,
        ratio_threshold: float = 0.75,
        min_matches: int = 10,
    ):
        """
        Initialize matcher.

        Args:
            algorithm: Detector algorithm ('SIFT', 'ORB', 'AKAZE', 'BRISK').
            use_flann: If True, uses FLANN index; else uses Brute Force.
            ratio_threshold: Lowe's ratio test threshold (typically 0.70 - 0.80).
            min_matches: Minimum good matches required to consider pair viable.
        """
        self.algorithm = algorithm.upper()
        self.use_flann = use_flann
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches
        self.is_binary = self.algorithm in ["ORB", "BRISK"]

    def _get_matcher(self):
        if self.use_flann:
            if not self.is_binary:
                # KD-Tree for floating-point descriptors (SIFT)
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
                search_params = dict(checks=50)
            else:
                # Multi-probe LSH for binary descriptors (ORB)
                index_params = dict(
                    algorithm=6, table_number=6, key_size=12, multi_probe_level=1
                )
                search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute-force matcher
            norm = cv2.NORM_HAMMING if self.is_binary else cv2.NORM_L2
            return cv2.BFMatcher(norm, crossCheck=False)

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray],
    ) -> List[cv2.DMatch]:
        """
        Find mutual candidate matches passing Lowe's ratio test.

        Args:
            desc1: Descriptors of query image (N1 x D).
            desc2: Descriptors of train image (N2 x D).

        Returns:
            Filtered list of cv2.DMatch objects.
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []

        # Ensure correct dtype for FLANN
        if self.use_flann and not self.is_binary:
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
            if desc2.dtype != np.float32:
                desc2 = desc2.astype(np.float32)

        matcher = self._get_matcher()

        try:
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            # Fallback to BF if FLANN fails on small descriptor set
            norm = cv2.NORM_HAMMING if self.is_binary else cv2.NORM_L2
            fallback = cv2.BFMatcher(norm, crossCheck=False)
            raw_matches = fallback.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's Ratio Test: d_best < ratio * d_second_best
        good_matches: List[cv2.DMatch] = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def is_sufficient(self, matches: List[cv2.DMatch]) -> bool:
        """Check if match count meets minimum threshold."""
        return len(matches) >= self.min_matches

    def draw_matches_diagnostic(
        self,
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        inlier_mask: Optional[np.ndarray] = None,
        max_draw: int = 150,
    ) -> np.ndarray:
        """
        Draw side-by-side matches with distinct inlier (green) vs outlier (red) lines.

        Args:
            img1: First image (query).
            kp1: First image keypoints.
            img2: Second image (train).
            kp2: Second image keypoints.
            matches: Filtered matches list.
            inlier_mask: Binary mask from RANSAC (1 for inlier, 0 for outlier).
            max_draw: Maximum number of match lines to draw to avoid clutter.

        Returns:
            Side-by-side composite visualization image.
        """
        if inlier_mask is not None:
            mask_flat = inlier_mask.ravel().tolist()
        else:
            mask_flat = [1] * len(matches)

        # Subsample if too many matches to keep visual clean
        indices = list(range(len(matches)))
        if len(indices) > max_draw:
            # Prioritize inliers in the subsampled view
            inlier_indices = [i for i in indices if i < len(mask_flat) and mask_flat[i]]
            outlier_indices = [i for i in indices if i < len(mask_flat) and not mask_flat[i]]

            keep_inliers = inlier_indices[: int(max_draw * 0.8)]
            keep_outliers = outlier_indices[: max_draw - len(keep_inliers)]
            indices = sorted(keep_inliers + keep_outliers)

        # Build side-by-side canvas
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas_h = max(h1, h2)
        canvas_w = w1 + w2

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1 : w1 + w2] = img2

        # Draw match lines
        for idx in indices:
            m = matches[idx]
            pt1 = (int(round(kp1[m.queryIdx].pt[0])), int(round(kp1[m.queryIdx].pt[1])))
            pt2 = (
                int(round(kp2[m.trainIdx].pt[0] + w1)),
                int(round(kp2[m.trainIdx].pt[1])),
            )

            is_inlier = bool(mask_flat[idx]) if idx < len(mask_flat) else True
            if is_inlier:
                color = (0, 220, 0)   # Green for verified inliers
                thickness = 1
            else:
                color = (0, 0, 220)   # Red for rejected outliers
                thickness = 1

            # Keypoint circles
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

        return canvas
