"""
Geometric Transformation and Alignment Module
==============================================
Estimates 2D planar transformations from matched keypoint correspondences using RANSAC.
Transformation models:
- Projective Homography (8-DOF, 3x3)
- Partial Affine / Similarity (4-DOF: rotation + uniform scale + translation)
- Rigid Euclidean (3-DOF: rotation + translation)
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


class TransformationEstimator:
    """Robust estimator for 2D coordinate transformations between image pairs."""

    MODELS = ["homography", "affine", "euclidean"]

    def __init__(
        self,
        model: str = "affine",
        ransac_threshold: float = 4.0,
        min_inliers: int = 8,
        max_iters: int = 2000,
        confidence: float = 0.995,
    ):
        """
        Initialize estimator.

        Args:
            model: 'affine' (uniform height/width, recommended) or 'homography' (full projective).
            ransac_threshold: Maximum reprojection error (pixels) for a point to be an inlier.
            min_inliers: Minimum number of inliers required to declare success.
            max_iters: Maximum RANSAC iterations.
            confidence: Desired RANSAC confidence level.
        """
        self.model = model.lower()
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.max_iters = max_iters
        self.confidence = confidence

        if self.model not in self.MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {self.MODELS}")

    def estimate(
        self,
        kp_src: List[cv2.KeyPoint],
        kp_dst: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimate transformation H mapping source image coordinates to destination coordinates.

        Points:
            p_dst ~ H @ p_src

        Args:
            kp_src: Keypoints in source image (mapped from trainIdx).
            kp_dst: Keypoints in destination image (mapped from queryIdx).
            matches: Matches between dst (query) and src (train).

        Returns:
            Tuple of (H_matrix_3x3, inlier_mask, telemetry_dict).
        """
        telemetry: Dict[str, Any] = {
            "num_matches": len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": 0.0,
            "model": self.model,
            "success": False,
        }

        min_pts_required = 4 if self.model == "homography" else 3
        if len(matches) < min_pts_required:
            return None, None, telemetry

        pts_src = np.float32([kp_src[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_dst = np.float32([kp_dst[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        H_3x3: Optional[np.ndarray] = None
        mask: Optional[np.ndarray] = None

        if self.model == "affine":
            # 4-DOF Similarity: Uniform scale + rotation + 2D translation
            # Eliminates keystone/wedge expansion while allowing rotation and translation
            M, inliers_mask = cv2.estimateAffinePartial2D(
                pts_src,
                pts_dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )
            if M is not None:
                H_3x3 = np.eye(3, dtype=np.float64)
                H_3x3[:2, :] = M
                mask = inliers_mask

        elif self.model == "euclidean":
            # 3-DOF Rigid: Rotation + 2D translation (scale fixed to 1.0)
            M, inliers_mask = cv2.estimateAffine2D(
                pts_src,
                pts_dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )
            if M is not None:
                H_3x3 = np.eye(3, dtype=np.float64)
                H_3x3[:2, :] = M
                mask = inliers_mask

        else:  # 'homography'
            # 8-DOF Full Projective transformation
            H_est, inliers_mask = cv2.findHomography(
                pts_src,
                pts_dst,
                cv2.RANSAC,
                self.ransac_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )
            if H_est is not None:
                H_3x3 = H_est.astype(np.float64)
                mask = inliers_mask

        if H_3x3 is None or mask is None:
            return None, None, telemetry

        inlier_count = int(np.sum(mask))
        telemetry["num_inliers"] = inlier_count
        telemetry["inlier_ratio"] = float(inlier_count / len(matches))

        if inlier_count < self.min_inliers:
            return None, None, telemetry

        # Verify matrix conditioning (prevent singular or degenerate matrices)
        det = np.linalg.det(H_3x3)
        if abs(det) < 1e-6 or np.isnan(det) or np.isinf(det):
            return None, None, telemetry

        # Compute Root Mean Square Reprojection Error (RMSE) on inliers
        inlier_indices = np.where(mask.ravel() == 1)[0]
        if len(inlier_indices) > 0:
            inliers_src = pts_src[inlier_indices]
            inliers_dst = pts_dst[inlier_indices]

            projected_dst = cv2.perspectiveTransform(inliers_src, H_3x3)
            errors = np.linalg.norm(inliers_dst - projected_dst, axis=2).ravel()
            telemetry["rmse"] = float(np.sqrt(np.mean(errors ** 2)))

        telemetry["success"] = True
        return H_3x3, mask, telemetry
