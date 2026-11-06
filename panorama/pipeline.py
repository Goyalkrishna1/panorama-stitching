"""
Panorama Stitching Pipeline Engine
===================================
End-to-end orchestration pipeline coordinating:
1. Cylindrical / Spherical Surface Projection
2. Multi-Algorithm Feature Detection (SIFT, ORB, AKAZE, BRISK)
3. Ratio-Test Pairwise Matching & Inlier/Outlier Verification
4. Graph Topology Sequencing (Handles Shuffled / Unordered Photos)
5. Central Reference Coordinate Alignment (Minimizes Cumulative Error)
6. Global Canvas Transformation & Bounding Box Estimation
7. Seamless Multi-Band Laplacian Pyramid or Feathered Blending
8. Automated Largest Interior Rectangle (LIR) Cropping
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple
import time
import cv2
import numpy as np

from .detector import FeatureDetector
from .matcher import FeatureMatcher
from .geometry import TransformationEstimator
from .projection import SurfaceProjector
from .sequencer import ImageSequencer
from .blending import SeamBlender
from .cropping import CanvasCropper


@dataclass
class PairwiseReport:
    """Diagnostic data for a single image pair."""
    src_idx: int
    dst_idx: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    rmse: float
    transform_matrix: Optional[np.ndarray]
    match_visualization: Optional[np.ndarray] = None


@dataclass
class PanoramaResult:
    """Complete output and diagnostic telemetry of the stitching engine."""
    panorama: Optional[np.ndarray]
    raw_untrimmed: Optional[np.ndarray]
    stitching_order: List[int]
    pairwise_reports: List[PairwiseReport]
    skipped_images: List[int]
    intermediate_steps: List[np.ndarray] = field(default_factory=list)
    adjacency_matrix: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class PanoramaEngine:
    """High-accuracy multi-image panorama reconstruction engine."""

    def __init__(
        self,
        detector: str = "SIFT",
        max_features: int = 5000,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 4.0,
        min_matches: int = 10,
        min_inliers: int = 8,
        transform_model: str = "affine",  # "affine" (similarity), "homography", "euclidean"
        projection: str = "cylindrical",  # "cylindrical", "spherical", "planar"
        fov_degrees: float = 60.0,
        blend_mode: str = "multiband",    # "multiband", "feather", "simple"
        crop_mode: str = "rect",         # "rect", "tight", "none"
        enable_gain_comp: bool = True,
        engine: str = "custom",          # "custom" or "opencv"
    ):
        self.detector_name = detector.upper()
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.transform_model = transform_model.lower()
        self.projection = projection.lower()
        self.fov_degrees = fov_degrees
        self.blend_mode = blend_mode.lower()
        self.crop_mode = crop_mode.lower()
        self.enable_gain_comp = enable_gain_comp
        self.engine = engine.lower()

        # Initialize core components
        self.detector = FeatureDetector(algorithm=self.detector_name, max_features=max_features)
        self.matcher = FeatureMatcher(
            algorithm=self.detector_name,
            ratio_threshold=ratio_threshold,
            min_matches=min_matches,
        )
        self.estimator = TransformationEstimator(
            model=self.transform_model,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )
        self.blender = SeamBlender(
            mode=self.blend_mode,
            enable_gain_compensation=enable_gain_comp,
        )

    def stitch(
        self,
        images: List[np.ndarray],
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> PanoramaResult:
        """
        Execute full end-to-end panorama reconstruction.

        Args:
            images: List of BGR images (uint8).
            progress_cb: Optional callback func(status_message, progress_fraction_0_to_1).

        Returns:
            PanoramaResult containing final panorama and full diagnostic suite.
        """
        logs: List[str] = []
        t_start = time.time()

        def _log(msg: str, frac: float):
            logs.append(msg)
            if progress_cb:
                progress_cb(msg, min(frac, 1.0))

        n = len(images)
        if n == 0:
            return PanoramaResult(None, None, [], [], [], logs=logs)
        if n == 1:
            return PanoramaResult(images[0].copy(), images[0].copy(), [0], [], [], logs=logs)

        # ── OpenCV Native Engine (Optional Baseline) ─────────────────────────
        if self.engine == "opencv":
            _log("Invoking OpenCV cv2.Stitcher baseline engine …", 0.3)
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, pano = stitcher.stitch(images)
            if status == cv2.Stitcher_OK:
                _log(f"OpenCV stitch succeeded ({pano.shape[1]}x{pano.shape[0]} px)", 1.0)
                cropped = pano
                if self.crop_mode == "rect":
                    cropped = CanvasCropper.crop_largest_interior_rectangle(pano)
                elif self.crop_mode == "tight":
                    cropped = CanvasCropper.crop_tight_bbox(pano)

                diag = {
                    "execution_time_sec": time.time() - t_start,
                    "engine": "OpenCV Native",
                    "num_images": n,
                }
                return PanoramaResult(
                    panorama=cropped,
                    raw_untrimmed=pano,
                    stitching_order=list(range(n)),
                    pairwise_reports=[],
                    skipped_images=[],
                    intermediate_steps=[pano],
                    diagnostics=diag,
                    logs=logs,
                )
            else:
                _log("OpenCV native stitcher failed. Falling back to custom pipeline …", 0.1)

        # ── Stage 1: Surface Projection (Cylindrical / Spherical) ───────────
        if self.projection in ["cylindrical", "spherical"]:
            _log(f"Projecting images onto {self.projection} surface (FOV: {self.fov_degrees}°) …", 0.05)
            f_px = SurfaceProjector.calculate_focal_length(
                images[0].shape[1], self.fov_degrees
            )
            _log(f"Calculated focal length: {f_px:.1f} pixels", 0.08)

            work_images: List[np.ndarray] = []
            for idx, img in enumerate(images):
                if self.projection == "cylindrical":
                    warped = SurfaceProjector.warp_cylindrical(img, f_px)
                else:
                    warped = SurfaceProjector.warp_spherical(img, f_px)
                trimmed = SurfaceProjector.trim_curved_black_borders(warped)
                work_images.append(trimmed)
                _log(f"  Projected image {idx+1}/{n} ({trimmed.shape[1]}x{trimmed.shape[0]} px)", 0.08 + 0.08 * (idx + 1) / n)
        else:
            _log("Using Planar perspective projection …", 0.05)
            work_images = [img.copy() for img in images]

        # ── Stage 2: Feature Detection & Description ─────────────────────────
        _log(f"Detecting features with {self.detector_name} …", 0.18)
        all_keypoints: List = []
        all_descriptors: List = []

        for idx, img in enumerate(work_images):
            kp, desc = self.detector.detect_and_compute(img)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            _log(f"  Image {idx+1}/{n}: {len(kp)} keypoints detected", 0.18 + 0.10 * (idx + 1) / n)

        # ── Stage 3: Pairwise Matching & Connectivity Graph ──────────────────
        _log("Performing exhaustive pairwise feature matching & RANSAC verification …", 0.30)
        sequencer = ImageSequencer(n)
        pairwise_reports: List[PairwiseReport] = []

        total_pairs = n * (n - 1) // 2
        pair_counter = 0

        for i in range(n):
            for j in range(i + 1, n):
                matches = self.matcher.match(all_descriptors[i], all_descriptors[j])
                H, mask, telem = None, None, {"num_inliers": 0, "inlier_ratio": 0.0, "rmse": 0.0}

                if self.matcher.is_sufficient(matches):
                    H, mask, telem = self.estimator.estimate(
                        all_keypoints[j], all_keypoints[i], matches
                    )

                inliers = telem["num_inliers"]
                sequencer.add_edge(i, j, inliers if H is not None else 0)

                # Diagnostic match visualization
                match_vis = None
                if matches:
                    try:
                        match_vis = self.matcher.draw_matches_diagnostic(
                            work_images[i], all_keypoints[i],
                            work_images[j], all_keypoints[j],
                            matches, mask,
                        )
                    except Exception:
                        pass

                pairwise_reports.append(
                    PairwiseReport(
                        src_idx=j,
                        dst_idx=i,
                        num_matches=len(matches),
                        num_inliers=inliers,
                        inlier_ratio=telem["inlier_ratio"],
                        rmse=telem["rmse"],
                        transform_matrix=H,
                        match_visualization=match_vis,
                    )
                )

                pair_counter += 1
                status_icon = "✓" if H is not None else "✗"
                _log(f"  Pair ({i+1},{j+1}): {len(matches)} matches, {inliers} inliers {status_icon}",
                     0.30 + 0.20 * pair_counter / total_pairs)

        adj_matrix = sequencer.get_adjacency_matrix()

        # ── Stage 4: Optimal Stitching Sequence ──────────────────────────────
        stitch_order = sequencer.determine_stitching_order()
        _log(f"Determined optimal spatial order: {[idx+1 for idx in stitch_order]}", 0.52)

        ordered_work = [work_images[idx] for idx in stitch_order]
        ordered_kp = [all_keypoints[idx] for idx in stitch_order]
        ordered_desc = [all_descriptors[idx] for idx in stitch_order]
        n_ordered = len(stitch_order)

        # ── Stage 5: Central Reference Frame Transformation Chaining ─────────
        _log("Computing transformations referenced to central anchor image …", 0.58)

        # Pairwise step transforms: T[i] maps image (i+1) into image (i)
        step_transforms: List[Optional[np.ndarray]] = []
        for i in range(n_ordered - 1):
            matches = self.matcher.match(ordered_desc[i], ordered_desc[i + 1])
            H, mask, telem = self.estimator.estimate(
                ordered_kp[i + 1], ordered_kp[i], matches
            )
            if H is not None:
                step_transforms.append(H)
                _log(f"  Link {i+1} -> {i+2}: verified ({telem['num_inliers']} inliers)", 0.58 + 0.12 * (i + 1) / n_ordered)
            else:
                step_transforms.append(None)
                _log(f"  Link {i+1} -> {i+2}: unverified (insufficient overlap)", 0.58 + 0.12 * (i + 1) / n_ordered)

        # Anchor is the middle image of the sequence
        mid_idx = n_ordered // 2
        _log(f"Using Image {stitch_order[mid_idx]+1} as central reference frame (zero distortion anchor)", 0.72)

        # Accumulate transforms to map each image directly to reference frame
        H_to_ref: List[Optional[np.ndarray]] = [None] * n_ordered
        H_to_ref[mid_idx] = np.eye(3, dtype=np.float64)

        # Forward propagation (rightwards: mid -> end)
        for i in range(mid_idx + 1, n_ordered):
            T_prev = step_transforms[i - 1]
            if T_prev is not None and H_to_ref[i - 1] is not None:
                H_to_ref[i] = H_to_ref[i - 1] @ T_prev

        # Backward propagation (leftwards: mid -> 0)
        for i in range(mid_idx - 1, -1, -1):
            T_curr = step_transforms[i]  # maps (i+1) to i -> inv maps i to (i+1)
            if T_curr is not None and H_to_ref[i + 1] is not None:
                try:
                    H_to_ref[i] = H_to_ref[i + 1] @ np.linalg.inv(T_curr)
                except np.linalg.LinAlgError:
                    pass

        # Identify valid aligned subset
        valid_indices = [i for i, H in enumerate(H_to_ref) if H is not None]
        skipped_indices = [stitch_order[i] for i, H in enumerate(H_to_ref) if H is None]

        if not valid_indices:
            _log("Stitching failed: unable to geometrically align sequence.", 1.0)
            return PanoramaResult(None, None, stitch_order, pairwise_reports, stitch_order, logs=logs)

        valid_images = [ordered_work[i] for i in valid_indices]
        valid_transforms = [H_to_ref[i] for i in valid_indices]

        # ── Stage 6: Global Canvas Dimensioning & Warping ────────────────────
        _log(f"Computing unified canvas bounds for {len(valid_images)} images …", 0.78)

        all_corners = []
        for img, H in zip(valid_images, valid_transforms):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_c = cv2.perspectiveTransform(corners, H)
            all_corners.append(warped_c)

        all_corners = np.concatenate(all_corners, axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel()) - 2
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel()) + 2

        canvas_w = int(x_max - x_min)
        canvas_h = int(y_max - y_min)

        # Sanity bound clamps to prevent out-of-memory
        canvas_w = max(50, min(canvas_w, 20000))
        canvas_h = max(50, min(canvas_h, 10000))

        # Shift origin into positive quadrant
        T_offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        # Warp each image into global canvas coordinates
        warped_layers: List[np.ndarray] = []
        layer_masks: List[np.ndarray] = []
        intermediates: List[np.ndarray] = []

        step_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, (img, H) in enumerate(zip(valid_images, valid_transforms)):
            H_canvas = T_offset @ H
            warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h))
            mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 1).astype(np.uint8) * 255

            warped_layers.append(warped)
            layer_masks.append(mask)

            # Step view
            valid_px = mask > 0
            step_canvas[valid_px] = warped[valid_px]
            intermediates.append(CanvasCropper.crop_tight_bbox(step_canvas))

        # ── Stage 7: Blending ────────────────────────────────────────────────
        _log(f"Fusing images using {self.blend_mode} blending …", 0.88)
        raw_panorama = self.blender.blend(warped_layers, layer_masks)

        # ── Stage 8: Canvas Cropping ─────────────────────────────────────────
        _log(f"Applying canvas post-processing ({self.crop_mode}) …", 0.95)
        if self.crop_mode == "rect":
            final_panorama = CanvasCropper.crop_largest_interior_rectangle(raw_panorama)
        elif self.crop_mode == "tight":
            final_panorama = CanvasCropper.crop_tight_bbox(raw_panorama)
        else:
            final_panorama = raw_panorama

        total_time = time.time() - t_start
        _log(f"Reconstruction finished in {total_time:.2f}s! Final resolution: {final_panorama.shape[1]}x{final_panorama.shape[0]} px", 1.0)

        diagnostics = {
            "execution_time_sec": total_time,
            "stitched_count": len(valid_images),
            "total_input_count": n,
            "skipped_count": len(skipped_indices),
            "final_width": final_panorama.shape[1],
            "final_height": final_panorama.shape[0],
            "aspect_ratio": final_panorama.shape[1] / float(final_panorama.shape[0]),
            "detector": self.detector_name,
            "projection": self.projection,
            "transform_model": self.transform_model,
            "blend_mode": self.blend_mode,
            "crop_mode": self.crop_mode,
        }

        return PanoramaResult(
            panorama=final_panorama,
            raw_untrimmed=raw_panorama,
            stitching_order=stitch_order,
            pairwise_reports=pairwise_reports,
            skipped_images=skipped_indices,
            intermediate_steps=intermediates,
            adjacency_matrix=adj_matrix,
            diagnostics=diagnostics,
            logs=logs,
        )
