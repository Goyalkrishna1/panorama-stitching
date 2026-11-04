"""
Image Blending and Seam Fusion Module
=====================================
Combines multiple warped images on a unified global canvas with seam reduction:
- Simple Overlay
- Linear Distance-Transform Feathering (Weighted Alpha Blending)
- Burt & Adelson Multi-Band Laplacian Pyramid Blending
- Optional Photometric Gain Compensation (Exposure Equalization)
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


class SeamBlender:
    """Multi-method seam blender for overlapping warped images."""

    MODES = ["multiband", "feather", "simple"]

    def __init__(
        self,
        mode: str = "multiband",
        pyramid_levels: int = 4,
        enable_gain_compensation: bool = True,
    ):
        """
        Initialize blender.

        Args:
            mode: 'multiband', 'feather', or 'simple'.
            pyramid_levels: Octave levels for Laplacian pyramid (typically 3 - 5).
            enable_gain_compensation: Equalize luminance across overlap zones.
        """
        self.mode = mode.lower()
        self.pyramid_levels = pyramid_levels
        self.enable_gain_compensation = enable_gain_compensation

        if self.mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {self.MODES}")

    def blend(
        self,
        warped_images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """
        Blend a list of warped images situated on the same canvas.

        Args:
            warped_images: List of BGR images (all H x W x 3).
            masks: List of binary uint8 masks (all H x W, values 0 or 255).

        Returns:
            Blended composite BGR image (H x W x 3).
        """
        if not warped_images:
            raise ValueError("No images provided to blender.")

        if len(warped_images) == 1:
            return warped_images[0].copy()

        # Optional Gain Compensation across overlapping regions
        work_images = [img.copy() for img in warped_images]
        if self.enable_gain_compensation:
            work_images = self._apply_gain_compensation(work_images, masks)

        if self.mode == "simple":
            return self._blend_simple(work_images, masks)
        elif self.mode == "feather":
            return self._blend_feather(work_images, masks)
        else:  # multiband
            return self._blend_multiband_pairwise(work_images, masks)

    def _apply_gain_compensation(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Normalize brightness across consecutive overlapping image pairs.
        """
        n = len(images)
        out = [images[0].copy()]

        for i in range(1, n):
            overlap = (masks[i - 1] > 0) & (masks[i] > 0)
            img_curr = images[i].astype(np.float32)

            if np.count_nonzero(overlap) > 100:
                # Compute per-channel mean ratio in overlap region
                ref_means = [np.mean(images[i - 1][..., c][overlap]) for c in range(3)]
                curr_means = [np.mean(img_curr[..., c][overlap]) for c in range(3)]

                for c in range(3):
                    if curr_means[c] > 1e-3:
                        gain = ref_means[c] / curr_means[c]
                        # Clamp gain to reasonable range (0.7 to 1.4) to avoid blown highlights
                        gain = float(np.clip(gain, 0.7, 1.4))
                        img_curr[..., c] = img_curr[..., c] * gain

            out.append(np.clip(img_curr, 0, 255).astype(np.uint8))

        return out

    def _blend_simple(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Sequential overlay (last image takes priority)."""
        h, w = images[0].shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for img, mask in zip(images, masks):
            valid = mask > 0
            canvas[valid] = img[valid]
        return canvas

    def _blend_feather(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """
        Multi-image distance-transform feathering.
        Computes continuous alpha weights proportional to distance from borders.
        """
        h, w = images[0].shape[:2]
        weight_sum = np.zeros((h, w), dtype=np.float32)
        blended = np.zeros((h, w, 3), dtype=np.float32)

        # Compute distance transforms for each mask
        weight_maps: List[np.ndarray] = []
        for mask in masks:
            if np.count_nonzero(mask) == 0:
                weight_maps.append(np.zeros((h, w), dtype=np.float32))
                continue
            # Euclidean distance to nearest zero-pixel
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            dist_max = dist.max()
            if dist_max > 0:
                dist /= dist_max
            weight_maps.append(dist)
            weight_sum += dist

        # Normalize weights and accumulate
        epsilon = 1e-6
        for img, w_map in zip(images, weight_maps):
            norm_w = w_map / (weight_sum + epsilon)
            norm_w_3d = np.repeat(norm_w[:, :, np.newaxis], 3, axis=2)
            blended += img.astype(np.float32) * norm_w_3d

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _blend_multiband_pairwise(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """
        Accumulates panorama by pairwise Burt-Adelson multi-band Laplacian blending.
        """
        n = len(images)
        accum_img = images[0].copy()
        accum_mask = masks[0].copy()

        for i in range(1, n):
            accum_img, accum_mask = self._pyramid_blend_pair(
                accum_img, accum_mask, images[i], masks[i]
            )

        return accum_img

    def _pyramid_blend_pair(
        self,
        img_a: np.ndarray,
        mask_a: np.ndarray,
        img_b: np.ndarray,
        mask_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend two images using Laplacian Pyramids over their boundary."""
        h, w = img_a.shape[:2]

        # Combined mask
        valid_a = mask_a > 0
        valid_b = mask_b > 0
        overlap = valid_a & valid_b

        # If no overlap, simply combine
        if not np.any(overlap):
            out = np.zeros_like(img_a)
            out[valid_a] = img_a[valid_a]
            out[valid_b] = img_b[valid_b]
            union_mask = (mask_a | mask_b).astype(np.uint8)
            return out, union_mask

        # Compute seam weight mask from distance transform
        dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5).astype(np.float32)
        dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5).astype(np.float32)

        weight_a = dist_a / (dist_a + dist_b + 1e-6)
        # Binarize weight mask in non-overlap regions
        weight_a[valid_a & ~valid_b] = 1.0
        weight_a[~valid_a & valid_b] = 0.0

        # Construct Gaussian and Laplacian Pyramids
        # Pad dimensions to be divisible by 2^levels
        divisor = 2 ** self.pyramid_levels
        pad_h = (divisor - (h % divisor)) % divisor
        pad_w = (divisor - (w % divisor)) % divisor

        if pad_h > 0 or pad_w > 0:
            padded_a = cv2.copyMakeBorder(
                img_a, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            ).astype(np.float32)
            padded_b = cv2.copyMakeBorder(
                img_b, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            ).astype(np.float32)
            padded_weight = cv2.copyMakeBorder(
                weight_a, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            ).astype(np.float32)
        else:
            padded_a = img_a.astype(np.float32)
            padded_b = img_b.astype(np.float32)
            padded_weight = weight_a.astype(np.float32)

        # Build Laplacian pyramids for images and Gaussian pyramid for weights
        lap_a = self._build_laplacian_pyramid(padded_a, self.pyramid_levels)
        lap_b = self._build_laplacian_pyramid(padded_b, self.pyramid_levels)
        gauss_w = self._build_gaussian_pyramid(padded_weight, self.pyramid_levels)

        # Blend each pyramid level: L_blend = G_w * L_a + (1 - G_w) * L_b
        blended_pyr = []
        for la, lb, gw in zip(lap_a, lap_b, gauss_w):
            gw_3d = np.repeat(gw[:, :, np.newaxis], 3, axis=2)
            blended_level = gw_3d * la + (1.0 - gw_3d) * lb
            blended_pyr.append(blended_level)

        # Reconstruct image from Laplacian pyramid
        reconstructed = self._reconstruct_from_laplacian(blended_pyr)

        # Unpad
        blended_crop = reconstructed[:h, :w]
        blended_uint8 = np.clip(blended_crop, 0, 255).astype(np.uint8)

        union_mask = ((mask_a > 0) | (mask_b > 0)).astype(np.uint8) * 255
        return blended_uint8, union_mask

    def _build_gaussian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        pyr = [img]
        curr = img
        for _ in range(levels):
            curr = cv2.pyrDown(curr)
            pyr.append(curr)
        return pyr

    def _build_laplacian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        gauss_pyr = self._build_gaussian_pyramid(img, levels)
        lap_pyr = []
        for i in range(levels):
            h_next, w_next = gauss_pyr[i].shape[:2]
            up = cv2.pyrUp(gauss_pyr[i + 1], dstsize=(w_next, h_next))
            lap = gauss_pyr[i] - up
            lap_pyr.append(lap)
        lap_pyr.append(gauss_pyr[-1])
        return lap_pyr

    def _reconstruct_from_laplacian(self, pyr: List[np.ndarray]) -> np.ndarray:
        curr = pyr[-1]
        for i in range(len(pyr) - 2, -1, -1):
            h_target, w_target = pyr[i].shape[:2]
            up = cv2.pyrUp(curr, dstsize=(w_target, h_target))
            curr = up + pyr[i]
        return curr
