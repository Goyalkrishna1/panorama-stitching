"""
Unit Tests: Feature Detector
"""

import unittest
import numpy as np
import cv2

from panorama.detector import FeatureDetector


class TestFeatureDetector(unittest.TestCase):

    def setUp(self):
        # Create a synthetic checkerboard pattern
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(0, 300, 30):
            for j in range(0, 300, 30):
                if (i // 30 + j // 30) % 2 == 0:
                    self.test_img[i : i + 30, j : j + 30] = 255

    def test_sift_detector(self):
        detector = FeatureDetector(algorithm="SIFT", max_features=500)
        kps, descs = detector.detect_and_compute(self.test_img)
        self.assertGreater(len(kps), 0)
        self.assertIsNotNone(descs)
        self.assertEqual(descs.shape[1], 128)  # SIFT has 128-D descriptors

    def test_orb_detector(self):
        detector = FeatureDetector(algorithm="ORB", max_features=500)
        kps, descs = detector.detect_and_compute(self.test_img)
        self.assertGreater(len(kps), 0)
        self.assertIsNotNone(descs)
        self.assertEqual(descs.shape[1], 32)  # ORB has 32-byte (256-bit) descriptors

    def test_keypoint_drawing(self):
        detector = FeatureDetector(algorithm="SIFT", max_features=200)
        kps, _ = detector.detect_and_compute(self.test_img)
        vis = detector.draw_keypoints(self.test_img, kps, rich=True)
        self.assertEqual(vis.shape, self.test_img.shape)


if __name__ == "__main__":
    unittest.main()
