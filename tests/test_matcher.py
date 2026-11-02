"""
Unit Tests: Feature Matcher & Geometry
"""

import unittest
import numpy as np
import cv2

from panorama.detector import FeatureDetector
from panorama.matcher import FeatureMatcher
from panorama.geometry import TransformationEstimator


class TestMatcherAndGeometry(unittest.TestCase):

    def setUp(self):
        # Create image 1: synthetic patterned square
        self.img1 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(self.img1, (200, 200), 80, (255, 255, 255), -1)
        cv2.rectangle(self.img1, (100, 100), (300, 300), (128, 128, 128), 3)
        cv2.putText(self.img1, "IPCV", (150, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Create image 2: translated & slightly rotated version of image 1
        M = cv2.getRotationMatrix2D((200, 200), 5.0, 1.0)
        M[0, 2] += 25.0
        M[1, 2] += 15.0
        self.img2 = cv2.warpAffine(self.img1, M, (400, 400))

    def test_matching_and_homography(self):
        det = FeatureDetector(algorithm="SIFT", max_features=1000)
        kp1, desc1 = det.detect_and_compute(self.img1)
        kp2, desc2 = det.detect_and_compute(self.img2)

        matcher = FeatureMatcher(algorithm="SIFT", ratio_threshold=0.8)
        matches = matcher.match(desc1, desc2)
        self.assertGreater(len(matches), 10)

        estimator = TransformationEstimator(model="affine", min_inliers=6)
        H, mask, telem = estimator.estimate(kp2, kp1, matches)

        self.assertIsNotNone(H)
        self.assertIsNotNone(mask)
        self.assertTrue(telem["success"])
        self.assertGreater(telem["num_inliers"], 5)


if __name__ == "__main__":
    unittest.main()
