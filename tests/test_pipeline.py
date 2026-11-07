"""
Unit Tests: Full Pipeline Integration
"""

import unittest
import glob
import cv2
import numpy as np

from panorama.pipeline import PanoramaEngine, PanoramaResult


class TestPipeline(unittest.TestCase):

    def test_pipeline_on_demo_data(self):
        # Test on the scenic city images
        files = sorted(glob.glob("demo_images/scenic_city/*.jpg"))[:3]
        self.assertGreaterEqual(len(files), 2)

        images = [cv2.imread(f) for f in files]
        for img in images:
            self.assertIsNotNone(img)

        engine = PanoramaEngine(
            detector="SIFT",
            transform_model="affine",
            projection="cylindrical",
            fov_degrees=60.0,
            blend_mode="feather",
            crop_mode="tight",
        )

        result: PanoramaResult = engine.stitch(images)
        self.assertIsNotNone(result.panorama)
        h, w = result.panorama.shape[:2]
        self.assertGreater(w, images[0].shape[1])  # Panorama must be wider than single image
        self.assertGreater(h, 50)


if __name__ == "__main__":
    unittest.main()
