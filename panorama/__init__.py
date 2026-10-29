"""
Panorama Reconstruction & Image Stitching Package
=================================================
A modular computer vision library for multi-image panorama synthesis.
"""

from .detector import FeatureDetector
from .matcher import FeatureMatcher
from .geometry import TransformationEstimator
from .projection import SurfaceProjector
from .sequencer import ImageSequencer
from .blending import SeamBlender
from .cropping import CanvasCropper
from .pipeline import PanoramaEngine, PanoramaResult, PairwiseReport

__version__ = "1.0.0"
__all__ = [
    "FeatureDetector",
    "FeatureMatcher",
    "TransformationEstimator",
    "SurfaceProjector",
    "ImageSequencer",
    "SeamBlender",
    "CanvasCropper",
    "PanoramaEngine",
    "PanoramaResult",
    "PairwiseReport",
]
