"""
Panorama Reconstruction Command-Line Interface (CLI)
====================================================
Headless execution tool for multi-image panorama stitching.

Usage examples:
  python cli.py --input demo_images/scenic_city --output output/panorama.png
  python cli.py --input demo_images/architecture --detector ORB --blend feather
  python cli.py --input path/to/photos --projection cylindrical --fov 65 --crop rect
"""

import argparse
import os
import glob
import sys
import time
import cv2

from panorama.pipeline import PanoramaEngine, PanoramaResult


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="High-Accuracy Multi-Image Panorama Reconstruction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to directory containing input images (jpg, png) or a file pattern.",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/stitched_panorama.png",
        help="Path where the final panorama will be saved.",
    )
    parser.add_argument(
        "--detector",
        choices=["SIFT", "ORB", "AKAZE", "BRISK"],
        default="SIFT",
        help="Feature detection and description algorithm.",
    )
    parser.add_argument(
        "--transform",
        choices=["affine", "homography", "euclidean"],
        default="affine",
        help="Geometric transformation model (affine prevents wedge flare).",
    )
    parser.add_argument(
        "--projection",
        choices=["cylindrical", "spherical", "planar"],
        default="cylindrical",
        help="Surface projection before alignment.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Estimated horizontal camera Field of View in degrees.",
    )
    parser.add_argument(
        "--blend",
        choices=["multiband", "feather", "simple"],
        default="multiband",
        help="Seam blending algorithm.",
    )
    parser.add_argument(
        "--crop",
        choices=["rect", "tight", "none"],
        default="rect",
        help="Canvas boundary post-processing.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum keypoints detected per image.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=4.0,
        help="RANSAC reprojection error threshold in pixels.",
    )
    parser.add_argument(
        "--engine",
        choices=["custom", "opencv"],
        default="custom",
        help="Stitching backend engine.",
    )
    return parser.parse_args()


def load_input_images(input_path: str):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    if os.path.isdir(input_path):
        files = []
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        files = sorted(list(set(files)))
    else:
        files = sorted(glob.glob(input_path))

    if not files:
        print(f"Error: No image files found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    images = []
    names = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
            names.append(os.path.basename(f))
        else:
            print(f"Warning: Could not decode {f}", file=sys.stderr)

    return images, names


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_arguments()

    print("=" * 65)
    print(" [*] PANORAMA RECONSTRUCTION & STITCHING SYSTEM (CLI)")
    print("=" * 65)

    images, names = load_input_images(args.input)
    print(f"Loaded {len(images)} images from: {args.input}")
    for idx, (img, name) in enumerate(zip(images, names)):
        h, w = img.shape[:2]
        print(f"  [{idx+1}] {name} ({w}x{h} px)")

    if len(images) < 2:
        print("Error: Minimum 2 overlapping images required.", file=sys.stderr)
        sys.exit(1)

    print("-" * 65)
    print("Configuration:")
    print(f"  - Detector:          {args.detector}")
    print(f"  - Transform:         {args.transform.title()}")
    print(f"  - Projection:        {args.projection.title()} (FOV: {args.fov} deg)")
    print(f"  - Seam Blending:     {args.blend.title()}")
    print(f"  - Crop Mode:         {args.crop.title()}")
    print(f"  - Backend Engine:    {args.engine.title()}")
    print("-" * 65)

    engine = PanoramaEngine(
        detector=args.detector,
        max_features=args.max_features,
        ratio_threshold=args.ratio,
        ransac_threshold=args.ransac_thresh,
        transform_model=args.transform,
        projection=args.projection,
        fov_degrees=args.fov,
        blend_mode=args.blend,
        crop_mode=args.crop,
        engine=args.engine,
    )

    def cli_progress(msg: str, frac: float):
        pct = int(frac * 100)
        sys.stdout.write(f"\r[{pct:3d}%] {msg:<50}")
        sys.stdout.flush()

    res: PanoramaResult = engine.stitch(images, progress_cb=cli_progress)
    print()  # newline after progress

    if res.panorama is None:
        print("\n[!] Failed to reconstruct panorama. Inspect match overlaps or adjust threshold.")
        sys.exit(1)

    # Save output
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(args.output, res.panorama)
    if not success:
        print(f"Error saving image to {args.output}", file=sys.stderr)
        sys.exit(1)

    h_out, w_out = res.panorama.shape[:2]
    aspect = w_out / float(h_out)
    print("=" * 65)
    print("[+] STITCHING COMPLETE")
    print(f"  - Output Path:       {args.output}")
    print(f"  - Resolution:        {w_out} x {h_out} px ({aspect:.2f}:1 aspect ratio)")
    print(f"  - Stitched Images:   {res.diagnostics.get('stitched_count', len(images))} of {len(images)}")
    print(f"  - Execution Time:    {res.diagnostics.get('execution_time_sec', 0.0):.2f} seconds")
    print(f"  - Optimal Sequence:  {[idx+1 for idx in res.stitching_order]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
