# Author: Goyalkrishna1 | Streamlit Studio UI
"""
Interactive Web Application for Panorama Reconstruction & Image Stitching
=========================================================================
Run with:
    streamlit run app.py
"""

import os
import glob
import time
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from panorama.pipeline import PanoramaEngine, PanoramaResult
from panorama.detector import FeatureDetector

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config & Theme
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Panorama Studio | Computer Vision Stitcher",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7E57C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: rgba(30, 136, 229, 0.08);
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def resize_max_dim(img: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def encode_image_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Image encoding error.")
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.caption("Computer Vision & Algorithm Parameters")
    st.markdown("---")

    # 1. Feature Detection
    st.markdown("### 🔍 Feature Detection")
    detector_algo = st.selectbox(
        "Detector & Descriptor",
        options=["SIFT", "ORB", "AKAZE", "BRISK"],
        index=0,
        help="SIFT: Scale/rotation invariant (Unit IV core). ORB/AKAZE/BRISK: Fast binary descriptors.",
    )
    max_features = st.slider(
        "Max Keypoints per Image",
        min_value=500,
        max_value=8000,
        value=4000,
        step=500,
    )
    ratio_threshold = st.slider(
        "Lowe's Ratio Test (d1 / d2)",
        min_value=0.50,
        max_value=0.90,
        value=0.75,
        step=0.01,
        help="Rejects ambiguous matches. Lower is stricter (0.70-0.75 is standard).",
    )

    st.markdown("---")

    # 2. Geometry & Surface Projection
    st.markdown("### 📐 Geometry & Projections")
    projection_mode = st.selectbox(
        "Surface Projection",
        options=["cylindrical", "spherical", "planar"],
        index=0,
        format_func=lambda x: {
            "cylindrical": "Cylindrical (Eliminates edge curvature)",
            "spherical": "Spherical (Equirectangular)",
            "planar": "Planar (Standard perspective)",
        }[x],
        help="Cylindrical maps horizontal panning to pure translation, eliminating trapezoidal flare.",
    )

    if projection_mode in ["cylindrical", "spherical"]:
        camera_fov = st.slider(
            "Horizontal Field of View (FOV°)",
            min_value=35,
            max_value=120,
            value=60,
            step=5,
            help="Normal kit lens ≈ 55–65°, phone wide camera ≈ 70–80°.",
        )
    else:
        camera_fov = 60.0

    transform_type = st.selectbox(
        "Transformation Model",
        options=["affine", "homography", "euclidean"],
        index=0,
        format_func=lambda x: {
            "affine": "Affine Similarity (4-DOF: Uniform scale & height)",
            "homography": "Projective Homography (8-DOF: Full 3x3)",
            "euclidean": "Rigid Euclidean (3-DOF: Rotation + Translation)",
        }[x],
        help="Affine maintains parallel vertical walls and uniform scale across wide panoramas.",
    )

    ransac_thresh = st.slider(
        "RANSAC Inlier Threshold (px)",
        min_value=1.0,
        max_value=8.0,
        value=4.0,
        step=0.5,
    )

    st.markdown("---")

    # 3. Blending & Seams
    st.markdown("### 🎨 Seam Blending & Cropping")
    blend_mode = st.selectbox(
        "Blending Method",
        options=["multiband", "feather", "simple"],
        index=0,
        format_func=lambda x: {
            "multiband": "Multi-Band Laplacian Pyramid (Burt-Adelson)",
            "feather": "Distance-Transform Alpha Feathering",
            "simple": "Simple Overlay (Hard Seam)",
        }[x],
    )
    gain_comp = st.checkbox("Photometric Exposure Equalization", value=True)

    crop_mode = st.selectbox(
        "Canvas Border Cropping",
        options=["rect", "tight", "none"],
        index=0,
        format_func=lambda x: {
            "rect": "Largest Interior Rectangle (Clean edges)",
            "tight": "Tight Bounding Box",
            "none": "Raw Canvas (Uncropped)",
        }[x],
    )

    st.markdown("---")

    # Engine Selection
    engine_choice = st.radio(
        "Backend Engine",
        options=["custom", "opencv"],
        format_func=lambda x: "Custom Algorithmic Pipeline" if x == "custom" else "OpenCV cv2.Stitcher Baseline",
        index=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header Section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🌌 Panorama Reconstruction Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-image feature matching, RANSAC homography estimation, cylindrical projection, and multi-band Laplacian blending.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State & Preset Loaders
# ─────────────────────────────────────────────────────────────────────────────

if "image_list" not in st.session_state:
    st.session_state.image_list = []
if "image_names" not in st.session_state:
    st.session_state.image_names = []

col_p1, col_p2, col_p3 = st.columns([1.2, 1.2, 2.0])

with col_p1:
    if st.button("🏰 Load Scenic City (4 Photos)", use_container_width=True):
        files = sorted(glob.glob("demo_images/scenic_city/*.jpg"))
        st.session_state.image_list = [cv2.imread(f) for f in files]
        st.session_state.image_names = [os.path.basename(f) for f in files]
        st.success(f"Loaded {len(files)} city images!")

with col_p2:
    if st.button("🏙️ Load Architecture (4 Photos)", use_container_width=True):
        files = sorted(glob.glob("demo_images/architecture/*.jpg"))
        st.session_state.image_list = [cv2.imread(f) for f in files]
        st.session_state.image_names = [os.path.basename(f) for f in files]
        st.success(f"Loaded {len(files)} architecture images!")

with col_p3:
    if st.session_state.image_list:
        if st.button("🗑️ Clear Canvas", use_container_width=False):
            st.session_state.image_list = []
            st.session_state.image_names = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# File Uploader
# ─────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload 2 to 6 overlapping photos (JPG, PNG) captured from a panning viewpoint:",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if uploaded:
    st.session_state.image_list = []
    st.session_state.image_names = []
    for file in uploaded:
        try:
            pil_img = Image.open(file)
            bgr = pil_to_bgr(pil_img)
            st.session_state.image_list.append(bgr)
            st.session_state.image_names.append(file.name)
        except Exception as err:
            st.warning(f"Could not parse {file.name}: {err}")

loaded_images = st.session_state.image_list
loaded_names = st.session_state.image_names

if len(loaded_images) < 2:
    st.info("👋 Select one of the demo datasets above or upload at least 2 overlapping photos to begin.")
    st.stop()

# Thumbnail Preview Strip
with st.expander(f"📷 Active Input Sequence ({len(loaded_images)} images ready)", expanded=True):
    cols = st.columns(min(len(loaded_images), 6))
    for idx, (img, name) in enumerate(zip(loaded_images, loaded_names)):
        with cols[idx % len(cols)]:
            h, w = img.shape[:2]
            st.image(to_pil(img), caption=f"{name}\n({w}×{h})", use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Execution Trigger
# ─────────────────────────────────────────────────────────────────────────────

run_col1, run_col2 = st.columns([2, 1])
with run_col1:
    stitch_button = st.button("🚀 Reconstruct Panorama", type="primary", use_container_width=True)

if stitch_button:
    engine = PanoramaEngine(
        detector=detector_algo,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_thresh,
        transform_model=transform_type,
        projection=projection_mode,
        fov_degrees=camera_fov,
        blend_mode=blend_mode,
        crop_mode=crop_mode,
        enable_gain_comp=gain_comp,
        engine=engine_choice,
    )

    prog_bar = st.progress(0.0)
    status_label = st.empty()

    def update_progress(msg: str, frac: float):
        prog_bar.progress(min(frac, 1.0))
        status_label.markdown(f"**Step:** {msg}")

    t0 = time.time()
    result: PanoramaResult = engine.stitch(loaded_images, progress_cb=update_progress)
    elapsed = time.time() - t0

    prog_bar.progress(1.0)
    status_label.markdown(f"✅ **Stitching Complete in {elapsed:.2f}s!**")

    # ─────────────────────────────────────────────────────────────────────────
    # Result Presentation Tabs
    # ─────────────────────────────────────────────────────────────────────────

    tab_studio, tab_keypoints, tab_matches, tab_steps, tab_theory = st.tabs(
        [
            "🌟 Panorama Studio",
            "🔬 Keypoint Extractor",
            "🎯 Pairwise Matching",
            "🎞️ Construction Timeline",
            "📐 Mathematical Telemetry",
        ]
    )

    # ── TAB 1: Panorama Studio ───────────────────────────────────────────────
    with tab_studio:
        if result.panorama is not None:
            h_res, w_res = result.panorama.shape[:2]
            aspect = w_res / float(h_res)
            mp = (w_res * h_res) / 1e6

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-badge"><div class="metric-val">{w_res} × {h_res}</div><div class="metric-lbl">Resolution (px)</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-badge"><div class="metric-val">{aspect:.2f} : 1</div><div class="metric-lbl">Aspect Ratio</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-badge"><div class="metric-val">{mp:.2f} MP</div><div class="metric-lbl">Megapixels</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-badge"><div class="metric-val">{elapsed:.2f} s</div><div class="metric-lbl">Total Time</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # View Toggle: Cropped vs Raw
            view_opt = st.radio(
                "Display Mode:",
                options=["Final Clean Panorama (Cropped)", "Raw Global Canvas (Before Crop)"],
                horizontal=True,
            )

            display_img = result.panorama if "Clean" in view_opt else result.raw_untrimmed
            st.image(to_pil(resize_max_dim(display_img, 1600)), use_container_width=True)

            # Download Buttons
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    "💾 Download High-Res PNG",
                    data=encode_image_bytes(result.panorama, ".png"),
                    file_name="panorama_stitched.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with d_col2:
                st.download_button(
                    "💾 Download Compressed JPEG",
                    data=encode_image_bytes(result.panorama, ".jpg"),
                    file_name="panorama_stitched.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
        else:
            st.error("Stitching was unsuccessful. Please verify image overlap or try SIFT with 0.8 ratio.")

    # ── TAB 2: Keypoint Extractor ────────────────────────────────────────────
    with tab_keypoints:
        st.subheader("Keypoint Scale & Orientation Visualization")
        st.caption(
            "Visualizing detected local feature extrema with scale octave radii and dominant gradient orientations."
        )

        selected_img_idx = st.selectbox(
            "Select image to inspect:",
            range(len(loaded_images)),
            format_func=lambda i: f"Image {i+1}: {loaded_names[i]}",
        )

        target_img = loaded_images[selected_img_idx]
        det = FeatureDetector(algorithm=detector_algo, max_features=max_features)
        kps, descs = det.detect_and_compute(target_img)
        stats = det.get_statistics(kps)

        col_kp1, col_kp2, col_kp3 = st.columns(3)
        with col_kp1:
            st.metric("Total Keypoints Detected", stats["count"])
        with col_kp2:
            st.metric("Average Extrema Scale", f"{stats['avg_size']:.2f} px")
        with col_kp3:
            st.metric("Average Response Strength", f"{stats['avg_response']:.2f}")

        annotated_img = det.draw_keypoints(target_img, kps, rich=True)
        st.image(to_pil(resize_max_dim(annotated_img, 1400)), use_container_width=True)

    # ── TAB 3: Pairwise Matching & Inlier Diagnostics ────────────────────────
    with tab_matches:
        st.subheader("Pairwise Correspondences & Inlier Verification")
        st.caption("Lines indicate correspondences passing Lowe's ratio test: 🟢 Green = Verified RANSAC Inlier, 🔴 Red = Rejected Outlier")

        valid_reports = [r for r in result.pairwise_reports if r.match_visualization is not None]
        if valid_reports:
            report_options = [
                f"{loaded_names[r.dst_idx]} ↔ {loaded_names[r.src_idx]} ({r.num_inliers}/{r.num_matches} inliers)"
                for r in valid_reports
            ]
            chosen_report_idx = st.selectbox("Select Image Pair to Inspect:", range(len(valid_reports)), format_func=lambda i: report_options[i])
            chosen_report = valid_reports[chosen_report_idx]

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Lowe's Ratio Matches", chosen_report.num_matches)
            with m_col2:
                st.metric("RANSAC Inliers", chosen_report.num_inliers)
            with m_col3:
                st.metric("Inlier Precision", f"{chosen_report.inlier_ratio * 100:.1f}%")

            st.image(to_pil(resize_max_dim(chosen_report.match_visualization, 1400)), use_container_width=True)
        else:
            st.info("No pairwise matches were recorded.")

        # Inlier Heatmap
        if result.adjacency_matrix is not None:
            st.markdown("---")
            st.subheader("📊 Inlier Connectivity Heatmap")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mat = result.adjacency_matrix
            n_imgs = len(loaded_images)

            fig, ax = plt.subplots(figsize=(max(5, n_imgs), max(4, n_imgs - 1)))
            cax = ax.imshow(mat, cmap="viridis")
            ax.set_xticks(range(n_imgs))
            ax.set_yticks(range(n_imgs))
            short_labels = [f"Img {i+1}" for i in range(n_imgs)]
            ax.set_xticklabels(short_labels, rotation=45, ha="right")
            ax.set_yticklabels(short_labels)

            for i in range(n_imgs):
                for j in range(n_imgs):
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="white" if mat[i,j] < mat.max()/2 else "black", fontsize=9)

            ax.set_title("Pairwise RANSAC Inlier Counts Across All Images")
            fig.colorbar(cax, ax=ax, label="Inlier Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── TAB 4: Construction Timeline ─────────────────────────────────────────
    with tab_steps:
        st.subheader("Step-by-Step Panoramic Canvas Synthesis")
        st.caption("Visualizing the progressive alignment and accumulation of images onto the unified coordinate system.")

        if result.intermediate_steps:
            for step_num, step_canvas in enumerate(result.intermediate_steps):
                ih, iw = step_canvas.shape[:2]
                st.markdown(f"**Stage {step_num + 1}** — Cumulative Canvas ({iw} × {ih} px):")
                st.image(to_pil(resize_max_dim(step_canvas, 1200)), use_container_width=True)
                st.markdown("---")

    # ── TAB 5: Mathematical Telemetry ────────────────────────────────────────
    with tab_theory:
        st.subheader("Mathematical Telemetry & Transformation Matrices")
        st.markdown(
            """
            In planar computer vision, points between two camera views related by rotation or a planar scene 
            are linked by a $3 \\times 3$ matrix:
            $$
            \\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} \\sim 
            \\begin{bmatrix} h_{11} & h_{12} & h_{13} \\\\ h_{21} & h_{22} & h_{23} \\\\ h_{31} & h_{32} & h_{33} \\end{bmatrix}
            \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}
            $$
            """
        )

        for rep in result.pairwise_reports:
            if rep.transform_matrix is not None:
                st.markdown(f"##### Transform Matrix: `{loaded_names[rep.dst_idx]}` ⟵ `{loaded_names[rep.src_idx]}`")
                M = rep.transform_matrix
                latex_mat = (
                    f"$$\\mathbf{{H}} = \\begin{{bmatrix}} "
                    f"{M[0,0]:.4f} & {M[0,1]:.4f} & {M[0,2]:.2f} \\\\ "
                    f"{M[1,0]:.4f} & {M[1,1]:.4f} & {M[1,2]:.2f} \\\\ "
                    f"{M[2,0]:.6f} & {M[2,1]:.6f} & {M[2,2]:.4f} "
                    f"\\end{{bmatrix}}$$"
                )
                st.markdown(latex_mat)
                st.caption(f"Inliers: {rep.num_inliers} | Inlier Ratio: {rep.inlier_ratio*100:.1f}% | Reprojection RMSE: {rep.rmse:.3f} px")
                st.markdown("---")

        st.subheader("📋 Execution Event Log")
        st.code("\n".join(result.logs), language="text")

        # Export Report
        st.download_button(
            "📄 Export Full Diagnostics JSON",
            data=json.dumps(result.diagnostics, indent=2),
            file_name="stitching_diagnostics.json",
            mime="application/json",
        )
