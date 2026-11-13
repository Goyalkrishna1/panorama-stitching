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
    page_title="PanoraVision Pro | Computational Panorama Lab",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Hide Default Sidebar */
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #00C6FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 0.8rem;
    }
    .telemetry-chip-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1.2rem;
    }
    .telemetry-chip {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        letter-spacing: 0.3px;
    }
    /* Top Control Ribbon: Pure Black Background & Crisp White Separator */
    .topbar-wrapper {
        background-color: #000000;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-bottom: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
    }
    .topbar-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }
    .topbar-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .metric-badge {
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.08) 0%, rgba(15, 23, 42, 0.45) 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* File Uploader Drop Box: Centered, Spacious & High-Tech */
    [data-testid="stFileUploader"] {
        text-align: center !important;
    }
    [data-testid="stFileUploader"] > label {
        text-align: center !important;
        display: block !important;
        width: 100% !important;
        font-size: 1.0rem !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] > div,
    section[data-testid="stFileUploadDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 200px !important;
        padding: 40px 24px !important;
        border: 2px dashed rgba(56, 189, 248, 0.5) !important;
        border-radius: 14px !important;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.45) 100%) !important;
        gap: 12px !important;
        transition: all 0.25s ease-in-out !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #38BDF8 !important;
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.14) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.25) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] div {
        text-align: center !important;
        width: 100% !important;
    }
    [data-testid="stFileUploaderDropzone"] span {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stFileUploaderDropzone"] small {
        font-size: 0.9rem !important;
        color: #94A3B8 !important;
        margin-top: 2px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        margin: 8px auto 0 auto !important;
        padding: 8px 24px !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
    }

    /* Selectbox Dropdown height enhancement */
    div[data-baseweb="select"] > div {
        min-height: 42px !important;
        border-radius: 8px !important;
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
# Header Section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">PanoraVision Pro • Advanced Stitching Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Computational Multi-Perspective Stitching • Scale-Invariant Keypoints • RANSAC Consensus • Multi-Band Laplacian Synthesis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="telemetry-chip-bar">
        <span class="telemetry-chip">🔬 SIFT Invariant Detector</span>
        <span class="telemetry-chip">📐 8-DOF Projective Homography</span>
        <span class="telemetry-chip">🎯 RANSAC Outlier Rejection</span>
        <span class="telemetry-chip">🌀 Cylindrical Panning Warper</span>
        <span class="telemetry-chip">🎨 Laplacian Multi-Band Blender</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Top Control Ribbon (Engine Configuration Bar)
# ─────────────────────────────────────────────────────────────────────────────

# Optimal System Defaults
max_features = 4500
ratio_threshold = 0.75
engine_choice = "custom"
camera_fov = 60.0
ransac_thresh = 4.0
gain_comp = True

st.markdown('<div class="topbar-wrapper">', unsafe_allow_html=True)
tb1, tb2, tb3 = st.columns([1.1, 1.45, 1.45])

with tb1:
    st.markdown('<div class="topbar-label">🔍 Feature Extraction</div>', unsafe_allow_html=True)
    detector_algo = st.selectbox(
        "Detector & Descriptor",
        options=["SIFT", "ORB"],
        index=0,
        help="SIFT: Scale/rotation invariant (Unit IV core). ORB: Fast binary descriptor.",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="topbar-badge">⚡ <b>4,500 pts</b> • Lowe <b>0.75</b></div>',
        unsafe_allow_html=True,
    )

with tb2:
    st.markdown('<div class="topbar-label">📐 Surface & Geometry</div>', unsafe_allow_html=True)
    p_col, t_col = st.columns(2)
    with p_col:
        projection_mode = st.selectbox(
            "Surface Projection",
            options=["cylindrical", "spherical", "planar"],
            index=0,
            format_func=lambda x: {"cylindrical": "Cylindrical", "spherical": "Spherical", "planar": "Planar"}[x],
            help="Cylindrical maps horizontal panning to pure translation.",
            label_visibility="collapsed",
        )
    with t_col:
        transform_type = st.selectbox(
            "Transformation Model",
            options=["affine", "homography", "euclidean"],
            index=0,
            format_func=lambda x: {"affine": "Affine (4-DOF)", "homography": "Homography (8-DOF)", "euclidean": "Euclidean (3-DOF)"}[x],
            help="Geometric model for inter-image alignment.",
            label_visibility="collapsed",
        )

with tb3:
    st.markdown('<div class="topbar-label">🎨 Blending & Crop</div>', unsafe_allow_html=True)
    b_col, c_col = st.columns(2)
    with b_col:
        blend_mode = st.selectbox(
            "Blending Method",
            options=["multiband", "feather", "simple"],
            index=0,
            format_func=lambda x: {"multiband": "Multi-Band", "feather": "Feathering", "simple": "Simple"}[x],
            help="Multi-Band uses Laplacian pyramids to eliminate seams.",
            label_visibility="collapsed",
        )
    with c_col:
        crop_mode = st.selectbox(
            "Canvas Border Cropping",
            options=["rect", "tight", "none"],
            index=0,
            format_func=lambda x: {"rect": "Rectangle", "tight": "Tight", "none": "Raw"}[x],
            help="Removes irregular empty borders.",
            label_visibility="collapsed",
        )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State & Preset Loaders
# ─────────────────────────────────────────────────────────────────────────────

if "image_list" not in st.session_state:
    st.session_state.image_list = []
if "image_names" not in st.session_state:
    st.session_state.image_names = []

st.markdown("##### ⚡ 1-Click Tested Datasets (Guaranteed 100% Accurate Stitching)")
col_p1, col_p2, col_p3, col_p4 = st.columns([1.2, 1.2, 1.2, 1.0])

with col_p1:
    if st.button("🕌 Taj Mahal (4 Photos)", use_container_width=True):
        files = sorted(glob.glob("demo_images/taj_mahal/*.jpg"))
        st.session_state.image_list = [cv2.imread(f) for f in files]
        st.session_state.image_names = [os.path.basename(f) for f in files]
        st.success(f"Loaded {len(files)} Taj Mahal photos!")

with col_p2:
    if st.button("🏔️ Mountain (6 Photos)", use_container_width=True):
        files = sorted(glob.glob("demo_images/mountain_nature/*.jpg"))
        st.session_state.image_list = [cv2.imread(f) for f in files]
        st.session_state.image_names = [os.path.basename(f) for f in files]
        st.success(f"Loaded {len(files)} nature photos!")

with col_p3:
    if st.button("🗺️ Budapest (6 Photos)", use_container_width=True):
        files = sorted(glob.glob("demo_images/budapest/*.jpg"))
        st.session_state.image_list = [cv2.imread(f) for f in files]
        st.session_state.image_names = [os.path.basename(f) for f in files]
        st.success(f"Loaded {len(files)} Budapest photos!")

with col_p4:
    if st.session_state.image_list:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.image_list = []
            st.session_state.image_names = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# File Uploader
# ─────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Or upload your own phone photos (JPG, PNG) with 30-50% overlap:",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if uploaded:
    st.session_state.image_list = []
    st.session_state.image_names = []
    for file in uploaded:
        try:
            pil_img = Image.open(file)
            # Auto-downscale very large phone photos to 1400px for optimal memory and precision
            w, h = pil_img.size
            if max(w, h) > 1400:
                scale = 1400.0 / float(max(w, h))
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
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
