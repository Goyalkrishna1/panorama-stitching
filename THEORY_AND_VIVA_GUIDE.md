# 📚 IPCV Unit IV: Panorama Stitching & Feature Matching
## Comprehensive Academic Theory & Viva Guide

---

## 1. Local Feature Detection & Description (SIFT & ORB)

### 1.1 Scale-Invariant Feature Transform (SIFT)
Developed by David Lowe (1999, 2004), SIFT achieves invariance to scale, rotation, affine distortion, and 3D viewpoint changes.

#### Stage 1: Scale-Space Extrema Detection
- The only linear scale-space kernel is the Gaussian function $G(x, y, \sigma)$:
  $$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
- To detect stable scale-space locations efficiently, SIFT computes the **Difference-of-Gaussians (DoG)**:
  $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y) = L(x, y, k\sigma) - L(x, y, \sigma)$$
- **Extrema Selection**: Each candidate pixel is compared against its 26 neighbors (8 in the current scale, 9 in the scale above, 9 in the scale below).

#### Stage 2: Sub-Pixel Keypoint Localization & Filtering
1. **Sub-pixel Interpolation**: Uses a 2nd-order Taylor expansion of scale-space function $D(\mathbf{x})$:
   $$\mathbf{\hat{x}} = -\frac{\partial^2 D^{-1}}{\partial \mathbf{x}^2} \frac{\partial D}{\partial \mathbf{x}}$$
   If $|D(\mathbf{\hat{x}})| < 0.03$, the candidate is discarded due to low contrast.
2. **Eliminating Edge Responses**: Poorly localized edges are rejected using the Hessian matrix $H$:
   $$\mathbf{H} = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$$
   Using eigenvalue ratio $r = \alpha / \beta$:
   $$\frac{\operatorname{Tr}(\mathbf{H})^2}{\operatorname{Det}(\mathbf{H})} = \frac{(r + 1)^2}{r} < \frac{(r_{\text{threshold}} + 1)^2}{r_{\text{threshold}}} \quad (r_{\text{threshold}} = 10)$$

#### Stage 3: Orientation Assignment
- For each localized keypoint, gradient magnitude $m(x, y)$ and direction $\theta(x, y)$ are computed:
  $$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$
  $$\theta(x, y) = \operatorname{atan2}(L(x, y+1) - L(x, y-1), L(x+1, y) - L(x-1, y))$$
- An orientation histogram with 36 bins ($10^\circ$ each) is formed around the keypoint, weighted by gradient magnitude and a Gaussian window. The peak determines the canonical orientation.

#### Stage 4: 128-Dimensional Feature Descriptor
- A $16 \times 16$ neighborhood around the keypoint is divided into sixteen $4 \times 4$ sub-regions.
- In each sub-region, an 8-bin orientation histogram is accumulated.
- Total descriptor dimensionality: $4 \times 4 \times 8 = 128$ dimensions.
- The 128-element vector is normalized to unit length, clipped at $0.2$, and re-normalized to achieve illumination invariance.

---

### 1.2 Oriented FAST and Rotated BRIEF (ORB)
Designed by Ethan Rublee et al. (2011) as an efficient, patent-free alternative to SIFT and SURF.
- **Detector**: Modified FAST (Features from Accelerated Segment Test). Uses an intensity centroid to compute the orientation angle:
  $$C = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right), \quad \theta = \operatorname{atan2}(m_{01}, m_{10})$$
- **Descriptor**: Steered BRIEF (Binary Robust Independent Elementary Features). Rotates test pixel pairs by $\theta$:
  $$g_n(p, \theta) = f_n(R_\theta p)$$
  Produces a 256-bit (32-byte) binary descriptor compared via **Hamming Distance** (XOR + population count).

---

## 2. Feature Matching & Lowe's Ratio Test

### 2.1 Lowe's Ratio Distance Metric
In real scenes with repetitive patterns (e.g., brick textures, windows, foliage), false matches are common.
- For each query keypoint, find its closest neighbor ($d_1$) and second closest neighbor ($d_2$) in descriptor space.
- A match is accepted only if:
  $$\frac{d_1}{d_2} < \tau \quad (\text{typically } \tau = 0.70 \text{ to } 0.75)$$
- **Why it works**: For correct matches, $d_1$ is substantially smaller than $d_2$. For ambiguous/incorrect matches, $d_1 \approx d_2$, yielding a ratio near $1.0$.

---

## 3. Geometric Transformations & RANSAC

### 3.1 2D Transformation Hierarchy
| Model | DOF | Minimum Points | Invariants | Matrix Form ($3\times3$) |
|---|---|---|---|---|
| **Euclidean (Rigid)** | 3 | 2 pairs | Lengths, angles, areas | Rotation + Translation |
| **Similarity (Affine Partial)** | 4 | 2 pairs | Angles, parallel lines, uniform scale | Scale $\cdot$ Rotation + Translation |
| **Affine** | 6 | 3 pairs | Parallel lines, area ratios | Arbitrary 2D linear + Translation |
| **Projective (Homography)** | 8 | 4 pairs | Straight lines, cross-ratio | Full $3\times3$ with $h_{33} = 1$ |

### 3.2 The Projective Homography Matrix
$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim
\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$
- Homography holds exactly when:
  1. The camera undergoes pure rotation around its optical center ($\mathbf{t} = 0$).
  2. The scene being imaged is completely planar ($Z = 0$).

### 3.3 Random Sample Consensus (RANSAC)
- **Problem with DLT (Direct Linear Transform)**: Least squares estimation is extremely sensitive to outliers. A single false match can ruin the estimated matrix.
- **RANSAC Algorithm**:
  1. Randomly sample the minimum subset of point correspondences (4 for Homography, 3 for Affine).
  2. Compute candidate transformation matrix $\mathbf{H}$.
  3. Count inliers whose symmetric transfer error satisfies:
     $$d(\mathbf{x}_i', \mathbf{H}\mathbf{x}_i) < \epsilon_{\text{threshold}}$$
  4. Repeat for $N$ iterations:
     $$N = \frac{\ln(1 - p)}{\ln(1 - (1 - \epsilon)^s)}$$
     where $p$ is confidence ($0.99$), $\epsilon$ is outlier ratio, and $s$ is sample size.
  5. Re-estimate $\mathbf{H}$ using all verified inliers via SVD.

---

## 4. Cylindrical Projection & Seam Blending

### 4.1 Why Cylindrical Projection?
Projecting wide-angle panning photos directly onto a flat canvas causes points at the edges to diverge ($x \to \infty$). Mapping onto a cylinder converts rotational panning into horizontal translation:
$$\theta = \frac{x_{\text{cyl}} - c_x}{f}, \quad h = \frac{y_{\text{cyl}} - c_y}{f}$$
Backward mapping formula:
$$x_{\text{src}} = f \tan\theta + c_x, \quad y_{\text{src}} = \frac{y_{\text{cyl}} - c_y}{\cos\theta} + c_y$$

### 4.2 Burt-Adelson Multi-Band Laplacian Pyramid
- **Problem**: Simple alpha feathering blurs high-frequency details (ghosting). Direct cuts leave noticeable brightness seams.
- **Laplacian Solution**: Decomposes spatial frequencies:
  - Low frequencies (illumination gradients) are blended smoothly over wide spatial transitions.
  - High frequencies (textures, edges) are blended sharply over narrow boundaries.

---

## 5. Quick Viva Q&A Reference

**Q1: Why is SIFT scale-invariant while Harris corner detector is not?**  
*A:* Harris calculates second-moment eigenvalues in a fixed window. When an image is zoomed in, a corner becomes a smooth edge within that window. SIFT searches for extrema across a continuous Gaussian scale space ($L(x, y, \sigma)$) and detects feature scale explicitly.

**Q2: What is the difference between SIFT and SURF?**  
*A:* SIFT approximates LoG using Difference-of-Gaussians (DoG) with continuous filtering. SURF uses box filters to approximate Gaussian 2nd-order derivatives, evaluated in $O(1)$ constant time using Integral Images.

**Q3: How many degrees of freedom does a 2D Homography matrix have?**  
*A:* 8 degrees of freedom. Although it is a $3 \times 3$ matrix (9 entries), it is defined up to an arbitrary scale factor ($H \sim \lambda H$), leaving 8 independent parameters.

**Q4: Why does our project support Affine Similarity as well as Homography?**  
*A:* Projective homography has non-linear denominators $(h_{31}x + h_{32}y + h_{33})$. Across wide multi-photo panoramas, small estimation errors cause the denominator to approach zero, causing the canvas to flare outward in a trapezoidal "wedge". Affine similarity constrains the bottom row to $[0, 0, 1]$, guaranteeing uniform vertical height.

**Q5: Why align outward from the center image ($H_{\text{mid}} = I$)?**  
*A:* Sequential left-to-right chaining compounds alignment errors. By fixing the central image as the anchor, error propagation is halved and distributed symmetrically to the left and right.
