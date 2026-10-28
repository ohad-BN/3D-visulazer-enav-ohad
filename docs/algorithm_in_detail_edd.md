# Algorithm in Detail (Engineering Design)

## 1. Mathematical & Geometric Formulation
* **3D Volume Domain:**
  * Matrix $M$ of dimensions $sx \times sy \times sz$, where $1 \le sx, sy, sz \le 1024$.
  * Valid voxel coordinates $(x, y, z)$ satisfy $0 \le x < sx$, $0 \le y < sy$, $0 \le z < sz$.
* **Case A: 3-Point Plane (Isosceles Triangle $a, b, c \in \mathbb{R}^3$):**
  * Constraint: $|\text{dist}(a,b) - \text{dist}(a,c)| < \epsilon$ where $\epsilon = 0.001$.
  *Unified Mapping: Set top vertices $a_1 = a$ and $a_2 = a$.
* **Case B: 4-Point Plane (Isosceles Trapezoid $a_1, a_2, b, c \in \mathbb{R}^3$):**
  * Equal legs constraint: $|\text{dist}(a_1,b) - \text{dist}(a_2,c)| < \epsilon$.
  * Base ordering constraint: $\text{dist}(a_1,a_2) \le \text{dist}(b,c)$.
  * Coplanarity constraint: $(\vec{a_1b} \times \vec{a_1c}) \cdot \vec{a_1a_2} = 0$ (within $\epsilon$).

## 2. Step-by-Step Projection Algorithm
* **Step 1 - Dimensions & Distances:**
  * Leg length:
    $$D_{ab} = \text{dist}(a_1, b) = \|b - a_1\|_2$$
  * Base width:
    $$D_{bc} = \text{dist}(b, c) = \|c - b\|_2$$
  * Allocate 2D output matrix $E$ of integer dimensions $\lceil D_{bc} \rceil \times \lceil D_{ab} \rceil$, initialized to $0$.
* **Step 2 - Normalized Side Step Vectors:**
  * Left leg unit vector:
    $$\hat{v}_{ab} = \frac{b - a_1}{D_{ab}}$$
  * Right leg unit vector:
    $$\hat{v}_{ac} = \frac{c - a_2}{D_{ab}}$$
* **Step 3 - Outer Scanline Loop (Vertical Axis $y$):**
  * Initialize marching endpoints: $p_1 = a_1$, $p_2 = a_2$.
  * Iterate $y = 0, 1, \dots, \lfloor D_{ab} \rfloor - 1$:
    * Advance endpoints along legs:
      $$p_1 \leftarrow a_1 + y \cdot \hat{v}_{ab}, \quad p_2 \leftarrow a_2 + y \cdot \hat{v}_{ac}$$
    * Compute horizontal span length:
      $$d_{12} = \|p_2 - p_1\|_2$$
    * Compute horizontal unit step vector (if $d_{12} > 0$):
      $$\hat{dp} = \frac{p_2 - p_1}{d_{12}}$$
* **Step 4 - Inner Scanline Loop (Horizontal Axis $x_1$):**
  * Initialize sampling point $p = p_1$.
  * Compute centering horizontal offset:
    $$x_{\text{offset}} = \left\lfloor \frac{D_{bc} - d_{12}}{2} \right\rfloor$$
  * Iterate $x_1 = 0, 1, \dots, \lfloor d_{12} \rfloor$:
    * Target column index: $x = x_{\text{offset}} + x_1$.
    * Nearest-neighbor voxel lookup (using rounded/integer indices $p_x, p_y, p_z$):
      $$v = \begin{cases} M[\lfloor p_x \rfloor][\lfloor p_y \rfloor][\lfloor p_z \rfloor] & \text{if } 0 \le p_x < sx, \; 0 \le p_y < sy, \; 0 \le p_z < sz \\ 0 & \text{otherwise} \end{cases}$$
    * Write pixel: $E[x][y] = v$.
    * Advance sample point: $p \leftarrow p + \hat{dp}$.
* **Step 5 - Output:** Return 2D matrix $E$ as a single video frame.

## 3. Complexity & High-FPS Optimization (> 60 FPS)
* **Time Complexity:** $\mathcal{O}(D_{ab} \cdot D_{bc})$ operations per frame ($O(1)$ lookup per pixel).
* **Space Complexity:** $\mathcal{O}(sx \cdot sy \cdot sz)$ for volume $M$, plus $\mathcal{O}(D_{ab} \cdot D_{bc})$ for 2D frame buffer $E$.
* **Optimization Strategy for 60–300+ FPS:**
  * **Vectorization / JIT:** Replace nested Python loops with NumPy vectorized meshgrids or Numba `@njit(parallel=True)` compilation.
  * **Memory Layout:** Store $M$ as a contiguous 1D/3D C-order array in RAM to maximize CPU L1/L2 cache hits during scanline traversal.
  * **Pre-allocation:** Reuse the 2D frame buffer $E$ across frames to eliminate garbage collection overhead.
  * **Division Avoidance:** Pre-calculate reciprocals ($1 / D_{ab}$ and $1 / d_{12}$) and use linear interpolation (LERP) across scanlines.