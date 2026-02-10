# TUBASPIS: Turning Bands Spectral Importance Sampling for Simulation of Non-Stationary Gaussian Random Fields


## Overview
This project implements the spectral algorithm for simulating Gaussian random fields with non-stationary covariance functions, as described by **Emery and Arroyo (2017)**.

The method generates realizations by summing cosine waves with random frequencies and phases. The key innovation is using **Importance Sampling**: frequencies are drawn from a proposal density $g(u)$, and weights are adjusted based on the local spectral density $f_x(u)$ at each target location $x$.

## Usage

There are two implementations: a Python reference and a high-performance C++/OpenMP version.

### Python

1. Ensure Python 3.11+ is installed.
2. Install dependencies:
   ```bash
   pip install matplotlib numpy scipy
   ```
3. Run:
   ```bash
   python main.py config.json
   ```

### C++ (recommended for large simulations)

The C++ implementation parallelizes the simulation kernel with OpenMP, giving ~10x single-thread speedup over Python and linear scaling across cores.

**Requirements:** CMake 3.16+, a C++17 compiler with OpenMP (GCC, MSVC, or Clang with libomp).

**Build:**
```bash
cd cpp
mkdir build && cd build
cmake .. -G "MinGW Makefiles"      # or -G "Unix Makefiles", "Ninja", etc.
cmake --build .
```

**Run:**
```bash
./tubaspis <config.json> [output.npy]
```

The output is a `.npy` file that can be loaded directly in Python with `np.load()`. If no output path is given, it writes to `output.npy`.

**Visualize:**
```bash
python cpp/scripts/plot.py output.npy config.json
```

**Additional config options** (C++ only):
- `seed`: RNG seed for reproducibility (default 42).
- `proposal_scale`: Scale parameter for the multivariate-t proposal (default 3.0).
- `proposal_nu`: Shape parameter for the proposal density (default 0.3).

### Configuration Format
Create a JSON file (e.g., `config.json`) with the following structure:

**2D Simulation:**
```json
{
  "grid": {
    "x_min": 0, "x_max": 200, "x_n": 201,
    "y_min": 0, "y_max": 200, "y_n": 201
  },
  "L": 5000,
  "type": "gaussian"
}
```

**3D Simulation:**
Add `z` parameters to the grid configuration:
```json
{
  "grid": {
    "x_min": 0, "x_max": 200, "x_n": 100,
    "y_min": 0, "y_max": 200, "y_n": 100,
    "z_min": 0, "z_max": 200, "z_n": 50
  },
  "L": 5000,
  "type": "matern",
  "matern_nu": 0.5
}
```

**Parameters:**
- `grid`: Defines the spatial domain and resolution.
- `L`: Number of cosine lines to sum (higher = better convergence but slower).
- `batch_size`: Number of frequency lines to process in parallel (default 1000). Reduce this (e.g., 100) for large 3D grids to avoid memory issues.
- `type`: Field type, either `"gaussian"` or `"matern"`.
- `matern_nu`: Smoothness parameter for Matern covariance (default 0.5).
- `anisotropy`: (Optional) `"linear_y"` (default) or `"lva_azimuth_ramp"`.
- `ranges`: (Optional) List of correlation ranges $[r_1, r_2, r_3]$ for LVA.
- `azimuth_start` / `azimuth_end`: (Optional) Start and end angles (in degrees) for the azimuth ramp LVA.

### Examples
The project includes several example configurations. Both Python and C++ accept the same config format:

1.  **`config.json`**: Basic 2D Gaussian simulation with linearly varying range along the Y-axis.
    ```bash
    python main.py config.json
    # or
    cpp/build/tubaspis config.json output.npy
    ```

2.  **`config_3d.json`**: 3D simulation illustrating how to setup a volumetric grid.
    ```bash
    python main.py config_3d.json
    # or
    cpp/build/tubaspis config_3d.json output_3d.npy
    ```

3.  **`config_lva.json`**: 3D simulation with **Locally Varying Anisotropy (LVA)**.
    - Demonstrates rotating anisotropy where the azimuth angle varies linearly from `azimuth_start` to `azimuth_end` along the X-axis.
    - Uses defined `ranges` for primary, secondary, and tertiary directions.
    ```bash
    python main.py config_lva.json
    # or
    cpp/build/tubaspis config_lva.json output_lva.npy
    ```

## Key Equations

### 1. Univariate Simulation
The field $Y(x)$ is simulated as:
$$Y(x) = \frac{1}{\sqrt{L}} \sum_{l=1}^{L} \sqrt{\frac{2 f_x(u_l)}{g(u_l)}} \cos(\langle u_l, x \rangle + \phi_l)$$
Where:
- $L$: Number of lines (large integer, e.g., 5000).
- $u_l$: Random frequency vectors drawn from proposal density $g$.
- $\phi_l$: Random phases uniformly distributed in $[0, 2\pi]$.
- $f_x(u)$: The spectral density of the target covariance at location $x$.

### 2. Multivariate (Vector) Simulation
For a vector field with $P$ components:
$$Y(x) = \frac{1}{\sqrt{L}} \sum_{l=1}^{L} \sum_{p=1}^{P} \alpha_{x,p}(u_{l,p}) \cos(\langle u_{l,p}, x \rangle + \phi_{l,p})$$
Where $\alpha_{x,p}$ are vector coefficients derived from the matrix of spectral densities.

## Implementation Plan (TODO List)

### Phase 1: Core Engine Setup
- [x] **Define Grid:** Create a 2D coordinate grid (e.g., $201 \times 201$ nodes). (Implemented in example usage)
- [x] **Proposal Sampler ($g$):** Implement `sample_proposal_frequencies`.
    - The paper recommends using the spectral density of an **isotropic Matérn covariance** for $g$ to ensure stability.
    - *Tip:* Sample directions uniformly on the unit sphere and radii from the Matérn radial density.
- [x] **Phase Sampler:** Implement uniform sampling for $\phi \sim U[0, 2\pi]$.

### Phase 2: Univariate Gaussian Fields
- [x] **Local Anisotropy Map ($\Sigma_x$):** Implement a function that returns a $2 \times 2$ covariance matrix for any grid location $x$.
    - *Task:* Replicate the example where the range varies linearly from 5 ($y=0$) to 30 ($y=200$).
- [x] **Gaussian Spectral Density ($f_x$):** Implement the spectral density formula for anisotropic Gaussian covariance.
    - Formula: $f(u) = |\Sigma|^{1/2} (2\sqrt{\pi})^{-d} \exp(-0.25 u^T \Sigma u)$.
- [x] **Main Loop:** Implement the summation loop (Eq. 9).
    - *Optimization:* Vectorize over grid points $x$, but loop over lines $L$ to manage memory.

### Phase 3: Mixture Models (Matérn & Exponential)
The paper treats Exponential and Matérn models as **scale mixtures of Gaussians**.
- [x] **Latent Variable Sampling:** Inside the loop, sample a latent scalar $a_l$ for each line. (Implemented for Matérn)
    - For **Exponential Covariance:** $a_l \sim \text{Gamma}(0.5, 1)$.
    - For **Matérn Covariance:** $a_l \sim \text{Exponential}(1)$ (standard Gamma).
- [x] **Parameter Scaling:** Modify the local anisotropy $\Sigma_x$ to $4 a_l \Sigma_x$ before evaluating the Gaussian density. (Implemented for Matérn)
- [ ] **Shape Parameter:** For Matérn, multiply the weight by $a_l^{\mu(x)-1} / \Gamma(\mu(x))$. (Implemented for constant $\mu(x)$, **TODO: Varying $\mu(x)$**)

### Phase 4: Multivariate Extension
- [ ] **Matrix Coefficients:** Implement the matrix $A_x(u)$ construction from **Equation (40)**.
    - This supports spatially varying shapes, scales, and anisotropy directions for multiple components.
    (Currently deferred due to missing information on Equation 40)

## References
All citations refer to:
**Emery, X. and Arroyo, D.** (2017). *Spectral simulation of vector random fields with non-stationary covariance*. Stochastic Environmental Research and Risk Assessment.