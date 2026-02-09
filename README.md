# TUBASPIS: Turning Bands Spectral Importance Sampling for Simulation of Non-Stationary Gaussian Random Fields


## Overview
[cite_start]This project implements the spectral algorithm for simulating Gaussian random fields with non-stationary covariance functions, as described by **Emery and Arroyo (2017)**[cite: 4].

The method generates realizations by summing cosine waves with random frequencies and phases. [cite_start]The key innovation is using **Importance Sampling**: frequencies are drawn from a proposal density $g(u)$, and weights are adjusted based on the local spectral density $f_x(u)$ at each target location $x$[cite: 9, 61].

## Key Equations

### 1. Univariate Simulation
The field $Y(x)$ is simulated as:
$$Y(x) = \frac{1}{\sqrt{L}} \sum_{l=1}^{L} \sqrt{\frac{2 f_x(u_l)}{g(u_l)}} \cos(\langle u_l, x \rangle + \phi_l)$$
Where:
- [cite_start]$L$: Number of lines (large integer, e.g., 5000)[cite: 53].
- [cite_start]$u_l$: Random frequency vectors drawn from proposal density $g$[cite: 98].
- [cite_start]$\phi_l$: Random phases uniformly distributed in $[0, 2\pi]$[cite: 98].
- [cite_start]$f_x(u)$: The spectral density of the target covariance at location $x$[cite: 66].

### 2. Multivariate (Vector) Simulation
For a vector field with $P$ components:
$$Y(x) = \frac{1}{\sqrt{L}} \sum_{l=1}^{L} \sum_{p=1}^{P} \alpha_{x,p}(u_{l,p}) \cos(\langle u_{l,p}, x \rangle + \phi_{l,p})$$
[cite_start]Where $\alpha_{x,p}$ are vector coefficients derived from the matrix of spectral densities[cite: 515].

## Implementation Plan (TODO List)

### Phase 1: Core Engine Setup
- [x] [cite_start]**Define Grid:** Create a 2D coordinate grid (e.g., $201 \times 201$ nodes)[cite: 123]. (Implemented in example usage)
- [x] **Proposal Sampler ($g$):** Implement `sample_proposal_frequencies`.
    - [cite_start]The paper recommends using the spectral density of an **isotropic Matérn covariance** for $g$ to ensure stability[cite: 124].
    - *Tip:* Sample directions uniformly on the unit sphere and radii from the Matérn radial density.
- [x] **Phase Sampler:** Implement uniform sampling for $\phi \sim U[0, 2\pi]$.

### Phase 2: Univariate Gaussian Fields
- [x] **Local Anisotropy Map ($\Sigma_x$):** Implement a function that returns a $2 \times 2$ covariance matrix for any grid location $x$.
    - [cite_start]*Task:* Replicate the example where the range varies linearly from 5 ($y=0$) to 30 ($y=200$)[cite: 123].
- [x] [cite_start]**Gaussian Spectral Density ($f_x$):** Implement the spectral density formula for anisotropic Gaussian covariance[cite: 105, 119].
    - Formula: $f(u) = |\Sigma|^{1/2} (2\sqrt{\pi})^{-d} \exp(-0.25 u^T \Sigma u)$.
- [x] **Main Loop:** Implement the summation loop (Eq. 9).
    - *Optimization:* Vectorize over grid points $x$, but loop over lines $L$ to manage memory.

### Phase 3: Mixture Models (Matérn & Exponential)
[cite_start]The paper treats Exponential and Matérn models as **scale mixtures of Gaussians**[cite: 143, 325].
- [x] **Latent Variable Sampling:** Inside the loop, sample a latent scalar $a_l$ for each line. (Implemented for Matérn)
    - [cite_start]For **Exponential Covariance:** $a_l \sim \text{Gamma}(0.5, 1)$[cite: 146].
    - [cite_start]For **Matérn Covariance:** $a_l \sim \text{Exponential}(1)$ (standard Gamma)[cite: 334].
- [x] [cite_start]**Parameter Scaling:** Modify the local anisotropy $\Sigma_x$ to $4 a_l \Sigma_x$ before evaluating the Gaussian density[cite: 147]. (Implemented for Matérn)
- [ ] [cite_start]**Shape Parameter:** For Matérn, multiply the weight by $a_l^{\mu(x)-1} / \Gamma(\mu(x))$[cite: 335]. (Implemented for constant $\mu(x)$, **TODO: Varying $\mu(x)$**)

### Phase 4: Multivariate Extension
- [ ] **Matrix Coefficients:** Implement the matrix $A_x(u)$ construction from **Equation (40)**.
    - [cite_start]This supports spatially varying shapes, scales, and anisotropy directions for multiple components[cite: 630].
    (Currently deferred due to missing information on Equation 40)

## References
All citations refer to:
**Emery, X. and Arroyo, D.** (2017). *Spectral simulation of vector random fields with non-stationary covariance*. Stochastic Environmental Research and Risk Assessment.