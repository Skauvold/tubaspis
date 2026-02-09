import argparse
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import gamma


class NonStationarySpectralSimulator:
    """
    Implements the spectral simulation algorithm for non-stationary Gaussian random fields
    as described in Emery and Arroyo (2017).
    """

    def __init__(self, grid_coords, L=5000, dim=2):
        """
        Initialize the simulator.

        Args:
            grid_coords (np.ndarray): (N, d) array of coordinates to simulate.
            [cite_start]L (int): Number of lines (cosine waves) to sum. Default 5000[cite: 124].
            dim (int): Spatial dimension (default 2).
        """
        self.grid = grid_coords
        self.L = L
        self.d = dim
        self.num_points = grid_coords.shape[0]

    def _sample_proposal_frequencies(self, scale=3.0, nu=0.3):
        """
        Sample L frequency vectors 'u' from the proposal density 'g'.

        Per the paper, 'g' is chosen as the spectral density of an isotropic
        [cite_start]Matern covariance to favor large frequencies[cite: 888].

        Args:
            scale (float): Scale factor for the proposal Matern.
            nu (float): Shape parameter for the proposal Matern.

        Returns:
            u_vectors (np.ndarray): (L, d) array of frequency vectors.
            g_values (np.ndarray): (L,) array of probability densities g(u).
        """
        if self.d != 2:
            raise NotImplementedError("Sampling proposal frequencies currently implemented for 2D only.")

        # 1. Sample directions uniformly on the unit sphere.
        angles = np.random.uniform(0, 2 * np.pi, self.L)

        # 2. Sample radii 'r' from the radial density P(r) = r * g(r) (unnormalized).
        # g(r) is proportional to (alpha_sq + r^2)^(-nu - 1)
        # where alpha_sq = (2 * nu) / (scale**2)

        alpha_sq = (2 * nu) / (scale**2)
        # B = nu + 1 (exponent in the radial density is -(nu + 1) for 2D)

        # Inverse transform sampling for x = r^2
        # P(x) = C * (alpha_sq + x)^(-B) for x > 0, with B = nu + 1
        # CDF(x) = 1 - (alpha_sq / (alpha_sq + x))^(B - 1) = 1 - (alpha_sq / (alpha_sq + x))^nu
        # x = alpha_sq * (U**(-1 / nu) - 1)

        # Generate uniform random numbers
        U = np.random.uniform(0, 1, self.L)

        # Clamp U to avoid issues with U very close to 0, which would lead to large x
        # and potentially overflow if not handled carefully, or sqrt(negative) if U is too close to 1.
        U_clamped = np.maximum(U, 1e-9) # Avoid U=0
        
        # Calculate x = r^2
        x_vals = alpha_sq * (U_clamped**(-1 / nu) - 1)
        
        # Ensure x_vals are non-negative; very small U_clamped could lead to large positive x_vals,
        # but if nu is very small and U_clamped is too close to 1, x_vals could be slightly negative
        # due to floating point precision issues.
        x_vals = np.maximum(x_vals, 0)

        # Radii are sqrt(x)
        radii = np.sqrt(x_vals)

        # 3. Combine radii and angles to get u_vectors
        u_vectors = np.zeros((self.L, self.d))
        u_vectors[:, 0] = radii * np.cos(angles)
        u_vectors[:, 1] = radii * np.sin(angles)

        # 4. Compute g(u) values for the sampled u_vectors
        # Normalized spectral density for isotropic Matern in 2D (sigma^2 = 1 for proposal)
        # S(k) = (4 * pi * Gamma(nu + 1) * (alpha_sq**nu)) / Gamma(nu) * ( alpha_sq + k^2 )**(-(nu + 1))
        
        constant_factor = (4 * np.pi * gamma(nu + 1) * (alpha_sq**nu)) / gamma(nu)
        g_values = constant_factor * (alpha_sq + radii**2)**(-(nu + 1))
        
        return u_vectors, g_values

    def _get_local_anisotropy(self, location):
        """
        Define the non-stationary anisotropy matrix Sigma_x for a given location.

        [cite_start]Example from Section 2.3.2[cite: 123]:
        Practical range varies from 5 (at y=0) to 30 (at y=200).

        Returns:
            sigma (np.ndarray): (d, d) positive semi-definite matrix.
        """
        y_coord = location[1]

        min_range = 5.0
        max_range = 30.0
        max_grid_y = 200.0 # Based on the example usage grid definition

        # Clamp y_coord to ensure it's within [0, max_grid_y] for linear interpolation
        clamped_y = np.clip(y_coord, 0, max_grid_y)
        
        # Linear interpolation of the range
        range_val = min_range + (max_range - min_range) * (clamped_y / max_grid_y)
        
        # Construct the anisotropy matrix (Sigma_x)
        # In the spectral density f(u) = exp(-0.25 * u^T * Sigma * u), a larger
        # spatial range corresponds to a larger Sigma in the exponent.
        sigma_x = (range_val**2) * np.eye(self.d)
        
        return sigma_x

    def _evaluate_gaussian_spectral_density(self, u, sigma):
        """
        Evaluates the Gaussian spectral density f_x(u) for a specific u and local sigma.

        [cite_start]Formula: Eq (12) and (14) from the paper[cite: 85, 105].
        f(u) = |Sigma|^0.5 * (2*sqrt(pi))^(-d) * exp(-0.25 * u.T * Sigma * u)
        """
        # Calculate determinant of Sigma
        det_sigma = np.linalg.det(sigma)
        
        # Calculate u.T * Sigma * u
        u_sigma_u = np.dot(np.dot(u, sigma), u)
        
        # Calculate the constant factor
        constant_factor = (det_sigma**0.5) * ((2 * np.sqrt(np.pi))**(-self.d))
        
        # Calculate the exponential term
        exp_term = np.exp(-0.25 * u_sigma_u)
        
        return constant_factor * exp_term

    def simulate_univariate_gaussian(self):
        """
        [cite_start]Simulates a univariate non-stationary Gaussian field using Eq (9)[cite: 96].

        Returns:
            field (np.ndarray): Simulated values at grid locations (N,).
        """
        print("Starting Univariate Simulation...")

        # 1. Generate Proposal Frequencies
        u_vectors, g_vals = self._sample_proposal_frequencies()

        # [cite_start]2. Generate Random Phases [cite: 98]
        phases = np.random.uniform(0, 2 * np.pi, self.L)

        # Initialize accumulator
        field = np.zeros(self.num_points)

        # 3. Main Loop over Lines (L)
        # We loop over L to save memory, as (N x L) can be very large.
        for l in range(self.L):
            u_l = u_vectors[l]
            phi_l = phases[l]
            g_u = g_vals[l]

            # Pre-compute dot product <u_l, x> for all x
            dot_prods = np.dot(self.grid, u_l)

            # TODO: Vectorize this part if possible, or iterate grid points efficiently
            # For each location x, we need the LOCAL spectral density f_x(u_l)

            f_x_vals = np.zeros(self.num_points)
            for i in range(self.num_points):
                loc = self.grid[i]
                sigma_x = self._get_local_anisotropy(loc)
                f_x_vals[i] = self._evaluate_gaussian_spectral_density(u_l, sigma_x)

            # [cite_start]Calculate weights: w = sqrt(2 * f_x / g_u) [cite: 96]
            weights = np.sqrt(2 * f_x_vals / g_u)

            # Accumulate cosine wave
            field += weights * np.cos(dot_prods + phi_l)

        # Normalize
        return field / np.sqrt(self.L)

    def simulate_univariate_matern(self, nu_local=0.5, varying_shape=False):
        """
        Simulates a non-stationary Matern field.
        [cite_start]Uses the Scale Mixture of Gaussians approach described in Section 2.3.4[cite: 324].

        Args:
            nu_local (float): The smoothness parameter (mu) for the local Matern covariance.
                              For nu_local = 0.5, this corresponds to the Exponential covariance.
            varying_shape (bool): If True, the shape parameter nu_local also varies with location.
                                   (Not implemented yet, assumed constant for now).
        Returns:
            field (np.ndarray): Simulated values at grid locations (N,).
        """
        print(f"Starting Univariate Matern Simulation with nu_local={nu_local}...")

        # 1. Generate Proposal Frequencies (same as Gaussian)
        u_vectors, g_vals = self._sample_proposal_frequencies()

        # 2. Generate Random Phases (same as Gaussian)
        phases = np.random.uniform(0, 2 * np.pi, self.L)

        # Initialize accumulator
        field = np.zeros(self.num_points)

        # 3. Main Loop over Lines (L)
        for l in range(self.L):
            u_l = u_vectors[l]
            phi_l = phases[l]
            g_u = g_vals[l]

            # Sample latent variable a_l for this line
            # [cite_start]For Matérn Covariance: a_l ~ Exponential(1) [cite: 334].
            a_l = np.random.exponential(1) # Scale=1 (mean=1)

            # Pre-compute dot product <u_l, x> for all x
            dot_prods = np.dot(self.grid, u_l)

            f_x_vals = np.zeros(self.num_points)
            # For each location x, we need the LOCAL spectral density f_x(u_l)
            for i in range(self.num_points):
                loc = self.grid[i]
                
                # Get base anisotropy matrix
                base_sigma_x = self._get_local_anisotropy(loc)
                
                # Scale local anisotropy by 4 * a_l
                # [cite_start]Sigma_eff = 4 * a_l * Sigma_x [cite: 147].
                effective_sigma_x = 4 * a_l * base_sigma_x
                
                f_x_vals[i] = self._evaluate_gaussian_spectral_density(u_l, effective_sigma_x)

            # Calculate weights: w = sqrt(2 * f_x / g_u)
            weights = np.sqrt(2 * f_x_vals / g_u)

            # Apply shape parameter correction for Matérn
            # [cite_start]Multiply weight by a_l^(mu(x)-1) / Gamma(mu(x)) [cite: 335].
            # Assuming mu(x) is constant nu_local for now.
            if nu_local > 0: # Avoid division by zero if nu_local is 0 (gamma(0) is undefined)
                weight_correction = (a_l**(nu_local - 1)) / gamma(nu_local)
                weights *= weight_correction
            else:
                raise ValueError("nu_local must be greater than 0 for Matern covariance.")

            # Accumulate cosine wave
            field += weights * np.cos(dot_prods + phi_l)

        # Normalize
        return field / np.sqrt(self.L)


def run_simulation(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Grid setup
    grid_cfg = config['grid']
    x_range = np.linspace(grid_cfg['x_min'], grid_cfg['x_max'], grid_cfg['x_n'])
    y_range = np.linspace(grid_cfg['y_min'], grid_cfg['y_max'], grid_cfg['y_n'])
    xv, yv = np.meshgrid(x_range, y_range)
    grid_coords = np.column_stack([xv.ravel(), yv.ravel()])

    # Simulator setup
    L = config.get('L', 5000)
    sim = NonStationarySpectralSimulator(grid_coords, L=L)

    # Run
    sim_type = config.get('type', 'gaussian').lower()
    if sim_type == 'gaussian':
        print("Running Gaussian Simulation...")
        result = sim.simulate_univariate_gaussian()
        title = "Simulated Univariate Gaussian Field"
    elif sim_type == 'matern':
        nu = config.get('matern_nu', 0.5)
        print(f"Running Matern Simulation (nu={nu})...")
        result = sim.simulate_univariate_matern(nu_local=nu)
        title = f"Simulated Univariate Matern Field (nu={nu})"
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")

    # Plot
    result_grid = result.reshape(grid_cfg['y_n'], grid_cfg['x_n'])
    plt.figure()
    plt.imshow(result_grid, origin='lower', extent=[grid_cfg['x_min'], grid_cfg['x_max'], grid_cfg['y_min'], grid_cfg['y_max']])
    plt.title(title)
    plt.colorbar(label="Field Value")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Non-Stationary Spectral Simulator")
    parser.add_argument("config_file", help="Path to configuration JSON file")
    args = parser.parse_args()

    run_simulation(args.config_file)
