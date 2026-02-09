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
        
        Uses a Multivariate T-distribution sampling method which corresponds to 
        the isotropic Matern spectral density.
        """
        # Calculate alpha squared from scale and nu
        alpha_sq = (2 * nu) / (scale**2)
        
        # 1. Generate Z ~ N(0, I_d)
        Z = np.random.normal(0, 1, size=(self.L, self.d))
        
        # 2. Generate Y ~ Chi-square(2*nu)
        # We need independent Y for each line? Yes.
        Y = np.random.chisquare(2 * nu, size=self.L)
        
        # 3. Calculate u vectors: u = Z * (alpha / sqrt(Y))
        # Reshape Y for broadcasting
        scaling = np.sqrt(alpha_sq / Y).reshape(-1, 1) # alpha / sqrt(Y) = sqrt(alpha_sq / Y)
        u_vectors = Z * scaling
        
        # 4. Compute g(u) values
        # Formula: g(u) = C * (alpha^2 + |u|^2)^(-(nu + d/2))
        # C = Gamma(nu + d/2) * alpha^(2*nu) / (Gamma(nu) * pi^(d/2))
        
        squared_radii = np.sum(u_vectors**2, axis=1)
        
        numer = gamma(nu + self.d / 2.0) * (alpha_sq**nu)
        denom = gamma(nu) * (np.pi**(self.d / 2.0))
        constant_factor = numer / denom
        
        g_values = constant_factor * (alpha_sq + squared_radii)**(-(nu + self.d / 2.0))
        
        return u_vectors, g_values

    def _get_local_anisotropy(self, location):
        """
        Define the non-stationary anisotropy matrix Sigma_x for a given location.
        """
        # Use y-coordinate (index 1) to vary range, assuming 2D or 3D
        if self.d >= 2:
            y_coord = location[1]
        else:
            y_coord = 0 # Fallback for 1D
            
        min_range = 5.0
        max_range = 30.0
        max_grid_y = 200.0 

        clamped_y = np.clip(y_coord, 0, max_grid_y)
        range_val = min_range + (max_range - min_range) * (clamped_y / max_grid_y)
        
        sigma_x = (range_val**2) * np.eye(self.d)
        
        return sigma_x

    def _evaluate_gaussian_spectral_density(self, u, sigma):
        """
        Evaluates the Gaussian spectral density f_x(u).
        """
        det_sigma = np.linalg.det(sigma)
        u_sigma_u = np.dot(np.dot(u, sigma), u)
        
        constant_factor = (det_sigma**0.5) * ((2 * np.sqrt(np.pi))**(-self.d))
        exp_term = np.exp(-0.25 * u_sigma_u)
        
        return constant_factor * exp_term

    def simulate_univariate_gaussian(self):
        """
        Simulates a univariate non-stationary Gaussian field.
        """
        print(f"Starting Univariate Simulation ({self.d}D)...")

        # 1. Generate Proposal Frequencies
        u_vectors, g_vals = self._sample_proposal_frequencies()

        # 2. Generate Random Phases
        phases = np.random.uniform(0, 2 * np.pi, self.L)

        # Initialize accumulator
        field = np.zeros(self.num_points)

        # 3. Main Loop
        for l in range(self.L):
            u_l = u_vectors[l]
            phi_l = phases[l]
            g_u = g_vals[l]

            dot_prods = np.dot(self.grid, u_l)

            # Local spectral densities
            f_x_vals = np.zeros(self.num_points)
            for i in range(self.num_points):
                loc = self.grid[i]
                sigma_x = self._get_local_anisotropy(loc)
                f_x_vals[i] = self._evaluate_gaussian_spectral_density(u_l, sigma_x)

            weights = np.sqrt(2 * f_x_vals / g_u)
            field += weights * np.cos(dot_prods + phi_l)

        return field / np.sqrt(self.L)

    def simulate_univariate_matern(self, nu_local=0.5, varying_shape=False):
        """
        Simulates a non-stationary Matern field.
        """
        print(f"Starting Univariate Matern Simulation with nu_local={nu_local} ({self.d}D)...")

        u_vectors, g_vals = self._sample_proposal_frequencies()
        phases = np.random.uniform(0, 2 * np.pi, self.L)
        field = np.zeros(self.num_points)

        for l in range(self.L):
            u_l = u_vectors[l]
            phi_l = phases[l]
            g_u = g_vals[l]
            a_l = np.random.exponential(1) 

            dot_prods = np.dot(self.grid, u_l)

            f_x_vals = np.zeros(self.num_points)
            for i in range(self.num_points):
                loc = self.grid[i]
                base_sigma_x = self._get_local_anisotropy(loc)
                effective_sigma_x = 4 * a_l * base_sigma_x
                f_x_vals[i] = self._evaluate_gaussian_spectral_density(u_l, effective_sigma_x)

            weights = np.sqrt(2 * f_x_vals / g_u)

            if nu_local > 0: 
                weight_correction = (a_l**(nu_local - 1)) / gamma(nu_local)
                weights *= weight_correction
            else:
                raise ValueError("nu_local must be greater than 0")

            field += weights * np.cos(dot_prods + phi_l)

        return field / np.sqrt(self.L)


def run_simulation(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Grid setup
    grid_cfg = config['grid']
    x_range = np.linspace(grid_cfg['x_min'], grid_cfg['x_max'], grid_cfg['x_n'])
    y_range = np.linspace(grid_cfg['y_min'], grid_cfg['y_max'], grid_cfg['y_n'])
    
    if 'z_min' in grid_cfg:
        # 3D case
        z_range = np.linspace(grid_cfg['z_min'], grid_cfg['z_max'], grid_cfg['z_n'])
        xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        grid_coords = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        dim = 3
        print(f"Initialized 3D Grid: {grid_cfg['x_n']}x{grid_cfg['y_n']}x{grid_cfg['z_n']}")
    else:
        # 2D case
        # Changed to indexing='ij' for consistency. 
        # xv varies along axis 0, yv along axis 1.
        xv, yv = np.meshgrid(x_range, y_range, indexing='ij')
        grid_coords = np.column_stack([xv.ravel(), yv.ravel()])
        dim = 2
        print(f"Initialized 2D Grid: {grid_cfg['x_n']}x{grid_cfg['y_n']}")

    # Simulator setup
    L = config.get('L', 5000)
    sim = NonStationarySpectralSimulator(grid_coords, L=L, dim=dim)

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
    plt.figure()
    if dim == 2:
        # Reshape: (nx, ny) because of indexing='ij'
        result_grid = result.reshape(grid_cfg['x_n'], grid_cfg['y_n'])
        # Transpose for plotting to match usual (x=horizontal, y=vertical) orientation if desired,
        # or just plot. imshow with origin='lower' expects [y, x] typically if we map index 0 to rows.
        # With ij: grid[i, j] -> x[i], y[j].
        # If we want x on horizontal axis, we should transpose to [y, x].
        plt.imshow(result_grid.T, origin='lower', extent=[grid_cfg['x_min'], grid_cfg['x_max'], grid_cfg['y_min'], grid_cfg['y_max']])
        plt.title(title)
    elif dim == 3:
        # Reshape: (nx, ny, nz)
        result_grid = result.reshape(grid_cfg['x_n'], grid_cfg['y_n'], grid_cfg['z_n'])
        # Plot middle Z-slice
        mid_z_idx = grid_cfg['z_n'] // 2
        z_val = grid_cfg['z_min'] + (grid_cfg['z_max'] - grid_cfg['z_min']) * (mid_z_idx / (grid_cfg['z_n'] - 1 if grid_cfg['z_n'] > 1 else 1))
        
        slice_data = result_grid[:, :, mid_z_idx]
        plt.imshow(slice_data.T, origin='lower', extent=[grid_cfg['x_min'], grid_cfg['x_max'], grid_cfg['y_min'], grid_cfg['y_max']])
        plt.title(f"{title} (Z={z_val:.2f})")
    
    plt.colorbar(label="Field Value")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Non-Stationary Spectral Simulator")
    parser.add_argument("config_file", help="Path to configuration JSON file")
    args = parser.parse_args()

    run_simulation(args.config_file)
