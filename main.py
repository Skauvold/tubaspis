import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class NonStationarySpectralSimulator:
    """
    Implements the spectral simulation algorithm for non-stationary Gaussian random fields
    as described in Emery and Arroyo (2017).
    """

    def __init__(self, grid_coords, config, L=5000, dim=2):
        """
        Initialize the simulator.

        Args:
            grid_coords (np.ndarray): (N, d) array of coordinates to simulate.
            config (dict): Configuration dictionary.
            L (int): Number of lines (cosine waves) to sum.
            dim (int): Spatial dimension (default 2).
        """
        self.grid = grid_coords
        self.config = config
        self.L = L
        self.d = dim
        self.num_points = grid_coords.shape[0]

    def _sample_proposal_frequencies(self, scale=3.0, nu=0.3):
        """
        Sample L frequency vectors 'u' from the proposal density 'g'.
        ... (rest of method unchanged, implicitly preserved if not replaced,
             but since we are replacing the class methods, we need to keep context or
             use a targeted replacement. I will replace the init and get_local_anisotropy
             but I need to be careful with the replace tool if I don't include the whole class.
             The replace tool works on string matching. I will replace the __init__ method first,
             then _get_local_anisotropy.)
        """
        # ... (This method is not being changed here, so I won't include it in new_string
        #      if I target specific blocks. However, the instruction asks to update methods.
        #      I'll do two replacements to be safe and clean.)
        pass

    # ... (Actual implementation below)

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
        scaling = np.sqrt(alpha_sq / Y).reshape(
            -1, 1
        )  # alpha / sqrt(Y) = sqrt(alpha_sq / Y)
        u_vectors = Z * scaling

        # 4. Compute g(u) values
        # Formula: g(u) = C * (alpha^2 + |u|^2)^(-(nu + d/2))
        # C = Gamma(nu + d/2) * alpha^(2*nu) / (Gamma(nu) * pi^(d/2))

        squared_radii = np.sum(u_vectors**2, axis=1)

        numer = gamma(nu + self.d / 2.0) * (alpha_sq**nu)
        denom = gamma(nu) * (np.pi ** (self.d / 2.0))
        constant_factor = numer / denom

        g_values = constant_factor * (alpha_sq + squared_radii) ** (
            -(nu + self.d / 2.0)
        )

        return u_vectors, g_values

    def _get_local_anisotropy(self, locations):
        """

        Define the non-stationary anisotropy matrix Sigma_x for given locations.



        Args:

            locations (np.ndarray): (N, d) array of locations.



        Returns:

            sigma (np.ndarray): (N, d, d) array of anisotropy matrices.

        """

        # Ensure locations is 2D

        if locations.ndim == 1:

            locations = locations[np.newaxis, :]

        N = locations.shape[0]

        mode = self.config.get("anisotropy", "linear_y")

        if mode == "lva_azimuth_ramp":

            # --- LVA Logic ---

            # 1. Ranges

            # Default to r1=100, r2=20, r3=4 (5x ratios)

            ranges = self.config.get("ranges", [100.0, 20.0, 4.0])

            r1, r2 = ranges[0], ranges[1]

            r3 = ranges[2] if len(ranges) > 2 else 1.0

            lam1, lam2, lam3 = r1**2, r2**2, r3**2

            # 2. Azimuth (Rotation around Z)

            # Varies linearly from 0 to 60 degrees along X

            x_coords = locations[:, 0]

            x_min = self.config["grid"]["x_min"]

            x_max = self.config["grid"]["x_max"]

            # Normalize x to [0, 1]

            x_norm = (x_coords - x_min) / (x_max - x_min)

            x_norm = np.clip(x_norm, 0, 1)

            # Azimuth in degrees (default 0 to 60)
            theta_deg_start = self.config.get('azimuth_start', 0.0)
            theta_deg_end = self.config.get('azimuth_end', 60.0)

            theta = np.deg2rad(
                theta_deg_start + (theta_deg_end - theta_deg_start) * x_norm
            )

            c = np.cos(theta)

            s = np.sin(theta)

            # 3. Construct Sigma = R * Lambda * R.T

            # For rotation around Z:

            # Sigma_xy block:

            # [[ c^2 lam1 + s^2 lam2,   cs(lam1 - lam2) ],

            #  [ cs(lam1 - lam2),       s^2 lam1 + c^2 lam2 ]]

            sigmas = np.zeros((N, self.d, self.d))

            val_00 = (c**2) * lam1 + (s**2) * lam2

            val_01 = c * s * (lam1 - lam2)

            val_11 = (s**2) * lam1 + (c**2) * lam2

            sigmas[:, 0, 0] = val_00

            sigmas[:, 0, 1] = val_01

            sigmas[:, 1, 0] = val_01

            sigmas[:, 1, 1] = val_11

            if self.d == 3:

                # Z-axis is independent (tertiary)

                sigmas[:, 2, 2] = lam3

            return sigmas

        else:

            # --- Default: Linear Y Range Logic ---

            # Use y-coordinate (index 1) to vary range

            if self.d >= 2:

                y_coords = locations[:, 1]

            else:

                y_coords = np.zeros(N)  # Fallback for 1D

            min_range = 5.0

            max_range = 30.0

            max_grid_y = 200.0

            clamped_y = np.clip(y_coords, 0, max_grid_y)

            range_vals = min_range + (max_range - min_range) * (clamped_y / max_grid_y)

            # Construct (N, d, d) matrices

            # Since currently they are diagonal s^2 * I

            sigmas = np.zeros((N, self.d, self.d))

            # Fill diagonal

            squared_ranges = range_vals**2

            # Efficient diagonal filling

            idx = np.arange(self.d)

            sigmas[:, idx, idx] = squared_ranges[:, np.newaxis]

            return sigmas

    def _evaluate_gaussian_spectral_density(self, u, sigmas):
        """
        Evaluates the Gaussian spectral density f_x(u) for a specific u and local sigmas.

        Args:
            u (np.ndarray): (d,) frequency vector.
            sigmas (np.ndarray): (N, d, d) anisotropy matrices.

        Returns:
            densities (np.ndarray): (N,) array of spectral densities.
        """
        # Calculate determinant of Sigmas
        # np.linalg.det works on stacked matrices (..., M, M)
        det_sigmas = np.linalg.det(sigmas)

        # Calculate u.T * Sigma * u for all N
        # shape: (N,)
        # einsum: 'j' is dimension 1 of vector, 'njk' is sigma, 'k' is dimension 2 of vector
        u_sigma_u = np.einsum("j,njk,k->n", u, sigmas, u)

        # Calculate the constant factor
        constant_factor = (det_sigmas**0.5) * ((2 * np.sqrt(np.pi)) ** (-self.d))

        # Calculate the exponential term
        exp_term = np.exp(-0.25 * u_sigma_u)

        return constant_factor * exp_term

    def simulate_univariate_gaussian(self, batch_size=1000):
        """
        Simulates a univariate non-stationary Gaussian field.
        """
        start_time = time.time()
        print(f"Starting Univariate Simulation ({self.d}D)...")
        print("Pre-computing local anisotropies...")

        local_sigmas = self._get_local_anisotropy(self.grid)

        # 1. Generate Proposal Frequencies
        u_vectors, g_vals = self._sample_proposal_frequencies()

        # 2. Generate Random Phases
        phases = np.random.uniform(0, 2 * np.pi, self.L)

        # Initialize accumulator
        field = np.zeros(self.num_points)

        print(f"Summing {self.L} lines (batch size {batch_size})...")

        # 3. Main Loop (Batched)
        for i in range(0, self.L, batch_size):
            # Slice batch
            end = min(i + batch_size, self.L)
            u_batch = u_vectors[i:end]  # (B, d)
            phi_batch = phases[i:end]  # (B,)
            g_batch = g_vals[i:end]  # (B,)

            # Dot products: (N, d) @ (B, d).T -> (N, B)
            dot_prods = np.dot(self.grid, u_batch.T)

            # Vectorized calculation of f_x for all points AND all lines in batch
            # u_batch: (B, d)
            # local_sigmas: (N, d, d)
            # Desired u_sigma_u: (N, B)
            # Formula: u_b^T * Sigma_n * u_b
            # einsum: 'bj,njk,bk->nb'
            u_sigma_u = np.einsum("bj,njk,bk->nb", u_batch, local_sigmas, u_batch)

            det_sigmas = np.linalg.det(local_sigmas)  # (N,)
            # Broadcast det_sigmas (N,) -> (N, B)
            det_sigmas_b = det_sigmas[:, np.newaxis]

            constant_factor = (det_sigmas_b**0.5) * ((2 * np.sqrt(np.pi)) ** (-self.d))
            f_x_vals = constant_factor * np.exp(-0.25 * u_sigma_u)  # (N, B)

            # Broadcast g_batch (B,) -> (1, B)
            weights = np.sqrt(2 * f_x_vals / g_batch[np.newaxis, :])  # (N, B)

            # Accumulate: sum over batch axis (axis 1)
            # cos broadcasting: (N, B) + (B,) -> (N, B)
            term = weights * np.cos(dot_prods + phi_batch[np.newaxis, :])
            field += np.sum(term, axis=1)

        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds.")
        return field / np.sqrt(self.L)

    def simulate_univariate_matern(
        self, nu_local=0.5, varying_shape=False, batch_size=1000
    ):
        """
        Simulates a non-stationary Matern field.
        """
        start_time = time.time()
        print(
            f"Starting Univariate Matern Simulation with nu_local={nu_local} ({self.d}D)..."
        )
        print("Pre-computing local anisotropies...")

        local_sigmas = self._get_local_anisotropy(self.grid)

        u_vectors, g_vals = self._sample_proposal_frequencies()
        phases = np.random.uniform(0, 2 * np.pi, self.L)
        field = np.zeros(self.num_points)

        print(f"Summing {self.L} lines (batch size {batch_size})...")

        for i in range(0, self.L, batch_size):
            end = min(i + batch_size, self.L)
            u_batch = u_vectors[i:end]  # (B, d)
            phi_batch = phases[i:end]  # (B,)
            g_batch = g_vals[i:end]  # (B,)

            # Sample latent variables for the batch
            a_batch = np.random.exponential(1, size=(end - i))  # (B,)

            # Dot products
            dot_prods = np.dot(self.grid, u_batch.T)  # (N, B)

            # u_sigma_u calculation (same as Gaussian)
            u_sigma_u = np.einsum(
                "bj,njk,bk->nb", u_batch, local_sigmas, u_batch
            )  # (N, B)

            det_sigmas = np.linalg.det(local_sigmas)  # (N,)

            # Apply 4*a_l scaling
            # a_batch: (B,)
            scale_factor = 4 * a_batch  # (B,)

            # Broadcast scale_factor: (1, B)
            scale_factor_b = scale_factor[np.newaxis, :]

            # |cS| = c^d |S|
            # (1, B) * (N, 1) -> (N, B)
            det_effective = (scale_factor_b**self.d) * det_sigmas[:, np.newaxis]

            # u_eff_sigma_u = scale * u_sigma_u
            u_eff_sigma_u = scale_factor_b * u_sigma_u  # (N, B)

            constant_factor = (det_effective**0.5) * ((2 * np.sqrt(np.pi)) ** (-self.d))
            f_x_vals = constant_factor * np.exp(-0.25 * u_eff_sigma_u)  # (N, B)

            weights = np.sqrt(2 * f_x_vals / g_batch[np.newaxis, :])

            if nu_local > 0:
                # Weight correction for Matern
                # a^(mu-1) / Gamma(mu)
                # a_batch: (B,) -> (1, B)
                weight_correction = (a_batch ** (nu_local - 1)) / gamma(nu_local)
                weights *= weight_correction[np.newaxis, :]
            else:
                raise ValueError("nu_local must be greater than 0")

            # Accumulate
            term = weights * np.cos(dot_prods + phi_batch[np.newaxis, :])
            field += np.sum(term, axis=1)

        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds.")
        return field / np.sqrt(self.L)


def run_simulation(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Grid setup
    grid_cfg = config["grid"]
    x_range = np.linspace(grid_cfg["x_min"], grid_cfg["x_max"], grid_cfg["x_n"])
    y_range = np.linspace(grid_cfg["y_min"], grid_cfg["y_max"], grid_cfg["y_n"])

    if "z_min" in grid_cfg:
        # 3D case
        z_range = np.linspace(grid_cfg["z_min"], grid_cfg["z_max"], grid_cfg["z_n"])
        xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        grid_coords = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        dim = 3
        print(
            f"Initialized 3D Grid: {grid_cfg['x_n']}x{grid_cfg['y_n']}x{grid_cfg['z_n']}"
        )
    else:
        # 2D case
        # Changed to indexing='ij' for consistency.
        # xv varies along axis 0, yv along axis 1.
        xv, yv = np.meshgrid(x_range, y_range, indexing="ij")
        grid_coords = np.column_stack([xv.ravel(), yv.ravel()])
        dim = 2
        print(f"Initialized 2D Grid: {grid_cfg['x_n']}x{grid_cfg['y_n']}")

    # Simulator setup
    L = config.get("L", 5000)
    batch_size = config.get("batch_size", 1000)
    sim = NonStationarySpectralSimulator(grid_coords, config, L=L, dim=dim)

    # Run
    sim_type = config.get("type", "gaussian").lower()
    if sim_type == "gaussian":
        print("Running Gaussian Simulation...")
        result = sim.simulate_univariate_gaussian(batch_size=batch_size)
        title = "Simulated Univariate Gaussian Field"
    elif sim_type == "matern":
        nu = config.get("matern_nu", 0.5)
        print(f"Running Matern Simulation (nu={nu})...")
        result = sim.simulate_univariate_matern(nu_local=nu, batch_size=batch_size)
        title = f"Simulated Univariate Matern Field (nu={nu})"
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")

    # Plot
    plt.figure()
    if dim == 2:
        # Reshape: (nx, ny) because of indexing='ij'
        result_grid = result.reshape(grid_cfg["x_n"], grid_cfg["y_n"])
        # Transpose for plotting to match usual (x=horizontal, y=vertical) orientation if desired,
        # or just plot. imshow with origin='lower' expects [y, x] typically if we map index 0 to rows.
        # With ij: grid[i, j] -> x[i], y[j].
        # If we want x on horizontal axis, we should transpose to [y, x].
        plt.imshow(
            result_grid.T,
            origin="lower",
            extent=[
                grid_cfg["x_min"],
                grid_cfg["x_max"],
                grid_cfg["y_min"],
                grid_cfg["y_max"],
            ],
        )
        plt.title(title)
    elif dim == 3:
        # Reshape: (nx, ny, nz)
        result_grid = result.reshape(grid_cfg["x_n"], grid_cfg["y_n"], grid_cfg["z_n"])
        # Plot middle Z-slice
        mid_z_idx = grid_cfg["z_n"] // 2
        z_val = grid_cfg["z_min"] + (grid_cfg["z_max"] - grid_cfg["z_min"]) * (
            mid_z_idx / (grid_cfg["z_n"] - 1 if grid_cfg["z_n"] > 1 else 1)
        )

        slice_data = result_grid[:, :, mid_z_idx]
        plt.imshow(
            slice_data.T,
            origin="lower",
            extent=[
                grid_cfg["x_min"],
                grid_cfg["x_max"],
                grid_cfg["y_min"],
                grid_cfg["y_max"],
            ],
        )
        plt.title(f"{title} (Z={z_val:.2f})")

    plt.colorbar(label="Field Value")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Non-Stationary Spectral Simulator"
    )
    parser.add_argument("config_file", help="Path to configuration JSON file")
    args = parser.parse_args()

    run_simulation(args.config_file)
