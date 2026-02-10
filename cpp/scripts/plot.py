#!/usr/bin/env python3
"""Visualize .npy output from the C++ TUBASPIS simulator."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot TUBASPIS .npy output")
    parser.add_argument("npy_file", help="Path to .npy output file")
    parser.add_argument("config_file", help="Path to config JSON file")
    args = parser.parse_args()

    data = np.load(args.npy_file)
    with open(args.config_file) as f:
        config = json.load(f)

    gc = config["grid"]
    extent = [gc["x_min"], gc["x_max"], gc["y_min"], gc["y_max"]]

    plt.figure(figsize=(8, 6))

    if data.ndim == 2:
        # 2D field: shape (x_n, y_n)
        plt.imshow(data.T, origin="lower", extent=extent)
        plt.title("Simulated Field (2D)")
    elif data.ndim == 3:
        # 3D field: shape (x_n, y_n, z_n) — plot middle z-slice
        mid = data.shape[2] // 2
        z_min, z_max, z_n = gc["z_min"], gc["z_max"], gc["z_n"]
        z_val = z_min + (z_max - z_min) * mid / max(z_n - 1, 1)
        plt.imshow(data[:, :, mid].T, origin="lower", extent=extent)
        plt.title(f"Simulated Field (Z={z_val:.1f})")
    else:
        print(f"Unexpected array shape: {data.shape}")
        return

    plt.colorbar(label="Field Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
