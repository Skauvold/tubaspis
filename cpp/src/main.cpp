// TUBASPIS C++/OpenMP implementation
// Usage: tubaspis <config.json> [output.npy]

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json.hpp>
#include "simulator.hpp"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <config.json> [output.npy]\n", argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string output_path = (argc >= 3) ? argv[2] : "output.npy";

    // Load config
    json config;
    {
        std::ifstream f(config_path);
        if (!f) {
            std::fprintf(stderr, "Error: cannot open %s\n", config_path.c_str());
            return 1;
        }
        f >> config;
    }

    // Build grid
    auto& gc = config["grid"];
    double x_min = gc["x_min"].get<double>(), x_max = gc["x_max"].get<double>();
    int    x_n   = gc["x_n"].get<int>();
    double y_min = gc["y_min"].get<double>(), y_max = gc["y_max"].get<double>();
    int    y_n   = gc["y_n"].get<int>();

    bool is_3d = gc.contains("z_min");

    if (is_3d) {
        // --- 3D ---
        double z_min = gc["z_min"].get<double>(), z_max = gc["z_max"].get<double>();
        int    z_n   = gc["z_n"].get<int>();

        int N = x_n * y_n * z_n;
        std::vector<double> grid(N * 3);

        // meshgrid with indexing='ij': outer loop x, then y, then z
        int idx = 0;
        for (int ix = 0; ix < x_n; ix++) {
            double x = (x_n > 1) ? x_min + ix * (x_max - x_min) / (x_n - 1) : x_min;
            for (int iy = 0; iy < y_n; iy++) {
                double y = (y_n > 1) ? y_min + iy * (y_max - y_min) / (y_n - 1) : y_min;
                for (int iz = 0; iz < z_n; iz++) {
                    double z = (z_n > 1) ? z_min + iz * (z_max - z_min) / (z_n - 1) : z_min;
                    grid[idx * 3 + 0] = x;
                    grid[idx * 3 + 1] = y;
                    grid[idx * 3 + 2] = z;
                    idx++;
                }
            }
        }

        std::printf("Grid: %dx%dx%d = %d points (3D)\n", x_n, y_n, z_n, N);
        std::vector<size_t> shape = {(size_t)x_n, (size_t)y_n, (size_t)z_n};
        sim::simulate<3>(config, grid.data(), N, shape, output_path);

    } else {
        // --- 2D ---
        int N = x_n * y_n;
        std::vector<double> grid(N * 2);

        // meshgrid with indexing='ij': outer loop x, then y
        int idx = 0;
        for (int ix = 0; ix < x_n; ix++) {
            double x = (x_n > 1) ? x_min + ix * (x_max - x_min) / (x_n - 1) : x_min;
            for (int iy = 0; iy < y_n; iy++) {
                double y = (y_n > 1) ? y_min + iy * (y_max - y_min) / (y_n - 1) : y_min;
                grid[idx * 2 + 0] = x;
                grid[idx * 2 + 1] = y;
                idx++;
            }
        }

        std::printf("Grid: %dx%d = %d points (2D)\n", x_n, y_n, N);
        std::vector<size_t> shape = {(size_t)x_n, (size_t)y_n};
        sim::simulate<2>(config, grid.data(), N, shape, output_path);
    }

    return 0;
}
