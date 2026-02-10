#pragma once
// Anisotropy models: compute per-point Sigma matrices and determinants.

#include <cmath>
#include <vector>
#include <json.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aniso {

using json = nlohmann::json;

// ----- Determinant for DxD stored as flat D*D array -----

template<int D>
inline double determinant(const double* m);

template<>
inline double determinant<2>(const double* m) {
    return m[0] * m[3] - m[1] * m[2];
}

template<>
inline double determinant<3>(const double* m) {
    // Row-major: m[r*3+c]
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// ----- Build Sigma arrays -----

// Returns N*D*D flat array of Sigma matrices and N-length det array.

template<int D>
void build_linear_y(const double* grid,    // N x D, row-major
                    int N,
                    std::vector<double>& sigma,    // out: N*D*D
                    std::vector<double>& det_sigma, // out: N
                    const json& /*config*/) {
    sigma.assign(N * D * D, 0.0);
    det_sigma.resize(N);

    constexpr double min_range = 5.0;
    constexpr double max_range = 30.0;
    constexpr double max_grid_y = 200.0;

    for (int n = 0; n < N; n++) {
        double y = grid[n * D + 1];  // y-coordinate
        if (y < 0.0) y = 0.0;
        if (y > max_grid_y) y = max_grid_y;
        double r = min_range + (max_range - min_range) * (y / max_grid_y);
        double r2 = r * r;

        double* s = &sigma[n * D * D];
        for (int i = 0; i < D; i++) s[i * D + i] = r2;

        det_sigma[n] = determinant<D>(s);
    }
}

template<int D>
void build_lva_azimuth_ramp(const double* grid,
                            int N,
                            std::vector<double>& sigma,
                            std::vector<double>& det_sigma,
                            const json& config) {
    sigma.assign(N * D * D, 0.0);
    det_sigma.resize(N);

    // Ranges
    std::vector<double> ranges = {100.0, 20.0, 4.0};
    if (config.contains("ranges")) {
        ranges = config["ranges"].get<std::vector<double>>();
    }
    double lam1 = ranges[0] * ranges[0];
    double lam2 = ranges[1] * ranges[1];
    double lam3 = (ranges.size() > 2) ? ranges[2] * ranges[2] : 1.0;

    // Azimuth ramp parameters
    double x_min = config["grid"]["x_min"].get<double>();
    double x_max = config["grid"]["x_max"].get<double>();
    double az_start = config.value("azimuth_start", 0.0);
    double az_end   = config.value("azimuth_end", 60.0);

    double x_span = x_max - x_min;
    constexpr double deg2rad = M_PI / 180.0;

    for (int n = 0; n < N; n++) {
        double x = grid[n * D + 0];  // x-coordinate
        double x_norm = (x - x_min) / x_span;
        if (x_norm < 0.0) x_norm = 0.0;
        if (x_norm > 1.0) x_norm = 1.0;

        double theta = (az_start + (az_end - az_start) * x_norm) * deg2rad;
        double c = std::cos(theta);
        double s = std::sin(theta);

        double* S = &sigma[n * D * D];

        // 2x2 block: R * diag(lam1, lam2) * R^T
        S[0 * D + 0] = c * c * lam1 + s * s * lam2;
        S[0 * D + 1] = c * s * (lam1 - lam2);
        S[1 * D + 0] = S[0 * D + 1];
        S[1 * D + 1] = s * s * lam1 + c * c * lam2;

        if constexpr (D == 3) {
            S[2 * D + 2] = lam3;
        }

        det_sigma[n] = determinant<D>(S);
    }
}

// Dispatcher
template<int D>
void build_anisotropy(const double* grid, int N,
                      std::vector<double>& sigma,
                      std::vector<double>& det_sigma,
                      const json& config) {
    std::string mode = config.value("anisotropy", std::string("linear_y"));
    if (mode == "lva_azimuth_ramp") {
        build_lva_azimuth_ramp<D>(grid, N, sigma, det_sigma, config);
    } else {
        build_linear_y<D>(grid, N, sigma, det_sigma, config);
    }
}

}  // namespace aniso
