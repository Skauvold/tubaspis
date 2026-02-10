#pragma once
// Core simulation kernels, templated on spatial dimension D.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <chrono>

#include <json.hpp>
#include "anisotropy.hpp"
#include "npy.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sim {

using json = nlohmann::json;

// ===== Small linear algebra (fully inlined for fixed D) =====

template<int D>
inline double dot_product(const double* a, const double* b) {
    double r = 0.0;
    for (int i = 0; i < D; i++) r += a[i] * b[i];
    return r;
}

// u^T * Sigma * u, Sigma is D*D row-major
template<int D>
inline double quadratic_form(const double* u, const double* sigma) {
    double r = 0.0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            r += u[i] * sigma[i * D + j] * u[j];
    return r;
}

// ===== Proposal frequency sampling (multivariate-t) =====

struct ProposalData {
    std::vector<double> u;       // L * D, row-major frequency vectors
    std::vector<double> g;       // L, proposal density values
    std::vector<double> phases;  // L, uniform [0, 2pi]
};

template<int D>
ProposalData sample_proposal(int L, double scale, double nu, std::mt19937_64& rng) {
    ProposalData pd;
    pd.u.resize(L * D);
    pd.g.resize(L);
    pd.phases.resize(L);

    double alpha_sq = (2.0 * nu) / (scale * scale);

    std::normal_distribution<double> normal(0.0, 1.0);
    std::chi_squared_distribution<double> chi2(2.0 * nu);
    std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);

    // g(u) constant: C = Gamma(nu + D/2) * alpha^(2*nu) / (Gamma(nu) * pi^(D/2))
    double g_constant = std::tgamma(nu + D / 2.0) * std::pow(alpha_sq, nu)
                      / (std::tgamma(nu) * std::pow(M_PI, D / 2.0));

    for (int l = 0; l < L; l++) {
        double Z[D];
        for (int i = 0; i < D; i++) Z[i] = normal(rng);

        double Y = chi2(rng);
        double scaling = std::sqrt(alpha_sq / Y);

        double* ul = &pd.u[l * D];
        double sq_radius = 0.0;
        for (int i = 0; i < D; i++) {
            ul[i] = Z[i] * scaling;
            sq_radius += ul[i] * ul[i];
        }

        // g(u) = C * (alpha_sq + |u|^2)^(-(nu + D/2))
        pd.g[l] = g_constant * std::pow(alpha_sq + sq_radius, -(nu + D / 2.0));

        pd.phases[l] = uniform(rng);
    }
    return pd;
}

// ===== Gaussian simulation kernel =====

template<int D>
void simulate_gaussian(const double* grid,  // N x D
                       int N, int L,
                       const std::vector<double>& sigma,      // N*D*D
                       const std::vector<double>& det_sigma,   // N
                       const ProposalData& pd,
                       std::vector<double>& field) {
    field.assign(N, 0.0);

    // Per-line weight: B_l = sqrt(2 / g_l)
    std::vector<double> B(L);
    for (int l = 0; l < L; l++) {
        B[l] = std::sqrt(2.0 / pd.g[l]);
    }

    double inv_sqrt_L = 1.0 / std::sqrt(static_cast<double>(L));

    // Per-point scale: A_n = det(Sigma_n)^(1/4) * (2*sqrt(pi))^(-D/2)
    double geo_const = std::pow(2.0 * std::sqrt(M_PI), -static_cast<double>(D) / 2.0);

    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        double A_n = std::pow(det_sigma[n], 0.25) * geo_const;
        const double* Sn = &sigma[n * D * D];
        const double* xn = &grid[n * D];

        double acc = 0.0;
        for (int l = 0; l < L; l++) {
            const double* ul = &pd.u[l * D];
            double dot = dot_product<D>(ul, xn);
            double uSu = quadratic_form<D>(ul, Sn);
            acc += B[l] * std::exp(-0.25 * uSu) * std::cos(dot + pd.phases[l]);
        }
        field[n] = A_n * acc * inv_sqrt_L;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double lines_per_sec = static_cast<double>(L) / elapsed;
    std::printf("  Kernel time: %.3f s  (%.0f lines/s)\n", elapsed, lines_per_sec);
}

// ===== Matern simulation kernel =====
//
// Scale-mixture of Gaussians: f_x with effective Sigma = 4*a_l * Sigma_n.
// Factor A_n = det(Sigma_n)^(1/4) * (2*sqrt(pi))^(-D/2) out of the inner loop.
// Per-line: C_l = sqrt(2/g_l) * (4*a_l)^(D/4) * a_l^(nu-1) / Gamma(nu)
// Inner loop exp coefficient: -0.5 * a_l (vs -0.25 for Gaussian)

template<int D>
void simulate_matern(const double* grid,
                     int N, int L,
                     const std::vector<double>& sigma,
                     const std::vector<double>& det_sigma,
                     const ProposalData& pd,
                     double nu_local,
                     std::mt19937_64& rng,
                     std::vector<double>& field) {
    field.assign(N, 0.0);

    // Sample latent variables a_l ~ Exp(1)
    std::vector<double> a_vals(L);
    {
        std::exponential_distribution<double> exp_dist(1.0);
        for (int l = 0; l < L; l++) a_vals[l] = exp_dist(rng);
    }

    double geo_const = std::pow(2.0 * std::sqrt(M_PI), -static_cast<double>(D) / 2.0);
    double inv_gamma_nu = 1.0 / std::tgamma(nu_local);

    std::vector<double> C(L);
    for (int l = 0; l < L; l++) {
        double a = a_vals[l];
        C[l] = std::sqrt(2.0 / pd.g[l])
             * std::pow(4.0 * a, static_cast<double>(D) / 4.0)
             * std::pow(a, nu_local - 1.0)
             * inv_gamma_nu;
    }

    double inv_sqrt_L = 1.0 / std::sqrt(static_cast<double>(L));

    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        double A_n = std::pow(det_sigma[n], 0.25) * geo_const;
        const double* Sn = &sigma[n * D * D];
        const double* xn = &grid[n * D];

        double acc = 0.0;
        for (int l = 0; l < L; l++) {
            const double* ul = &pd.u[l * D];
            double dot = dot_product<D>(ul, xn);
            double uSu = quadratic_form<D>(ul, Sn);
            acc += C[l] * std::exp(-0.5 * a_vals[l] * uSu) * std::cos(dot + pd.phases[l]);
        }
        field[n] = A_n * acc * inv_sqrt_L;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double lines_per_sec = static_cast<double>(L) / elapsed;
    std::printf("  Kernel time: %.3f s  (%.0f lines/s)\n", elapsed, lines_per_sec);
}

// ===== Top-level simulate function =====

template<int D>
void simulate(const json& config,
              const double* grid,
              int N,
              const std::vector<size_t>& grid_shape,
              const std::string& output_path) {
    int L = config.value("L", 5000);
    double proposal_scale = config.value("proposal_scale", 3.0);
    double proposal_nu    = config.value("proposal_nu", 0.3);
    unsigned seed         = config.value("seed", 42u);

    std::printf("Dimension: %d, Grid points: %d, Lines: %d\n", D, N, L);

    // RNG
    std::mt19937_64 rng(seed);

    // Build anisotropy
    std::vector<double> sigma, det_sigma;
    aniso::build_anisotropy<D>(grid, N, sigma, det_sigma, config);
    std::printf("Anisotropy matrices built.\n");

    // Sample frequencies
    std::printf("Sampling %d proposal frequencies...\n", L);
    auto pd = sample_proposal<D>(L, proposal_scale, proposal_nu, rng);

    // Simulate
    std::string sim_type = config.value("type", std::string("gaussian"));
    std::vector<double> field;

    if (sim_type == "gaussian") {
        std::printf("Running Gaussian simulation...\n");
        simulate_gaussian<D>(grid, N, L, sigma, det_sigma, pd, field);
    } else if (sim_type == "matern") {
        double nu = config.value("matern_nu", 0.5);
        std::printf("Running Matern simulation (nu=%.2f)...\n", nu);
        simulate_matern<D>(grid, N, L, sigma, det_sigma, pd, nu, rng, field);
    } else {
        throw std::runtime_error("Unknown simulation type: " + sim_type);
    }

    // Save
    npy::save_double(output_path, field.data(), grid_shape);
    std::printf("Output saved to %s\n", output_path.c_str());
}

}  // namespace sim
