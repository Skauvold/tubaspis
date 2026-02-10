#pragma once
// Minimal .npy writer for double arrays (NumPy format v1.0).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

inline void save_double(const std::string& path,
                        const double* data,
                        const std::vector<size_t>& shape) {
    // Build the header dict
    // e.g. {'descr': '<f8', 'fortran_order': False, 'shape': (201, 201), }
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        shape_str += ",";
        if (i + 1 < shape.size()) shape_str += " ";
    }
    shape_str += ")";

    std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': " +
                       shape_str + ", }";

    // Pad header to multiple of 64 bytes (magic(6) + version(2) + header_len(2) + dict + '\n')
    size_t prefix_len = 6 + 2 + 2;  // magic + version + header_len field
    size_t unpadded = prefix_len + dict.size() + 1;  // +1 for trailing '\n'
    size_t padding = (64 - (unpadded % 64)) % 64;
    dict.append(padding, ' ');
    dict.push_back('\n');

    uint16_t header_len = static_cast<uint16_t>(dict.size());

    // Total element count
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;

    // Write using C file I/O (avoids MinGW std::ofstream issues)
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    // Magic string
    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    std::fwrite(magic, 1, 6, f);

    // Version 1.0
    const unsigned char ver[2] = {1, 0};
    std::fwrite(ver, 1, 2, f);

    // Header length (little-endian)
    std::fwrite(&header_len, 1, 2, f);

    // Header dict
    std::fwrite(dict.data(), 1, dict.size(), f);

    // Raw data
    std::fwrite(data, sizeof(double), n_elements, f);

    std::fclose(f);
}

}  // namespace npy
