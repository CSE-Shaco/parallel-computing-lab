#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace second_moment {

std::vector<float> input(int samples, int features) {
    if (samples <= 0 || features <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
    std::vector<float> x(static_cast<std::size_t>(samples) * features);
    // Signed nonzero-mean data distinguishes a second moment from covariance.
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] = static_cast<float>(static_cast<int>((k * 17 + 3) % 31) - 9) / 16.0f;
    }
    return x;
}

std::vector<double> reference(const std::vector<float>& x, int samples, int features) {
    if (samples <= 0 || features <= 0 ||
        x.size() != static_cast<std::size_t>(samples) * features) {
        throw std::invalid_argument("invalid reference dimensions");
    }
    std::vector<double> out(static_cast<std::size_t>(features) * features);
    for (int i = 0; i < features; ++i) {
        for (int j = 0; j < features; ++j) {
            double sum = 0;
            for (int e = 0; e < samples; ++e) {
                sum += static_cast<double>(x[static_cast<std::size_t>(e) * features + i]) *
                       x[static_cast<std::size_t>(e) * features + j];
            }
            out[static_cast<std::size_t>(i) * features + j] = sum / samples;
        }
    }
    return out;
}

double validate(const std::vector<float>& got, const std::vector<double>& expected) {
    if (got.size() != expected.size()) {
        throw std::runtime_error("output size mismatch");
    }
    double max_error = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::abs(static_cast<double>(got[i]) - expected[i]);
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i]) ||
            error > 2e-4 + 2e-4 * std::abs(expected[i])) {
            throw std::runtime_error("CPU/GPU correctness mismatch at index " + std::to_string(i));
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

}  // namespace second_moment
