#include "validation.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int positive(const char* text) {
    std::size_t used = 0;
    const std::string value(text);
    const long result = std::stol(value, &used);
    if (used != value.size() || result < 1 || result > 1000000)
        throw std::invalid_argument("expected integer in [1,1000000]");
    return static_cast<int>(result);
}

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 4)
            throw std::invalid_argument("usage: kernel_benchmark [samples features repeats]");
        const int m = argc == 4 ? positive(argv[1]) : 257;
        const int n = argc == 4 ? positive(argv[2]) : 65;
        const int repeats = argc == 4 ? positive(argv[3]) : 40;
        constexpr int warmup = 10;
        const auto& variants = second_moment::kernel_variants;
        const auto count = variants.size();
        const auto x = second_moment::input(m, n);
        const auto expected = second_moment::reference(x, m, n);

        cudaDeviceProp prop{};
        int device = 0, runtime = 0, driver = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        second_moment::DeviceBuffer dx(x.size() * sizeof(float));
        second_moment::DeviceBuffer dout(expected.size() * sizeof(float));
        CUDA_CHECK(cudaMemcpy(dx.pointer, x.data(), x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        std::vector<double> errors(count);
        for (std::size_t k = 0; k < count; ++k) {
            errors[k] = second_moment::check_output(
                variants[k].launch, x, expected, m, n, dx, dout);
            for (int w = 0; w < warmup; ++w)
                variants[k].launch(dx.pointer, dout.pointer, m, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        second_moment::Event start, stop;
        std::vector<std::vector<float>> samples(count);
        for (auto& kernel_samples : samples) kernel_samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r) {
            // Rotate the first implementation each repetition. At the default
            // 40 repeats, each of four kernels occupies every position 10 times.
            for (std::size_t order = 0; order < count; ++order) {
                const std::size_t k = (static_cast<std::size_t>(r) + order) % count;
                CUDA_CHECK(cudaEventRecord(start.value));
                variants[k].launch(dx.pointer, dout.pointer, m, n);
                CUDA_CHECK(cudaEventRecord(stop.value));
                CUDA_CHECK(cudaEventSynchronize(stop.value));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
                if (!std::isfinite(ms) || ms <= 0)
                    throw std::runtime_error("invalid elapsed time");
                samples[k].push_back(ms);
            }
        }

        // Validation is outside the timed interval, both before and after it.
        for (std::size_t k = 0; k < count; ++k)
            errors[k] = std::max(errors[k], second_moment::check_output(
                variants[k].launch, x, expected, m, n, dx, dout));

        std::cout << std::setprecision(9)
                  << "{\"schema_version\":2,\"operation\":\"uncentered_second_moment\",\"gpu\":\""
                  << prop.name << "\",\"compute_capability\":\"" << prop.major << "." << prop.minor
                  << "\",\"runtime_version\":" << runtime << ",\"driver_version\":" << driver
                  << ",\"cuda_compiler_version\":\"" << __CUDACC_VER_MAJOR__ << "."
                  << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
                  << "\",\"samples\":" << m << ",\"features\":" << n
                  << ",\"warmup\":" << warmup << ",\"kernel_order\":\"rotating\",\"kernel_names\":[";
        for (std::size_t k = 0; k < count; ++k) {
            if (k) std::cout << ",";
            std::cout << "\"" << variants[k].name << "\"";
        }
        std::cout << "],\"validated\":true,\"results\":[";
        for (std::size_t k = 0; k < count; ++k) {
            if (k) std::cout << ",";
            const auto& variant = variants[k];
            std::cout << "{\"kernel\":\"" << variant.name << "\",\"max_abs_error\":" << errors[k]
                      << ",\"block\":[" << variant.block_x << "," << variant.block_y
                      << "],\"output_tile\":[" << variant.output_tile << "," << variant.output_tile
                      << "],\"reduction_tile\":" << variant.reduction_tile
                      << ",\"outputs_per_thread\":" << variant.outputs_per_thread
                      << ",\"milliseconds\":[";
            for (std::size_t i = 0; i < samples[k].size(); ++i) {
                if (i) std::cout << ",";
                std::cout << samples[k][i];
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
