#pragma once
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lab {
// A stateless indexed generator: every sample is independent of its worker.
inline std::uint64_t mix(std::uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
inline bool inside(std::uint64_t index, std::uint64_t seed) {
    // Keep 26 bits per coordinate: x*x+y*y fits exactly in uint64_t.
    const auto x = mix(seed + 2 * index) >> 38;
    const auto y = mix(seed + 2 * index + 1) >> 38;
    return x * x + y * y <= (UINT64_C(1) << 52);
}
inline std::uint64_t estimate(const std::string& mode, unsigned workers,
                              std::uint64_t samples, std::uint64_t seed) {
    if (workers == 0 || workers > 256 || samples == 0 || samples > UINT64_C(1000000000000))
        throw std::invalid_argument("workers must be 1..256; samples must be 1..10^12");
    if (mode != "serial" && mode != "local" && mode != "mutex")
        throw std::invalid_argument("mode must be serial, local, or mutex");
    if (mode == "serial") {
        if (workers != 1) throw std::invalid_argument("serial requires one worker");
        std::uint64_t count = 0;
        for (std::uint64_t i = 0; i < samples; ++i) count += inside(i, seed);
        return count;
    }
    std::vector<std::uint64_t> counts(workers, 0);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    std::mutex lock;
    std::uint64_t shared = 0;
    try {
        for (unsigned t = 0; t < workers; ++t) {
            // Balanced contiguous ranges, including N < workers and remainders.
            const auto first = samples / workers * t + (samples % workers < t ? samples % workers : t);
            const auto last = first + samples / workers + (t < samples % workers);
            threads.emplace_back([&, t, first, last] {
                std::uint64_t local = 0;
                for (auto i = first; i < last; ++i) {
                    if (inside(i, seed)) {
                        if (mode == "mutex") { std::lock_guard<std::mutex> guard(lock); ++shared; }
                        else ++local;
                    }
                }
                counts[t] = local; // One write per worker avoids hot adjacent counters.
            });
        }
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();
    if (mode == "mutex") return shared;
    std::uint64_t sum = 0;
    for (auto count : counts) sum += count;
    return sum;
}
}
