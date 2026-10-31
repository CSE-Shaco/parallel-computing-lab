#include "monte_carlo.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

static std::uint64_t number(const char* value) {
    const std::string s(value);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer");
    return std::stoull(s);
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("usage: pi_mc serial|local|mutex workers samples seed");
        auto workers = number(argv[2]);
        if (workers > 256) throw std::invalid_argument("too many workers");
        const auto samples = number(argv[3]), seed = number(argv[4]);
        const auto start = std::chrono::steady_clock::now();
        const auto count = lab::estimate(argv[1], static_cast<unsigned>(workers), samples, seed);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout << std::setprecision(15) << "{\"mode\":\"" << argv[1] << "\",\"threads\":" << workers
                  << ",\"samples\":" << samples << ",\"seed\":" << seed << ",\"count\":" << count
                  << ",\"pi\":" << 4.0*count/samples << ",\"ms\":" << ms << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
