#include "reference.hpp"
#include "tile_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// This is a host model of the cooperative layout and reduction schedule, not
// execution of a CUDA kernel. Its oracle is the independent direct FP64 sum.
template <class Layout>
std::vector<float> emulate(const std::vector<float>& x, int m, int n) {
    constexpr int tile = Layout::tile;
    const auto output_size = static_cast<std::size_t>(n) * n;
    std::vector<float> out(output_size, std::numeric_limits<float>::quiet_NaN());
    std::vector<unsigned> owners(output_size);
    for (int row_base = 0; row_base < n; row_base += tile) {
        for (int col_base = 0; col_base < n; col_base += tile) {
            std::vector<float> sums(tile * tile);
            for (int base = 0; base < m; base += tile) {
                std::vector<float> left(tile * tile, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> right(tile * tile, std::numeric_limits<float>::quiet_NaN());
                std::vector<unsigned> loads(tile * tile);
                for (int ty = 0; ty < Layout::thread_rows; ++ty) {
                    for (int tx = 0; tx < Layout::thread_columns; ++tx) {
                        for (int slot = 0; slot < Layout::outputs_per_thread; ++slot) {
                            const int local_row = Layout::row_offset(ty, slot);
                            require(local_row >= 0 && local_row < tile, "invalid shared row");
                            const auto shared = static_cast<std::size_t>(local_row) * tile + tx;
                            ++loads.at(shared);
                            const int sample = base + local_row;
                            const int feature_left = row_base + tx, feature_right = col_base + tx;
                            left.at(shared) = second_moment::input_in_bounds(sample, feature_left, m, n)
                                ? x.at(second_moment::input_index(sample, feature_left, n)) : 0.0f;
                            right.at(shared) = second_moment::input_in_bounds(sample, feature_right, m, n)
                                ? x.at(second_moment::input_index(sample, feature_right, n)) : 0.0f;
                            if (sample < m && feature_left + 1 < n && tx + 1 < tile) {
                                require(second_moment::input_index(sample, feature_left + 1, n) ==
                                        second_moment::input_index(sample, feature_left, n) + 1,
                                        "cooperative load must use adjacent feature words");
                            }
                        }
                    }
                }
                require(std::all_of(loads.begin(), loads.end(), [](unsigned count) { return count == 1; }),
                        "each shared cell needs exactly one cooperative writer");
                for (int ty = 0; ty < Layout::thread_rows; ++ty) {
                    for (int tx = 0; tx < Layout::thread_columns; ++tx) {
                        for (int k = 0; k < tile; ++k) {
                            const float value_right = right.at(static_cast<std::size_t>(k) * tile + tx);
                            for (int slot = 0; slot < Layout::outputs_per_thread; ++slot) {
                                const int local_row = Layout::row_offset(ty, slot);
                                sums.at(static_cast<std::size_t>(local_row) * tile + tx) +=
                                    left.at(static_cast<std::size_t>(k) * tile + local_row) * value_right;
                            }
                        }
                    }
                }
            }
            for (int ty = 0; ty < Layout::thread_rows; ++ty) {
                for (int tx = 0; tx < Layout::thread_columns; ++tx) {
                    for (int slot = 0; slot < Layout::outputs_per_thread; ++slot) {
                        const int local_row = Layout::row_offset(ty, slot);
                        const int row = row_base + local_row, col = col_base + tx;
                        if (row < n && col < n) {
                            const auto index = static_cast<std::size_t>(row) * n + col;
                            ++owners.at(index);
                            out.at(index) = sums.at(static_cast<std::size_t>(local_row) * tile + tx) / m;
                        }
                    }
                }
            }
        }
    }
    require(std::all_of(owners.begin(), owners.end(), [](unsigned count) { return count == 1; }),
            "each valid output needs exactly one thread owner");
    return out;
}

std::vector<float> varied_input(int m, int n) {
    std::vector<float> x(static_cast<std::size_t>(m) * n);
    std::uint32_t state = 0x6c8e9cf5u;
    for (auto& value : x) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>(static_cast<int>((state >> 16) % 257) - 128) / 128.0f;
    }
    return x;
}

}  // namespace

int main() {
    try {
        const int shapes[][2] = {{1,1},{1,17},{17,1},{7,9},{16,16},{17,19},
                                 {33,31},{257,65},{32,32},{33,33},{63,65},{65,63}};
        unsigned cases = 0;
        for (const auto& shape : shapes) {
            const int m = shape[0], n = shape[1];
            for (int pattern = 0; pattern < 2; ++pattern) {
                const auto x = pattern ? varied_input(m, n) : second_moment::input(m, n);
                const auto expected = second_moment::reference(x, m, n);
                second_moment::validate(emulate<second_moment::CoalescedLayout>(x, m, n), expected);
                second_moment::validate(emulate<second_moment::RegisterLayout>(x, m, n), expected);
                cases += 2;
            }
        }
        // A nonzero-mean hand example guards the mathematical operation itself.
        const std::vector<float> hand{1, 2, 3, 4, 5, 6};
        const std::vector<double> second_moment{35.0/3, 44.0/3, 44.0/3, 56.0/3};
        second_moment::validate(emulate<second_moment::CoalescedLayout>(hand, 3, 2), second_moment);
        second_moment::validate(emulate<second_moment::RegisterLayout>(hand, 3, 2), second_moment);
        cases += 2;
        std::cout << cases << " host layout/oracle cases passed (not GPU execution)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
