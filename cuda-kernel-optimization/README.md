# CUDA GPU Kernel Optimization

This subproject is part of [Parallel Computing Lab](../README.md), alongside OpenMP scheduling and C++ thread synchronization. It has no separate Git repository. The commands below run from `cuda-kernel-optimization/`; the top-level README documents the integrated build and all CPU-side tests.

**What I Built:** An independently written CUDA C++ benchmark for the uncentered second-moment matrix, with direct and shared-memory tiled kernels, a double-precision CPU reference, and repeatable measurement tools.

**How It Works:** Given row-major input `X` with `m` samples and `n` features, both kernels compute `S[i,j] = sum(X[e,i] * X[e,j]) / m`. Each thread owns one output element. The tiled implementation reuses input values within a 16×16 block.

**What I Verified:** The CPU reference matches a hand-computed example and rejects incorrect output, including seven size/non-finite regression cases; six Python analysis tests reject invalid or unverified timing data. **The recorded RTX 4060 WSL2 run passed CUDA compilation, GPU correctness, and Compute Sanitizer and captured 180 kernel timing samples.** The supplied results have been audited on macOS without repeating GPU execution. Performance differences are modest and shape-dependent; see [the measured run](results/rtx4060-wsl-run-02/README.md).

## Technical Highlights

- Defines the mathematical operation explicitly: this is `XᵀX/m`, not mean-centered covariance.
- Uses zero padding and unconditional block barriers to handle partial tiles safely.
- Compares every GPU output against a double-precision reference before and after timing.
- Measures individual launches with CUDA events after warmup and alternates kernel order across repetitions.
- Retains raw samples and GPU/runtime/build metadata; graphs are generated only from supplied measurements.

## Architecture

`src/kernels/` contains the direct and tiled CUDA kernels. `src/reference/reference.cpp` implements deterministic input, double-precision CPU computation, and absolute-plus-relative tolerance checking; `include/reference.hpp` exposes its host-only API. CUDA allocation and launch validation live in `include/validation.cuh`. `src/main.cu` owns device buffers, correctness gates, and event measurements. Python scripts run shape sweeps and summarize medians and min–max ranges.

The input and full output matrices reside on the GPU. Transfers and CPU reference work occur outside timed sections. RAII releases device allocations and events; allocation, transfer, event, launch and synchronization errors raise exceptions. Destructors perform best-effort cleanup without throwing.

## Build & Run

For a real Linux/NVIDIA host, the [GPU execution workflow](docs/gpu-execution.md)
provides one command to capture provenance, compile, check correctness, and only
then measure. Its host-only safeguard tests do not verify device execution.

Requirements: CMake 3.20+, a C++17 compiler, and an NVIDIA GPU with a compatible CUDA Toolkit/driver for GPU runtime targets. Without CUDA, the host reference and analysis checks remain available. GPU runtime tests require explicit activation.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DPARALLEL_ENABLE_GPU_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/kernel_benchmark 257 65 15
```

`native` architecture selection requires CMake 3.24+; with CMake 3.20–3.23 replace it with the compute capability of the actual target GPU, such as `86` only for a matching device. The executable accepts positive `samples features repeats`; defaults are `257 65 15`. Full CPU validation costs `O(m*n²)` and intentionally limits practical benchmark sizes. Reduce dimensions before attempting large workloads.

For a CPU-only standalone build, configure with `-DPARALLEL_ENABLE_CUDA=OFF`. The integrated top-level build places GPU executables under `../build/cuda-kernel-optimization/` relative to this subproject. In the benchmark command below, replace the binary argument with `--binary ../build/cuda-kernel-optimization/kernel_benchmark` when using that build. GPU tests default to disabled; a successful CPU-only CTest run does not establish GPU coverage.

```sh
python3 benchmarks/run_benchmark.py --binary build/kernel_benchmark \
  --build-options 'cmake Release; CUDA architectures=native; kernel flags=-O3' \
  --shape 257,65 --shape 1024,128 --shape 4096,256
python3 benchmarks/analyze_results.py results/benchmark.json \
  --figure results/figures/kernel-time.svg
```

Record the exact configure/build commands in `--build-options`; the runner records this user-supplied text without independently inspecting the compiler command. It hashes the executable with SHA-256 and refuses to overwrite existing result files. The analyzer rejects mixed GPU environments and duplicate shapes; choose a separate result file per independent sweep. Never feed historical notebook timings into the new workflow as verified data.

## Testing & Validation

Host checks executed on the supplied macOS arm64 environment:

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -I include \
  src/reference/reference.cpp tests/reference_test.cpp -o /tmp/reference_test
/tmp/reference_test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py'
```

GPU test cases cover shapes `(1,1), (1,17), (17,1), (7,9), (16,16), (17,19), (33,31), (257,65)`, testing each kernel against the CPU reference with `abs_error <= 2e-4 + 2e-4*abs(reference)`. They are implemented but **not executed** here. On the target GPU also run:

```sh
compute-sanitizer --tool memcheck ./build/correctness_test
compute-sanitizer --tool racecheck ./build/correctness_test
```

## Experimental Results

The [retained RTX 4060 WSL2 run](results/rtx4060-wsl-run-02/README.md) compiled the host reference, correctness test and benchmark using CUDA 13.3 (`nvcc` 13.3.73, `-std=c++17 -O3 -lineinfo -arch=sm_89`). Its host reference, eight GPU correctness shapes for both kernels, and before/after-timing CPU comparisons passed. Compute Sanitizer reported zero memcheck errors and zero racecheck errors/warnings on the eight-shape correctness executable.

| Shape (samples × features) | Baseline median (ms) | Tiled median (ms) | Baseline / tiled |
| --- | ---: | ---: | ---: |
| 257 × 65 | 0.009120 | 0.010208 | 0.893× |
| 1024 × 128 | 0.049152 | 0.048272 | 1.018× |
| 4096 × 256 | 0.604160 | 0.582384 | 1.037× |

Each row uses 30 measured launches per kernel after three warmups. Tiling took about 11.93% longer for the smallest shape, and about 1.79%/3.60% less time for the larger shapes. This is a single session with uncontrolled clocks, thermal state, power policy and background load; it is not a general speedup claim. Timings cover kernels only, excluding transfers and CPU work.

The separate macOS review checked all recorded result hashes, exact source correspondence after LF→CRLF conversion, independent statistics, raw/combined agreement, and byte-identical analyzer SVG regeneration. All 14 host Python tests (6 analysis, 8 runner safeguards) passed. Synthetic test fixtures are not GPU evidence. GPU execution was not repeated on the Mac, retained executable hashes cannot be rechecked without the local target binaries, and CMake/CTest remain **Not Verified**.

## Core Technical Challenges

The original coursework called the operation covariance despite omitting mean subtraction. The new project retains the actual second-moment operation and documents this distinction. See [source review](docs/source-review.md) and [methodology](docs/methodology.md).

## Limitations & Future Improvements

- Single-precision accumulation and fixed 16×16 blocks; tolerance is not a general guarantee for arbitrarily long reductions.
- Kernel-only timings; transfer cost and application throughput are not measured.
- Both kernels passed the recorded RTX 4060 validation; additional independent sweeps and controlled profiling are still needed to characterize performance beyond that session.
- Investigate tile shapes, coalescing, symmetry and a cuBLAS comparison only after correctness and baseline measurements are established.
- Publication remains subject to ownership and course-policy review. No license has been inferred or added.

## Implementation Decisions

The CPU reference is ordinary host C++ compiled as its own CMake library, so its API and correctness checks can be built and tested without a CUDA compiler. Kernel sources depend on that API but remain separated by implementation strategy. This keeps host validation available on machines without NVIDIA hardware while leaving device execution and timing gated on a supported CUDA environment.
