# CUDA GPU Kernel Optimization

This subproject is part of [Parallel Computing Lab](../README.md), alongside OpenMP scheduling and C++ thread synchronization. It has no separate Git repository. The commands below run from `cuda-kernel-optimization/`; the top-level README documents the integrated build and all CPU-side tests.

**What I Built:** A CUDA C++ benchmark for the uncentered second-moment matrix with four implementations: a direct baseline, the original 16×16 shared-memory tile, a coalesced-load 16×16 tile, and a 32×32 register-tiled kernel. It includes a double-precision CPU reference and repeatable validation and measurement tools.

**How It Works:** Given row-major input `X` with `m` samples and `n` features, each implementation computes `S[i,j] = sum(X[e,i] * X[e,j]) / m`. The first three assign one output to each thread. The register-tiled version assigns four output rows to each thread in a 32×8 block.

**What I Verified:** The [RTX 4060 WSL2 four-kernel run](results/rtx4060-wsl-run-03/README.md), pinned to source commit `33c5536`, passed direct CUDA compilation, the host reference and 50 layout/oracle cases, 12 GPU correctness shapes for all four kernels, and Compute Sanitizer memcheck/racecheck. It retained 640 timing samples. Register tiling reached 1.405× baseline median / candidate median at 1024×512, but lost on the smallest case; coalesced-load tiling lost to baseline on all four shapes. These are single-session, kernel-only results. The separate record audit reproduced statistics and the SVG and passed 34 host Python tests without repeating GPU execution on macOS.

## Technical Highlights

- Defines the mathematical operation explicitly: this is `XᵀX/m`, not mean-centered covariance.
- Tests the direct kernel, original tiled kernel, coalesced cooperative loads, and register reuse under one validation/measurement interface.
- Uses zero padding and unconditional block barriers to handle partial tiles safely.
- Compares every GPU output against a double-precision reference before and after timing.
- Uses ten warmups per kernel and rotates execution order across repetitions in the four-kernel workflow; CUDA events time individual launches.
- Retains raw samples and GPU/runtime/build metadata; graphs are generated only from supplied measurements.

## Architecture

`src/kernels/` contains the direct baseline, original 16×16 tile, coalesced-load 16×16 tile, and 32×32 register tile. The coalesced variant changes the order in which the left operand is cooperatively loaded while keeping the output and reduction tile at 16×16. The register variant uses a 32×8 block, a 32×32 output/reduction tile, and four accumulators per thread. `include/tile_layout.hpp` shares bounds and coordinate helpers with a CPU-only mapping oracle. `src/reference/reference.cpp` implements deterministic input, double-precision CPU computation, and absolute-plus-relative tolerance checking; `include/reference.hpp` exposes its host-only API. CUDA allocation and launch validation live in `include/validation.cuh`. `src/main.cu` owns device buffers, correctness gates, and event measurements. Python scripts validate and summarize the four-kernel measurement schema while retaining support for the historical two-kernel results.

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
./build/kernel_benchmark 257 65 40
```

`native` architecture selection requires CMake 3.24+; with CMake 3.20–3.23 replace it with the compute capability of the actual target GPU, such as `86` only for a matching device. The executable accepts positive `samples features repeats`; defaults are `257 65 40`. Full CPU validation costs `O(m*n²)` and intentionally limits practical benchmark sizes. Reduce dimensions before attempting large workloads.

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

The integrated runner also compiles the host tile-layout oracle and runs all 34 host Python tests from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py --only cuda-host --compiler clang++
```

The host layout oracle covers tile mapping and partial-dimension arithmetic; it cannot validate device execution, synchronization, or GPU performance.

The recorded baseline/tiled GPU run covered `(1,1), (1,17), (17,1), (7,9), (16,16), (17,19), (33,31), (257,65)`. The four-kernel correctness executable extends coverage to `(32,32), (33,33), (63,65), (65,63)` and compares every output against the CPU reference using `abs_error <= 2e-4 + 2e-4*abs(reference)`. Run 03 compiled and executed that expanded correctness program for all four variants, then passed memcheck and racecheck on the same program. Run the integrated GPU validation command in [the execution guide](docs/gpu-execution.md); it gates timing on correctness and memcheck/racecheck.

```sh
compute-sanitizer --tool memcheck ./build/correctness_test
compute-sanitizer --tool racecheck ./build/correctness_test
```

## Experimental Results

The [schema-2 four-kernel run](results/rtx4060-wsl-run-03/README.md) used source snapshot `33c5536`, CUDA 13.3 (`nvcc` 13.3.73), `-std=c++17 -O3 -lineinfo -arch=sm_89`, ten warmups per kernel, rotating execution order, and 40 measured launches per kernel/shape. All four variants share the same build, inputs and measurement session.

| Shape (samples × features) | Baseline median (ms) | Tiled median (ms) | Coalesced median (ms) | Register median (ms) | Baseline / register |
| --- | ---: | ---: | ---: | ---: | ---: |
| 257 × 65 | 0.009216 | 0.010240 | 0.013360 | 0.015328 | 0.601× |
| 1024 × 128 | 0.049152 | 0.048848 | 0.058176 | 0.046080 | 1.067× |
| 4096 × 256 | 0.604032 | 0.582528 | 0.700080 | 0.462704 | 1.305× |
| 1024 × 512 | 0.510624 | 0.490496 | 0.585712 | 0.363392 | 1.405× |

Register tiling used about 6.25%, 23.40% and 28.83% less median kernel time on the three larger cases, but about 66.32% more on the smallest. The coalesced-load variant was slower than baseline on every shape, so improved load orientation in the source alone is insufficient evidence of an optimization. The register variant also changes tile dimensions, block shape and per-thread work; the sweep does not isolate which change explains its timings. No profiler counters or actual register-allocation measurements were captured.

The record audit checked 20 source SHA-256 values exactly against the LF bytes at `33c5536`, all three result hashes, all 640 positive finite samples, per-shape raw/combined agreement, independently calculated statistics, and byte-identical SVG regeneration. Fourteen recorded commands returned zero, including the host checks, 12-shape GPU test, memcheck and racecheck. The observed maximum absolute error was `1.4495293e-08` on 257×65 and zero on the three other supplied inputs for all four kernels; this does not establish exact arithmetic or accuracy for other inputs.

The historical [run 02](results/rtx4060-wsl-run-02/README.md) remains immutable and separate: it measured the original two kernels under its own three-warmup/30-repeat policy and is not a denominator for the new variants. The separately verified CPU-only Release CMake build passed 6/6 CTests, including the 50-case layout oracle and 34 host Python tests. The record review passed those 34 Python tests again; it did not repeat CUDA execution on the Mac. CUDA-enabled CMake/GPU CTest remains unverified because the target run used direct `nvcc`. Generated target binaries are not committed, so their recorded hashes cannot be independently rehashed from this checkout.

Clocks, thermal state, power policy and background load were not fixed. These are 40 repeated launches in one session, not 40 independent sessions. Timings exclude transfers and CPU reference work and do not establish end-to-end application throughput.

## Core Technical Challenges

The original coursework called the operation covariance despite omitting mean subtraction. The new project retains the actual second-moment operation and documents this distinction. See [source review](docs/source-review.md) and [methodology](docs/methodology.md).

## Limitations & Future Improvements

- Single-precision accumulation and fixed per-variant tile/block configurations; tolerance is not a general guarantee for arbitrarily long reductions.
- Kernel-only timings; transfer cost and application throughput are not measured.
- All four kernels passed run 03 validation; additional independent sweeps and controlled profiling are still needed to characterize performance beyond that session.
- Repeat the same four-way comparison in independent sessions; profile load efficiency, cache behavior, barrier stalls, actual register usage and occupancy before drawing causal conclusions.
- Investigate symmetry-aware computation, tile-size sweeps and a cuBLAS comparison; these remain future work.
- Publication remains subject to ownership and course-policy review. No license has been inferred or added.

## Implementation Decisions

The CPU reference is ordinary host C++ compiled as its own CMake library, so its API and correctness checks can be built and tested without a CUDA compiler. Kernel sources depend on that API but remain separated by implementation strategy. This keeps host validation available on machines without NVIDIA hardware while leaving device execution and timing gated on a supported CUDA environment.
