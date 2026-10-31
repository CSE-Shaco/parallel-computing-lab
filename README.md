# Parallel Computing with C++, OpenMP, and CUDA

A parallel-computing lab spanning GPU kernels, CPU workload scheduling, and thread synchronization. The three experiments share an emphasis on explicit correctness checks, reproducible inputs, and honest measurement limits.

**What I Built:** Baseline and shared-memory tiled CUDA kernels for `XᵀX/m`, an OpenMP scheduling experiment, and deterministic C++11 Monte Carlo implementations with local or mutex-protected counters.

**How It Works:** Each experiment compares implementations of the same mathematical workload. CPU references or independent integer oracles check results; benchmark tools retain repeated samples and environment metadata.

**What I Verified:** C++ thread and OpenMP builds/tests, finite-value validation in the host CPU reference, and six analysis tests passed. Seven regression cases reject non-finite reference/observed values. Release, ASan/UBSan and Monte Carlo ThreadSanitizer runs passed; ThreadSanitizer covered Monte Carlo only. The existing 99 CPU timing records are preserved. **CUDA compilation, GPU correctness, and GPU performance: Not Verified.** CMake/CTest execution is also unverified because CMake is unavailable here.

## Technical Highlights

- Implement GPU data reuse with shared-memory tiles, guarded boundary loads, and unconditional block barriers.
- Gate CUDA measurements on a double-precision CPU reference, error checks, warmup, synchronization, and repeated event timings.
- Compare OpenMP static/dynamic/guided/auto policies using deterministic balanced and imbalanced workloads.
- Compare serial, local-counter reduction, and per-hit mutex updates on identical samples across 1–32 workers.
- Preserve raw CPU measurements, environment/build metadata, and executable hashes without turning a small smoke run into a general speedup claim.

## Project Structure

```text
cpp-multithreading/             C++11 threads, counters, independent oracle
openmp-scheduling/              Scheduling policies, work distribution, oracle
cuda-kernel-optimization/       CUDA kernels, host reference, tests, GPU tools
  src/kernels/                 Direct and shared-memory tiled implementations
  src/reference/               Host-only FP64 reference implementation
benchmarks/                    CPU measurement runner and retained evidence
tests/run_all.py               One-command CPU-side build and test runner
docs/                          Synchronization, load balancing, methodology
```

CUDA is a subproject of this repository, with no nested Git repository. Its [README](cuda-kernel-optimization/README.md) contains kernel-specific build and measurement instructions.

## CUDA GPU Computing

For row-major input `X` with `m` samples and `n` features, both kernels compute `S[i,j] = sum(X[e,i] * X[e,j]) / m`. This is an **uncentered second-moment matrix**, not mean-centered covariance. Each GPU thread computes one output element.

The baseline repeatedly reads the input from global memory. The 16×16 tiled implementation loads reusable values into block-local shared memory, reducing repeated global-memory fetches within a block. Extra synchronization and shared-memory access patterns also cost time; tiling is an optimization hypothesis until it is measured on an identified GPU.

The implemented correctness gate compares every GPU output against an FP64 CPU reference before and after timing, including non-multiple tile dimensions. FP32 accumulation uses `abs_error <= 2e-4 + 2e-4*abs(reference)` for the supplied bounded inputs; this tolerance is not a general guarantee for arbitrarily long reductions. CUDA APIs and launches are error checked. Three warmup launches per kernel, alternating order, CUDA events, and stop-event synchronization address basic timing mistakes.

No GPU result is included. Kernel timings would exclude CPU reference work and host/device transfers; they would not establish application throughput. See [kernel methodology](cuda-kernel-optimization/docs/methodology.md) and [optimization decisions](cuda-kernel-optimization/docs/optimization.md).

## OpenMP Scheduling and Load Balancing

[The scheduling experiment](openmp-scheduling/README.md) runs static, dynamic, guided, and implementation-selected auto scheduling over uniform and tail-heavy integer tasks. Each result is checked against serial computation. Worker-local statistics record assigned items/iterations and loop duration before the final barrier.

Static scheduling reduces assignment overhead for predictable work. Dynamic scheduling can redistribute irregular work at a per-chunk coordination cost; guided scheduling changes chunk sizes as work is consumed. Auto delegates the choice to the runtime and does not represent separate chunk policies. Chunk size trades assignment overhead against balancing opportunities. The retained small experiment does not identify a universally best policy.

## C++ Multithreading and Synchronization

[The Monte Carlo experiment](cpp-multithreading/README.md) assigns identical indexed samples to every serial/local/mutex configuration. Local counters are reduced after joining; mutex mode locks a shared counter for every hit. An unprotected shared update would be a C++ data race and undefined behavior, so unsafe execution is explained rather than included as a benchmark mode.

A mutex protects correctness while serializing contended updates. More workers can increase startup, scheduling, lock contention, and reduction costs; oversubscription and small workloads can make execution slower. The local-counter design reduces shared writes, but its benefit still depends on work size and the machine.

## Build Instructions

Requirements: a C++11 compiler for threads/OpenMP, a C++17 compiler for the host reference, and Python 3 for CLI/analysis tests. OpenMP additionally needs a compatible compiler/runtime. CUDA GPU targets need a compatible NVIDIA CUDA Toolkit; runtime GPU tests need an NVIDIA device.

One top-level CMake build, with CUDA explicitly disabled:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPARALLEL_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The host reference and analysis tests remain available without CUDA. Missing OpenMP disables only the scheduling target/tests; missing Python disables the Python-driven checks. Read the configuration output to see actual coverage. Use `-DPARALLEL_ENABLE_THREADS=OFF` or `-DPARALLEL_ENABLE_OPENMP=OFF` to disable either CPU experiment deliberately. `-DPARALLEL_BUILD_CUDA_SUBPROJECT=OFF` excludes the entire CUDA/host-reference subproject, allowing a C++11-only CPU build. These controls do not change the preserved source or result data.

To build CUDA when a compiler is installed, leave `PARALLEL_ENABLE_CUDA=ON` (the default). A missing CUDA compiler leaves the CPU targets available. Supply an architecture appropriate to the real device. `-DCMAKE_CUDA_ARCHITECTURES=native` requires CMake 3.24+ and an appropriate detected device; the project minimum is CMake 3.20.

GPU runtime tests are **off by default**, even when CUDA compiles. Enable `-DPARALLEL_ENABLE_GPU_TESTS=ON` only on a suitable NVIDIA host. The GPU test executable is `build/cuda-kernel-optimization/correctness_test`; the benchmark is `build/cuda-kernel-optimization/kernel_benchmark`.

## Testing and Validation

Without CMake or CUDA, run all CPU-side builds/checks from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py --compiler clang++
```

This runner uses a fresh temporary build directory and never benchmarks or overwrites previous outputs. It probes `-fopenmp`; unsupported OpenMP is reported as **SKIPPED**, not passed. Use `--require-openmp` when OpenMP coverage is required. On the verified macOS environment, use `/opt/homebrew/opt/llvm/bin/clang++`; Apple's system compiler may need a separate OpenMP installation/configuration.

Run the dependency-free benchmark analyzer tests alone, without compiling or timing the C++ experiments, with `PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py --only benchmarks`.

Select an experiment with `--only threads`, `--only openmp`, or `--only cuda-host`. Under CTest, use labels:

```sh
ctest --test-dir build -L threads --output-on-failure
ctest --test-dir build -L openmp --output-on-failure
ctest --test-dir build -L cuda-host --output-on-failure
# NVIDIA host only, after enabling PARALLEL_ENABLE_GPU_TESTS:
ctest --test-dir build -L gpu --output-on-failure
```

| Scope | Post-integration result |
| --- | --- |
| Monte Carlo independent integer oracle | 52 comparisons + 8 rejected invalid inputs passed |
| OpenMP independent oracle/distribution checks | 126 comparisons + 6 rejected invalid inputs passed |
| Host CPU reference | Hand-computed/reference rejection checks passed; 7 regression cases reject size mismatch and NaN/±∞ inputs |
| CUDA analysis parser | 6 unit tests passed; synthetic fixtures are not performance evidence |
| CPU benchmark tools | 28 validation tests passed; run with `PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py --only benchmarks` (no C++ build or timing run) |
| CMake/CTest execution | Not Verified; CMake unavailable |
| CUDA compilation, GPU correctness, Compute Sanitizer, performance | Not Verified; CUDA compiler/NVIDIA GPU unavailable |
| Monte Carlo ThreadSanitizer | PASS: all 52 exact cases and 8 invalid-input cases, no diagnostics |
| OpenMP/CUDA ThreadSanitizer | Not run; Monte Carlo TSan result does not cover these runtimes |

See [integration validation](docs/merge-validation.md) for the executed command and preservation checks.

## Benchmark Methodology

CPU measurements include thread/team creation and synchronization in the timed region; process startup and correctness-reference work are excluded. The runner warms each configuration once, shuffles repeat order deterministically, records metadata and executable hashes, and refuses to overwrite raw output. The existing [99-record smoke run](benchmarks/results/macos-arm64-smoke-2026-10-31-v3/README.md) is exploratory and uncontrolled; its legacy combined SVG should not be read as a comparison between unlike workloads.

Use the [CPU Benchmark Guide](benchmarks/README.md) for build/run commands and the raw-measurement → derived-analysis workflow. It keeps Monte Carlo scaling, OpenMP uniform, and OpenMP tail-heavy visualizations separate and explains the provenance and limits of each output. CUDA build and measurement tools remain under `cuda-kernel-optimization/benchmarks/`; no GPU result has been verified.

## Results and Limitations

At 100,000 samples, the retained serial median was 0.083 ms, local-counter medians were 0.121 ms at 4 workers and 0.307 ms at 32 workers, and mutex mode at 4 workers was 1.812 ms. These exploratory values show that this small workload did not amortize overhead; they are not a sustained-throughput or general scalability claim.

No GPU performance improvement is reported. The original course reports/notebooks and supplied OpenMP function remain outside this repository. No license has been assigned automatically; remaining ownership/course-policy review and final author review are tracked separately before publication.
