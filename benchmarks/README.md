# CPU Benchmark Guide

This guide covers measured CPU timing for the Monte Carlo thread-count experiment and the OpenMP scheduling experiment. It describes how to build, run, preserve, and analyze a benchmark run. The CUDA tools have a separate workflow in [the CUDA project guide](../cuda-kernel-optimization/README.md); this CPU runner does not compile or measure GPU kernels.

## Data flow

```text
experiment sources ──build──> build/pi_mc + build/scheduling
                                   │
                              measured runs
                                   ▼
benchmarks/results/<run-id>/  raw.jsonl + metadata.json
                                   │ analyze (read-only input)
                                   ▼
benchmarks/analysis/<run-id>/ README + provenance + summaries + charts
```

`results/` stores measurements and their environment/build provenance. `analysis/` stores derived views. Keep raw measurements and metadata together; do not edit raw rows to make charts or summaries. The analyzer requires a separate output directory, refuses an existing destination, and leaves the input run unchanged. The retained `macos-arm64-smoke-2026-10-31-v3` measurement files are retained evidence; analysis outputs belong in the matching `benchmarks/analysis/` directory.

## Build the CPU experiments

Run commands from the repository root. Create `build/` before invoking the compiler:

```sh
mkdir -p build
clang++ -O3 -std=c++11 -Wall -Wextra -Wpedantic -pthread \
  cpp-multithreading/src/main.cpp -o build/pi_mc
clang++ -O3 -std=c++11 -Wall -Wextra -Wpedantic -fopenmp \
  openmp-scheduling/src/main.cpp -o build/scheduling
```

Use a compiler with an OpenMP runtime for the second command. The compiler, version, and exact flags recorded with a run are operator-supplied metadata. `--compiler-executable` can record the current version of a selected compiler, but does not prove that it built the measured binaries. Confirm that the compiler label and flags describe the binaries you actually built.

## Record a new run

Choose a new, unique result directory for each run. The runner refuses to overwrite an existing output directory. From the repository root:

```sh
python3 benchmarks/scripts/run_benchmarks.py \
  --build build \
  --out benchmarks/results/example-run-01 \
  --samples 100000 --repeats 3 --items 300 \
  --light 60 --heavy 3000 \
  --compiler 'compiler name and version' \
  --flags '-O3 -std=c++11; -pthread for pi_mc; -fopenmp for scheduling' \
  --timeout 30
```

The runner records raw JSONL measurements plus metadata: timestamp, platform, available CPU model/count fields, compiler/flags, executable SHA-256 hashes, OpenMP environment, workload arguments, and missing binaries. New runs capture a CPU model when the host exposes one; this field is not backfilled into the retained v3 record. The runner performs one discarded warmup per configuration, then runs the requested repeats in deterministically shuffled order. `--timeout` bounds each executable invocation. The Monte Carlo count and OpenMP checksums/work distributions are validated across configurations. A failed run is retained with failure status and partial evidence; the analyzer accepts only complete, validated runs. The runner does not control affinity, frequency, thermal state, or other system load. The repository ignores `results/local-*` for disposable runs; review source, metadata, and result ownership before intentionally tracking generated artifacts.

`--samples`, `--repeats`, `--items`, `--light`, and `--heavy` set workload sizes. The current suite sweeps Monte Carlo serial/local/mutex modes and 1/2/4/8/16/32 worker counts. OpenMP cases use uniform and tail-heavy task costs, static/dynamic/guided scheduling with chunk sizes 1/10/100, and `auto` with no user chunk (`chunk=0`). Auto is one runtime-selected policy per workload, not a chunk-size sweep.

## Derive summaries and plots

Point the analyzer at the raw run and a new, separate analysis directory:

```sh
python3 benchmarks/scripts/analyze_results.py \
  benchmarks/results/macos-arm64-smoke-2026-10-31-v3 \
  --out benchmarks/analysis/reanalysis-example-01
```

The derived directory contains a README and provenance record, summary JSON/CSV, a Monte Carlo scaling chart, and separate OpenMP uniform and tail-heavy charts. Provenance includes hashes of the raw run inputs; it distinguishes input-file integrity from build/compiler attestation. The analyzer validates the complete metadata-declared configuration/repeat matrix, finite measurements, output consistency, worker workload totals, and the legacy summary if one is present before writing any output. These are transformations of the input records; deriving charts does not execute benchmarks or add measurements. Re-run analysis to a different new directory if the analysis implementation changes, preserving prior derived outputs for comparison.

Inspect the retained [analysis README](analysis/macos-arm64-smoke-2026-10-31-v3/README.md), [CSV](analysis/macos-arm64-smoke-2026-10-31-v3/summary.csv), and individual charts: [Monte Carlo](analysis/macos-arm64-smoke-2026-10-31-v3/monte-carlo-scaling.svg), [OpenMP uniform](analysis/macos-arm64-smoke-2026-10-31-v3/openmp-uniform.svg), and [OpenMP tail-heavy](analysis/macos-arm64-smoke-2026-10-31-v3/openmp-tail.svg).

The Monte Carlo view groups the same sample count by execution mode and worker count, shows repeat ranges and a serial baseline, and uses a log time scale. OpenMP uniform and tail-heavy charts are separate because they are different workload shapes. They show policy/chunk timing ranges and any supported actual-team/load-balance fields. Neither chart ranks unrelated workloads on a common throughput axis.

## Interpret the retained smoke run

The existing v3 raw set has 99 records: 39 Monte Carlo records and 60 OpenMP records, each configuration measured three times. It was recorded on macOS 27.2 arm64 with 14 reported logical CPUs. The physical CPU model and core placement were not recorded. Several configurations are sub-millisecond, and the overall run is exploratory data from an uncontrolled host; thread/team creation and synchronization fall within the internal timing interval, while process startup and correctness-reference work do not. The raw set has no affinity, thermal, frequency, or background-load control.

These records validate the measurement and analysis path. They do not support a universal scheduling policy, general scalability result, or production speedup claim. The archived combined SVG is a legacy visualization; use the separated derived charts for experiment-specific inspection. See [the run record](results/macos-arm64-smoke-2026-10-31-v3/README.md) and [methodology](../docs/methodology.md) for its exact environment and limits.

The analysis tooling itself can be checked without a C++ compiler or new timing run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py --only benchmarks
```

The current benchmark-tool suite passes 28 tests, including malformed/incomplete record rejection, raw-input preservation, and runner failure handling. These are analysis/runner checks, not timing experiments.

## Benchmark boundary

The CPU runner has no CUDA build or GPU timing path. CUDA kernel build, correctness, Compute Sanitizer, and performance remain **Not Verified** in the current environment. Do not reuse course-report GPU speedup claims. Follow the [CUDA guide](../cuda-kernel-optimization/README.md) and report device results only after running on a recorded NVIDIA system.
