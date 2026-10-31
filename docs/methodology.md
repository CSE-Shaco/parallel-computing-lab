# Benchmark Methodology

The executable workflow and artifact layout are documented in the [CPU Benchmark Guide](../benchmarks/README.md). This page records what each measurement means and which conclusions the data can support.

## Input and Correctness

Use a fixed seed (42 in the retained run) and indexed samples for all Monte Carlo configurations. Counts must agree exactly. OpenMP computes a full serial reference after every timed execution and rejects any mismatch. Python tests independently compute both predicates/workloads.

## Timing Scope

Both executables use `std::chrono::steady_clock` for elapsed algorithm wall time. Monte Carlo includes thread creation, join, and reduction. OpenMP includes entering/leaving the parallel region and statistics updates; workload allocation and the post-run serial validation are excluded. OpenMP runtime setup may happen in calls before timing, so this is not whole-process startup timing. Per-worker time uses `omp_get_wtime()` within the same worker and excludes the final barrier via `nowait`.

The runner executes one discarded warmup process for every configuration. Measured repeats run in new processes in a deterministic shuffled order. Such warmup can warm code/file caches, but does not keep a runtime thread team alive. Default repeats are five; retained smoke evidence uses three. Analyze medians along with minimum/maximum; no confidence interval or statistical significance is claimed.

## Evidence

Each raw run directory stores metadata (UTC timestamp, platform, available CPU model/logical-CPU fields, compiler/version and flags supplied by the operator, OpenMP environment, executable SHA-256) and raw JSONL. Derived summaries and charts are written to a distinct analysis directory and must never replace the raw records or their metadata. The SHA values identify measured binaries; binaries themselves are ignored. An optional compiler probe reports the selected executable's current version, but is not build provenance for the measured binaries. Review the actual build command when recording a new run.

The retained run used Homebrew Clang 23.1.4, macOS 27.2 arm64, 14 reported logical CPUs, C++11, `-O3`, and `-pthread`/`-fopenmp`. The physical CPU model was not recorded. Exact commands and results are recorded in its README. This smoke run validates the measurement pipeline and captures real small-workload timings. It is insufficient for sustained throughput or production optimization claims.

## Reproduction Limits

No CPU affinity, clock/thermal control, isolated host, or CPU hardware counters were used. Other portfolio work could run concurrently. The physical CPU model and efficiency/performance core allocation were not recorded; no specific CPU model is claimed. Increasing repeats cannot repair uncontrolled machine conditions. For a credible performance study, repeat on a quiet host with recorded CPU model, runtime, affinity, frequency policy, larger sample/task sizes, and multiple fresh sessions.
