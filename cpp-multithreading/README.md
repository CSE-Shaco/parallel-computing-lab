# Deterministic Monte Carlo π Estimation

**What I Built:** A C++11 experiment with serial execution, per-worker local counters, and a shared counter protected by a mutex.

**How It Works:** For sample `i`, a stateless 64-bit integer mixer generates two 26-bit coordinates from `seed + 2*i` and `seed + 2*i + 1`. An exact integer circle test increments the hit count. Every mode receives the same samples; changing thread count only changes their partition.

**What I Verified:** An independent Python implementation checked exact hit counts in 52 configurations, including 1/2/4/8/16/32 workers, more workers than samples, remainder partitions, and unsigned seed wraparound. Eight malformed/invalid argument cases were rejected.

## Technical Highlights

- Separate workload generation from scheduling through indexed random samples.
- Assign each worker a contiguous, balanced range with no skipped remainder.
- Write local totals once and reduce them only after `join()`.
- Acquire a mutex per hit in shared-counter mode to expose synchronization cost.

## Build & Run

From the repository root:

```sh
mkdir -p build
clang++ -O3 -std=c++11 -Wall -Wextra -Wpedantic -pthread cpp-multithreading/src/main.cpp -o build/pi_mc
./build/pi_mc serial 1 1000000 42
./build/pi_mc local 8 1000000 42
./build/pi_mc mutex 8 1000000 42
python3 cpp-multithreading/tests/test_cli.py build/pi_mc
```

Arguments: `mode workers samples seed`. Workers must be 1..256; samples 1..10^12; seed is an unsigned 64-bit decimal integer. Serial requires one worker. Output is one JSON record with hit count, π estimate, and elapsed milliseconds. Successful execution does not imply high statistical precision for small sample counts.

## Synchronization Decisions

Each local worker updates a stack counter; the parent sums per-worker totals after joining. The shared mode locks only the increment, leaving sample computation parallel. This makes the lock a serialized hot spot. An unprotected shared increment would be a C++ data race and undefined behavior; there is no unsafe executable mode. See [synchronization](../docs/synchronization.md).

## Measured Results and Limits

[Exploratory measurements](../benchmarks/results/macos-arm64-smoke-2026-10-31-v3/README.md) cover all requested worker counts. With 100,000 samples, the serial median was 0.083 ms, compared with 0.121 ms for 4 local workers and 1.812 ms for 4 mutex workers. Thread creation/join is included in the timed interval; process startup is excluded. These small timings do not establish useful scaling or a general performance ratio. See the [CPU Benchmark Guide](../benchmarks/README.md) for raw/derived data handling and chart interpretation.

Thread count can exceed available execution capacity. Startup/join costs, mutex contention, oversubscription, and memory/cache effects can outweigh parallel work. Future experiments should vary sample count and record scheduling/hardware conditions. Original notebook outputs are not reused as portfolio evidence.
