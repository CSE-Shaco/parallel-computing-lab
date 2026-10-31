# Synchronization and Reproducibility

A read-modify-write on a plain shared integer is not atomic. Concurrent unsynchronized increments form a data race in C++; a plausible count does not make such a program valid. This project explains the failure without executing an intentionally undefined mode or comparing undefined behavior as a performance baseline.

The serial and local modes execute the same exact integer circle predicate. Thread-local stack counters avoid per-hit synchronization; each worker writes its result once, and the parent observes it after joining. The mutex mode serializes increments but keeps sample generation outside the critical section. Both produce identical counts for every seed and sample count.

The input domain is a 26-bit grid in the unit square. Comparing `x*x+y*y` against `2^52` uses exact uint64 arithmetic and avoids floating-point rounding differences between oracle and executable. The mixer has defined unsigned wraparound. This choice provides deterministic experimental samples; it does not prove independent random samples or certify Monte Carlo statistical quality.

Remainders are assigned one per worker until exhausted. Empty partitions are legal when samples are fewer than workers. In thread-creation failure handling, already-started workers are joined before propagating the exception. A later verification run passed the full Monte Carlo CLI suite under ThreadSanitizer (52 valid cases plus 8 rejected invalid inputs). This does not cover OpenMP or CUDA runtimes; their ThreadSanitizer status remains unverified.

A local benchmark is not a universal scaling curve. Tiny workloads emphasize startup, many threads can oversubscribe CPUs, and a contended mutex imposes a serialized segment. Treat these as mechanisms to investigate with controlled runs; see the measured evidence and methodology before interpreting timings.
