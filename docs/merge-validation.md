# Integration Validation

> Historical record: the merge and macOS checks below predate the [successful RTX 4060 WSL2 GPU run](../cuda-kernel-optimization/results/rtx4060-wsl-run-02/README.md). Their GPU-unverified statements describe those earlier stages. Current GPU compilation, correctness, sanitizer and measured-result status is documented with that run; CMake/CTest remains unverified.

Executed on 2026-10-31 after the CUDA project was moved into `cuda-kernel-optimization/`. The root repository is the existing Parallel Computing Lab repository; the CUDA subproject has no nested `.git` directory. No commit, Git initialization, history rewrite, remote change, or push was performed.

## Executed CPU-Side Checks

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py \
  --compiler /opt/homebrew/opt/llvm/bin/clang++ --require-openmp
```

The runner compiled C++11 targets with `-O3 -Wall -Wextra -Wpedantic`, using `-pthread` for Monte Carlo and `-fopenmp` for scheduling. The host reference compiled as C++17 with the same flags plus `-Werror`. Generated files went to a fresh temporary directory, outside the source tree. Existing measurements were not rerun or overwritten.

| Check | Executed outcome |
| --- | --- |
| Monte Carlo independent exact oracle | PASS: 52 configurations |
| Monte Carlo invalid input | PASS: 8 rejected cases |
| OpenMP independent oracle and work-distribution checks | PASS: 126 configurations |
| OpenMP invalid input | PASS: 6 rejected cases |
| Host reference, validation rejection, and input checks | PASS |
| CUDA result-analysis unit tests | PASS: 6 tests, using synthetic parser fixtures only |

The actual LLVM compiler/runtime supported OpenMP. A separate check with Apple's system compiler exercised the unsupported path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_all.py \
  --compiler /usr/bin/c++ --only openmp
```

That compiler rejected `-fopenmp`; the runner reported **SKIPPED**, not passed. `--require-openmp` makes lack of support an error when the OpenMP group is selected.

## Preservation Checks

The pre-merge snapshot includes both complete repositories and their Git metadata. Comparing the moved work against that snapshot found:

- All 20 existing CUDA project files remain. Eighteen are byte-identical; only its README and CMake integration were intentionally edited.
- All 21 existing Parallel project files remain. Nineteen are byte-identical; only the root README and CMake integration were intentionally edited.
- All CUDA kernels, host reference, correctness tests, CPU implementations, independent oracle scripts, benchmark scripts, and retained measurement data are byte-identical to their pre-merge versions.
- The retained run has 99 records, identical Monte Carlo counts across configurations, and `verified=true` in every OpenMP record. Raw JSONL, metadata, summary, and SVG were preserved unchanged.
- All 20 files of the live Parallel Git metadata and all 20 files of the archived live CUDA Git metadata matched their snapshot checksums. The latter is kept outside the active workspace.
- The source tree contains no generated Python cache directories. Test binaries and compiler probes are outside it.

The two new project files are this validation record and `tests/run_all.py`; the merged repository contains 43 nonignored project files. Management snapshots record private backup paths separately.

## Build Configuration and Coverage Limits

The top-level CMake project can select threads, OpenMP, and the CUDA subproject separately. CUDA detection is optional; disabling or lacking a CUDA compiler leaves the host-only reference and analysis available. GPU runtime tests require explicit `PARALLEL_ENABLE_GPU_TESTS=ON`. CTest labels select `threads`, `openmp`, `cuda-host`, or `gpu`; an unfiltered invocation runs all registered tests.

**CMake/CTest execution is Not Verified**, because CMake is unavailable in this environment. The configuration was reviewed and supplied for reproduction; the direct compiler runner above is the executed build/test path.

**CUDA compilation, GPU correctness, Compute Sanitizer, and GPU performance are Not Verified**, because `nvcc` and NVIDIA hardware are unavailable. No historical GPU speedup is adopted. The analysis fixtures are unit-test data, not benchmark results. At the time of this initial merge-only run, ThreadSanitizer had not been run.

The retained CPU timings remain small, uncontrolled exploratory measurements. Passing correctness tests does not establish a production performance advantage or complete race/memory-safety proof.

## Later Full-Verification Addendum

After this initial report, the host CPU reference was corrected to reject non-finite expected and observed values. Seven regression cases cover size mismatch, NaN, and positive/negative infinity. The Monte Carlo CLI test suite passed under ThreadSanitizer: 52 valid configurations and 8 rejected invalid-input cases, with no diagnostics. This TSan result is limited to Monte Carlo; it is not evidence about OpenMP or CUDA. Host Release (`-DNDEBUG`), ASan/UBSan, reanalysis of all 99 retained timing rows, and byte-identical regenerated summary/SVG also passed. CMake/CTest and GPU tests remain unverified. This file records the local merge and verification stages; separately maintained portfolio records contain later run details.
