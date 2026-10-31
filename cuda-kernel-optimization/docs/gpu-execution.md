# Running the GPU Validation and Benchmark

Use this workflow on a Linux host with an actual NVIDIA GPU, a compatible driver,
`nvcc`, and Python 3. No cloud account, remote connection, or upload is performed by
the runner. CMake is not required for this direct-compiler workflow.

First inspect the target host:

```sh
python3 benchmarks/run_gpu_validation.py --preflight-only
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Run commands from `cuda-kernel-optimization/`. Select an explicit architecture
supported by the real device and installed compiler. For example, `sm_75` is only
an example for a compatible compute-capability 7.5 target; do not copy it for an
unidentified GPU. The preflight cannot establish that a GPU context will work;
the GPU correctness executable checks actual execution before any timing run.

With the correctly selected architecture stored in `GPU_ARCH`, run:

```sh
python3 benchmarks/run_gpu_validation.py \
  --arch "$GPU_ARCH" \
  --out results/gpu-run-01 \
  --repeats 30 \
  --shape 257,65 --shape 1024,128 --shape 4096,256
```

Choose a new output directory each time. Existing directories and symlinks are
refused. Missing NVIDIA tools or a failed device inventory probe stop execution
before a result directory is created. Compilation or runtime failures retain a
`failed` execution record and logs; they never constitute measurements.

The runner records device inventory, CUDA compiler version, exact compiler and
execution arguments, source SHA-256 values, binary hashes, and relevant device
selection. It builds and runs the host reference, then the eight GPU correctness
shapes. If `compute-sanitizer` is installed, memcheck and racecheck must pass;
otherwise that check remains explicitly **Not Verified**. Only after correctness
does it build/run the benchmark and validate the requested shapes and repetition
counts. It uses the existing three warmup launches, alternating kernel order,
CUDA-event timing, and per-run CPU reference comparisons.

Successful runs contain:

- `execution.json`: complete status, commands, device/toolchain information, hashes.
- `logs/`: stdout/stderr for each executed build, test, and measurement step.
- `benchmark.json`: raw event samples with actual GPU/runtime metadata.
- `raw/`: per-shape documents retained even if a later shape fails.
- `summary.json` and `kernel-time.svg`: derived only from validated samples.
- `build/`: the executables used in the run, preserved for local evidence.

Before publishing curated results, review log paths and environment metadata and
exclude generated binaries. The full run directory is local evidence, not an
instruction to commit every artifact.

Both kernels compute the uncentered second moment `XᵀX/m`. Timing covers kernels
only, excluding transfers and the CPU reference. FP32 accumulation has a bounded
input/tolerance contract; no general numerical guarantee or GPU speedup follows
from host-only tests. CPU validation scales as `O(samples * features²)`, so begin
with the supplied shapes before increasing the workload.

The runner does not lock GPU clocks, thermal state, power policy, or background
load. Repeats describe the selected session; shared-host measurements do not
establish a universal performance advantage.

No GPU measurement has been executed in the current macOS environment. Host-only
runner safeguard tests do not verify CUDA compilation or device execution.
