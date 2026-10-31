# Coursework source review

The historical Assignment 3 materials were reviewed privately to identify the actual operation and measurement risks. They remain outside this repository in the portfolio source backup. No notebook, PDF, captured output or course material is included here. This implementation was written independently rather than publishing the notebook cells.

## Observations

- Both historical kernels calculate `XᵀX/m`; there is no feature-mean subtraction. This is a second-moment/Gram matrix, not centered covariance.
- The original 8×8 shared-memory indexing appears consistent with that operation, including zero padding at sample and feature boundaries. This is a source-level observation, not GPU correctness evidence.
- The original program allocates roughly 2 GB of input for `202599×2475` floats and computes `2475²` outputs. Allocation failure or incompatible hardware could invalidate a run, but CUDA return codes were not checked.
- Outputs were never copied back or compared. No CPU reference or independent expected value established correctness.
- CUDA events and stop-event synchronization were present, so the original code did not simply measure host launch latency. However event creation, recording, synchronization and elapsed-time errors were unchecked.
- Each process timed the direct kernel first and the tiled kernel second with no warmup. Five subprocess launches do not remove cold-start or fixed-order bias.
- The timing parser did not require a successful subprocess return code before trusting output.

## Changes in this project

The mathematical contract is explicit. Small deterministic signed inputs and an FP64 CPU reference gate measurements. Every timed launch checks launch status and synchronizes through its stop event. The initial two-kernel harness used three warmups and alternating order; the current four-kernel harness uses ten warmups and rotating order to reduce simple order bias. Raw samples, dimensions, GPU, compute capability, CUDA compiler/runtime and driver versions are retained. Every implementation is checked again after measurements.

No historical coursework performance number is reused. The [four-kernel run 03](../results/rtx4060-wsl-run-03/README.md) now records direct CUDA compilation, correctness, sanitizer checks and same-session timings for the independent implementation at source snapshot `33c5536`. The observations above concern the historical coursework source, not a failure of that validated run. Independently written source does not establish course-policy clearance: the portfolio publishing checklist retains that separate requirement.
