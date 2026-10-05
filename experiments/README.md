# Raw experiment data

The files in `raw/` are the unedited outputs behind
[docs/experiments.md](../docs/experiments.md), from the development machine (AMD Ryzen 9 9950X3D,
NVIDIA GeForce RTX 5080, Ubuntu 24.04 in WSL2, CUDA 13.3, 4 and 5 October 2026). Only local file
paths were shortened.

| File | What it is |
| --- | --- |
| `raw/machine-profile.txt` | The machine profile written by the third `gpu-offload calibrate --full`: speed curves (flops : flops per second) for every operation, precision and device, plus copy and start-up parameters. This is the file `src/policy.c` reads. |
| `raw/gpu-offload-status.txt` | `gpu-offload status` for that profile: the calibration table in readable form. |
| `raw/measurement-default-threads-report.txt` | Measurement study, first run, with OpenBLAS's default thread count. Kept as a record of a misleading result (see experiment 2). |
| `raw/measurement-16-threads-report.txt` | Measurement study, corrected run: per-process reports of what ten program runs spent in BLAS / LAPACK / FFTW. Written by the tool's earlier measurement-only version, so the layout differs slightly from today's `gpu-offload report`. |
| `raw/end-to-end-level5-report.txt` | Per-call reports of all thirteen workloads run under `gpu-offload run -l 5 --trace`. |
| `raw/end-to-end-timings.txt` | Timing tables: single runs at level 5, medians of 3 at levels 5 and 9 (incomplete), the thread-count experiment, and the matrix-product benchmark. |

Not kept: the binary per-call traces (hundreds of megabytes) and the traces of the first two
calibration runs, which were stored in a temporary directory.

To add data from your machine, see [CONTRIBUTING.md](../CONTRIBUTING.md).
