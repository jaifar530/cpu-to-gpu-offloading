---
title: Install and use
nav_order: 2
description: >-
  How to install gpu-offload on Ubuntu or WSL2, calibrate a machine, run NumPy, R or Octave
  programs with automatic GPU offloading, and every command, environment variable and file it uses.
---

# Install and use gpu-offload

## Requirements

| Requirement | Notes |
| --- | --- |
| Linux x86-64 | Developed on Ubuntu 24.04 inside WSL2 on Windows 11. Native Linux works the same way. |
| NVIDIA GPU and driver | Any GPU that cuBLAS and cuSOLVER support. In WSL2 the Windows driver is used; do not install a Linux driver there. |
| CUDA 13 runtime libraries | cuBLAS and cuSOLVER. The release package was built against CUDA 13.3. |
| Python 3 with NumPy | Used by the command-line tool, not by your programs. |
| A system BLAS / LAPACK | For example OpenBLAS. The calibration workload links to it. |

Without a GPU or without CUDA the tool still installs and its measurement commands work; nothing
is offloaded.

## Install from the release package

```bash
# Download gpu-offload_0.1.1_amd64.deb from
# https://github.com/jaifar530/cpu-to-gpu-offloading/releases
sudo apt install ./gpu-offload_0.1.1_amd64.deb
gpu-offload status
```

The package installs everything under `/usr/lib/gpu-offload/` and a `gpu-offload` command in
`/usr/bin`.

## Build from source

```bash
sudo apt install build-essential cmake ninja-build python3-numpy \
                 libopenblas-dev liblapacke-dev libfftw3-dev
# NVIDIA's CUDA toolkit 13.x, for example cuda-toolkit-13-3 from NVIDIA's apt repository

git clone https://github.com/jaifar530/cpu-to-gpu-offloading.git
cd cpu-to-gpu-offloading
./build.sh test       # builds into ~/build/gpu-offload and runs the test suite
./build.sh package    # builds the .deb
```

The build directory is a complete installation: `~/build/gpu-offload/gpu-offload` runs from there.
If CMake does not find the CUDA toolkit, it builds the measurement part only and says so.

## First steps

### 1. Calibrate the machine

```bash
gpu-offload calibrate
```

This runs each supported operation on the CPU and then on the GPU, in four precisions, at sizes
from 128 up to 4096, and writes the machine profile to `~/.config/gpu-offload/profile`.

| Option | Effect |
| --- | --- |
| `--quick` | Sizes up to 2048: fastest, coarser for large problems. |
| (default) | Sizes up to 4096. |
| `--full` | Sizes up to 8192. This took 31 to 34 minutes on the development machine; the largest sizes dominate, so the smaller settings take much less. |
| `--precs sd` | Only the listed precisions (`s`, `d`, `c`, `z`). |
| `--system` | Writes `/etc/gpu-offload/profile` for all users (needs root). |

Calibrate again after changing the GPU, the CPU, the BLAS library, or its thread count.

Calibration measures the CPU **as your programs will use it**. With OpenBLAS the default of one
thread per logical CPU can be far slower than one thread per physical core; `calibrate` tests this
and prints a note if so. On the development machine `OPENBLAS_NUM_THREADS=16` made NumPy workloads
15 to 25 times faster than the default of 32.

### 2. Run a program

```bash
gpu-offload run python3 train.py
gpu-offload run Rscript analysis.R
gpu-offload run ./my_fortran_solver input.dat
```

At exit a summary is printed to standard error:

```
gpu-offload: level 5 on NVIDIA GeForce RTX 5080 (GPU start-up 0.47 s)
  GPU-capable calls: 1 on the GPU (0.60 s), 1 on the CPU
  estimated time saved by offloading: 0.34 s
  stayed on the CPU, predicted gain too small: 1
```

### 3. Choose how eager it should be

```bash
gpu-offload level        # show
gpu-offload level 7      # save for future runs
gpu-offload run -l 9 python3 train.py   # one run only
```

Level 1 offloads only when a large gain is predicted; level 10 offloads even at a small predicted
loss, to free the CPU. The default is 5. The full table is in [How it works](how-it-works.md#the-level).

### 4. Find out whether your program can benefit at all

```bash
gpu-offload measure python3 train.py
```

This runs the program on the CPU only and reports what share of its run time was inside
math-library calls, split by operation, precision and call size. If that share is small, no GPU
tool will make the program much faster. Then, with a profile in place:

```bash
gpu-offload estimate /path/to/trace     # predicted saving at each level, 1 to 10
```

## Command reference

| Command | What it does |
| --- | --- |
| `gpu-offload calibrate [--quick\|--full] [--precs sdcz] [--system] [--keep]` | Learn this machine and write the profile. `--keep` keeps the raw calibration traces. |
| `gpu-offload run [-l 1-10] [--trace DIR] [--verify] [-q] CMD...` | Run a program with automatic offloading. |
| `gpu-offload measure [-o DIR] CMD...` | Record a program's math-library calls without offloading, then print the report. |
| `gpu-offload report DIR...` | Print the report for recorded traces. |
| `gpu-offload estimate DIR...` | Predict from traces what each level would save. |
| `gpu-offload level [N] [--system]` | Show or set the aggressiveness level. |
| `gpu-offload status` | Show the installation, the GPU, the level and the calibration table. |
| `gpu-offload env` | Print the shell setting that offloads every program started from that shell. |

### Offloading everything started from a shell

```bash
eval "$(gpu-offload env)"     # sets LD_PRELOAD for this shell
python3 a.py; Rscript b.R     # both run with offloading
unset LD_PRELOAD              # back to normal
```

### Checking results on your own program

```bash
gpu-offload run --verify python3 train.py
```

In verify mode every GPU-capable call runs on both devices. The program receives the CPU result,
and the summary reports how many GPU results matched and the largest relative difference. Use this
before trusting offloading on a new workload. It is slower than normal running, by design.

## Settings

### Files

| File | Purpose |
| --- | --- |
| `~/.config/gpu-offload/profile` | Machine profile written by `calibrate` (plain text). |
| `~/.config/gpu-offload/config` | User settings; currently one line, `level N`. |
| `/etc/gpu-offload/profile`, `/etc/gpu-offload/config` | The same for all users; the per-user files win. |

### Environment variables

`gpu-offload run` sets these for you; they matter when you use `LD_PRELOAD` directly.

| Variable | Values | Effect |
| --- | --- | --- |
| `GPU_OFFLOAD_MODE` | `offload` (default), `measure`, `off` | `measure` records calls and never uses the GPU; `off` makes the library inert. |
| `GPU_OFFLOAD_LEVEL` | 1 to 10 | Overrides the configured level. |
| `GPU_OFFLOAD_PROFILE` | path | Use this profile instead of the default locations. |
| `GPU_OFFLOAD_TRACE` | directory | Record every call there (`*.trace`, `*.meta.json`). |
| `GPU_OFFLOAD_TRACE_MAX_MB` | number (default 2048) | Size cap of one trace file. |
| `GPU_OFFLOAD_SUMMARY` | `1` | Print the summary at exit. |
| `GPU_OFFLOAD_VERIFY` | `1` | Verify mode. |
| `GPU_OFFLOAD_FORCE` | `cpu`, `gpu` | Skip the cost model and force a device (used by calibration and tests). |
| `GPU_OFFLOAD_MIN_FLOPS` | number (default 1e6) | Calls with less work than this skip the decision entirely. |
| `GPU_OFFLOAD_BACKEND` | path | Load this GPU backend library instead of the installed one. |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| "this machine is not calibrated, so everything stays on the CPU" | Run `gpu-offload calibrate`. |
| Summary shows 0 calls on the GPU, reason "predicted gain too small" | The calls are too small, or in a precision where your GPU does not win. Check `gpu-offload status` and `gpu-offload measure`. |
| Reason "GPU start-up not yet worth it" | The program did too little heavy math to repay loading CUDA. Normal for short programs. |
| Reason "GPU unavailable" | The backend could not load: CUDA libraries missing, no GPU, or the process is a forked child. `gpu-offload status` shows whether the backend is installed. |
| Reason "not enough free GPU memory" | Another program holds the memory, or the problem is too large. A higher level leaves less memory in reserve. |
| Reason "GPU busy with other programs" | Other programs load the GPU above the level's limit. A higher level tolerates more. |
| CPU timings vary wildly between runs | Thread oversubscription or other load on the machine. Set `OPENBLAS_NUM_THREADS` to the number of physical cores. |
| A program misbehaves under `gpu-offload run` | Run it with `GPU_OFFLOAD_MODE=off` to rule the library out, then with `--verify`, and please [open an issue](https://github.com/jaifar530/cpu-to-gpu-offloading/issues). |
