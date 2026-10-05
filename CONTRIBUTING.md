# Contributing to gpu-offload

Contributions are welcome: code, measurements from your machine, bug reports, documentation and
ideas. The project is at version 0.1 and has been tested on a single machine, so there is a lot of
useful work at every level of difficulty.

## Ways to help

| Contribution | Effort | How |
| --- | --- | --- |
| **Share your machine's calibration** | 15 to 40 minutes, no coding | Run `gpu-offload calibrate`, then open a "Machine report" issue with the output of `gpu-offload status`. Every new CPU / GPU pair tells us where the cost model holds. |
| **Run the benchmark on a quiet machine** | 30 minutes, no coding | `python3 workloads/bench.py --reps 3 --levels 5,9` and post the table. The published benchmark is incomplete. |
| **Report a program that misbehaves** | Small | Open a bug report with the output of `gpu-offload run --verify your-command`. |
| **Add an operation** | Medium | See "Adding an operation" below. Good candidates: QR (`geqrf` / `orgqr`), `syevr`, `getrs` / `potrs`, `symm`, `trmm`. |
| **Offload FFT through cuFFT** | Medium to large | FFTW calls are already intercepted and their plans tracked in `src/fftw.c`. |
| **Keep data on the GPU between calls** | Large, research-grade | The main open problem. Start a discussion before writing code. |
| **Another GPU backend** (AMD, Intel, portable) | Large | Implement `struct ofl_backend` from `src/backend.h` in a new shared object. |
| **Improve the docs** | Any | The docs are the Markdown files in `docs/`. |

Issues labelled [`good first issue`](https://github.com/jaifar530/cpu-to-gpu-offloading/labels/good%20first%20issue)
and [`help wanted`](https://github.com/jaifar530/cpu-to-gpu-offloading/labels/help%20wanted) are a
good place to start.

## Development setup

Ubuntu 24.04 (native or WSL2) is the tested environment.

```bash
sudo apt install build-essential cmake ninja-build python3-numpy \
                 libopenblas-dev liblapacke-dev libfftw3-dev
# For the GPU backend: NVIDIA's CUDA toolkit 13.x (for example cuda-toolkit-13-3).

git clone https://github.com/jaifar530/cpu-to-gpu-offloading.git
cd cpu-to-gpu-offloading
./build.sh test
```

`./build.sh` configures and builds into `~/build/gpu-offload` (override with `GPU_OFFLOAD_BUILD`).
That directory is a complete installation: run `~/build/gpu-offload/gpu-offload`.

**You do not need a GPU to contribute.** Without CUDA the build produces the measurement part, and
the test suite runs the parts that need no GPU (measure mode, the reference checks on the CPU
libraries, and the fallback path). That is also what the continuous integration runs.

## Tests

`./build.sh test` runs `tests/run_tests.sh`:

1. **Measure mode**: a test program's trace must contain exactly the calls it makes.
2. **GPU path**: `tests/test_gpu_ops.c` checks every offloaded operation in four precisions against
   references it computes itself, once on the CPU libraries and once forced onto the GPU.
3. **Verify mode**: GPU results are compared with CPU results.
4. **Fallback**: with the backend missing, everything still passes on the CPU.
5. **Policy**: a synthetic profile must steer the decision both ways.
6. **Overhead**: prints the per-call cost of the library.

Steps 2 (GPU half), 3 and 5 are skipped automatically on a machine without a GPU.

Please make sure the suite passes before opening a pull request, and add to it when you add
behaviour. If you have a GPU, say in the pull request that the GPU steps ran.

## Adding an operation

1. **Describe it** in `tools/gen_wrappers.py` with one `op(...)` entry: argument lists for the
   Fortran and (if it exists) CBLAS interface, which integers are sizes, which pointers are
   operand buffers and whether they are read or written, and a flop estimate. This alone makes the
   operation measurable in all precisions and symbol naming schemes.
2. **To offload it**, add a `gpu="..."` mapping in that entry, a handler in `src/cuda_backend.c`
   (follow `do_potrf` for the pattern: validate, copy in, compute, check `info`, copy out last),
   an entry in `enum ofl_op` and `ofl_op_names`, and the output it should be verified on in
   `verify_setup` in `src/policy.c`.
3. **Calibrate it**: add a case to `src/calibrate.c` so profiles contain its speed curves.
4. **Test it**: add a case to `tests/test_gpu_ops.c` that checks the result against an independent
   reference (a residual or a reconstruction, not a comparison with another library).
5. **List it** in `GPU_OPS` in `tools/ofl_report.py`.

## Code style

- C11 for the library, Python 3 for the tools. No new dependencies without discussion.
- Match the surrounding code: 4-space indentation, lines up to about 120 characters, comments that
  explain *why* something is done.
- The preload library runs inside other people's programs. It must never change a result
  silently, never write to an output buffer before the GPU work has succeeded, and never crash the
  host program when the GPU is missing or fails.
- Keep `libgpuoffload.so` free of CUDA dependencies; everything GPU-specific belongs in the backend.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change with tests.
3. Run `./build.sh test`.
4. Open a pull request that says what changed, why, and how you tested it (including the machine,
   if performance is involved).

Small, focused pull requests are reviewed fastest. For anything large, open an issue first so the
approach can be agreed before you invest the work.

## Reporting measurements

Performance numbers are only useful with their context. Please include:

- CPU model, GPU model, driver and CUDA versions, operating system (and whether it is WSL2);
- the CPU math library and its thread count (`OPENBLAS_NUM_THREADS`);
- whether the machine was otherwise idle;
- the exact command, and medians of at least three runs.

## Licence of contributions

gpu-offload is dual-licensed: the [GNU AGPL v3](LICENSE) for everyone, and a commercial licence for
those who need other terms (see [LICENSING.md](LICENSING.md)). For both to remain possible, every
contributor agrees to the [Contributor Licence Agreement](CLA.md). In short: you keep the copyright
in your work, and you allow the project owner to publish it under the AGPL and to include it in
commercially licensed versions.

You agree by ticking the box in the pull-request template. Machine reports and bug reports posted
as issues do not need it.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
