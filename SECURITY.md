# Security policy

gpu-offload is a library that is loaded into other programs with `LD_PRELOAD`, so mistakes in it
can affect the programs it is loaded into. Reports of security problems are taken seriously.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's private reporting:
on the repository page choose **Security**, then **Report a vulnerability**
(https://github.com/jaifar530/cpu-to-gpu-offloading/security/advisories/new).

Include what you observed, how to reproduce it, and the version or commit. You will get a reply
through the advisory.

## Supported versions

The project is at version 0.1. Fixes are made on the `main` branch and included in the next
release.

## Scope and expectations

- gpu-offload is opt-in: it only affects processes started with `gpu-offload run`, with
  `LD_PRELOAD` set by the user, or from a shell where `gpu-offload env` was applied.
- It reads its configuration from `~/.config/gpu-offload/`, `/etc/gpu-offload/` and `GPU_OFFLOAD_*`
  environment variables, and can be told through `GPU_OFFLOAD_BACKEND` to load a different backend
  library. Treat those inputs with the same trust as `LD_PRELOAD` itself: do not use the tool with
  privileged or set-uid programs.
- Trace files record addresses, sizes and timings of math-library calls, not the data itself.
