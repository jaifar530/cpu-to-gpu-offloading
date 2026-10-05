#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Builds gpu-offload into ~/build/gpu-offload (override with GPU_OFFLOAD_BUILD).
# usage: build.sh [test] [package]
#   test     also run the test suite
#   package  also build the .deb package
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
build="${GPU_OFFLOAD_BUILD:-$HOME/build/gpu-offload}"

cmake -S "$src" -B "$build" -G Ninja >/dev/null
cmake --build "$build"
echo "built in $build (run $build/gpu-offload)"

for arg in "$@"; do
    case "$arg" in
    test) ctest --test-dir "$build" --output-on-failure -V | grep '^1: ' | cut -c4- ;;
    package) (cd "$build" && cpack >/dev/null && ls -1 "$build"/*.deb) ;;
    esac
done
