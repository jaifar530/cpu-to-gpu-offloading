// SPDX-License-Identifier: AGPL-3.0-only
// Interface between the preload library and a GPU backend.
// The backend is a separate shared object (libgpuoffload-cuda.so) that is only loaded when the
// first call is about to be offloaded, so processes that never offload never touch CUDA.
#ifndef OFL_BACKEND_H
#define OFL_BACKEND_H

#include "ofl.h"

#define OFL_BACKEND_ABI 1
#define OFL_BACKEND_ENTRY "ofl_cuda_backend"

typedef struct {
    double in_s, compute_s, out_s;
} ofl_gpu_times;

struct ofl_backend {
    int abi;
    // Creates the GPU context and library handles. Returns 0 on success.
    int (*init)(void);
    // Runs one call on the GPU. Returns 0 on success, otherwise an enum ofl_reason; in that case
    // nothing has been written to the caller's output buffers and the CPU can run the call.
    int (*run)(const ofl_call *c, ofl_gpu_times *t);
    int (*mem_info)(uint64_t *free_bytes, uint64_t *total_bytes);
    // GPU utilisation in percent across all programs, or -1 if unknown.
    int (*utilization)(void);
    const char *(*device_name)(void);
};

typedef const struct ofl_backend *(*ofl_backend_entry_fn)(void);

#endif
