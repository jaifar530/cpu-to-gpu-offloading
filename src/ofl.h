// SPDX-License-Identifier: AGPL-3.0-only
// gpu-offload: an LD_PRELOAD library that sits in front of BLAS / LAPACK / FFTW.
// It can measure what an unmodified program asks those libraries to do, and it can run the
// heavy calls on the GPU when a per-machine profile predicts that this is faster.
#ifndef OFL_H
#define OFL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define OFL_HIDDEN __attribute__((visibility("hidden")))
#define OFL_EXPORT __attribute__((visibility("default")))

#define OFL_MAGIC "GPUOFL1"
#define OFL_VERSION 2

enum { OFL_ROLE_R = 1, OFL_ROLE_W = 2, OFL_ROLE_RW = 3 };
enum {
    OFL_FLAG_WSQUERY = 1,   // LAPACK workspace-size query (lwork == -1), does no real work
    OFL_FLAG_VERIFIED = 2,  // verify mode: the GPU result matched the CPU result
    OFL_FLAG_MISMATCH = 4,  // verify mode: the GPU result differed beyond tolerance
    OFL_FLAG_FAILED = 8,    // LAPACK reported info != 0: the timing says nothing about the work
};
enum { OFL_DEV_CPU = 0, OFL_DEV_GPU = 1 };

// Why a call that has a GPU implementation ran on the CPU. tools/ofl_report.py mirrors the names.
enum ofl_reason {
    OFL_WHY_NONE = 0,     // ran on the GPU, or the function has no GPU implementation
    OFL_WHY_SLOWER,       // predicted gain below the threshold of the aggressiveness level
    OFL_WHY_WARMUP,       // GPU start-up cost not yet justified by the work seen so far
    OFL_WHY_VRAM,         // not enough free GPU memory
    OFL_WHY_BUSY,         // GPU busy with other programs
    OFL_WHY_ARGS,         // argument combination the GPU path does not handle
    OFL_WHY_ERROR,        // GPU path failed, fell back to the CPU
    OFL_WHY_NO_PROFILE,   // machine not calibrated
    OFL_WHY_NO_GPU,       // GPU backend unavailable in this process
    OFL_WHY_FORCED,       // device forced by configuration
    OFL_WHY_PROBE,        // sent to the CPU now and then to keep its measured speed current
    OFL_WHY_COUNT
};

// One intercepted call. The Python reader in tools/ofl_report.py mirrors this layout.
typedef struct {
    uint64_t t_ns;          // start time, ns since the library started in this process
    uint64_t dur_ns;        // wall time of the call (on whichever device ran it)
    uint64_t callsite;      // return address of the call (resolved to module+offset offline)
    uint64_t buf_ptr[3];    // main operand buffers
    uint64_t buf_bytes[3];  // their sizes (contiguous-storage estimate)
    double flops;           // rough floating-point operation count
    int64_t dims[4];        // operation-specific sizes (m, n, k, ...)
    uint32_t tid;
    uint16_t func;          // index into ofl_funcs[]
    uint8_t depth;          // 0 = called by the application, >0 = nested in another intercepted call
    uint8_t flags;
    uint32_t sig_pre[3];    // sampled content fingerprint before the call (buffers that are read)
    uint32_t sig_post[3];   // sampled content fingerprint after the call (buffers that are written)
    uint8_t buf_role[3];
    uint8_t device;         // OFL_DEV_*
    uint8_t reason;         // enum ofl_reason
    uint8_t variant;        // eigen / SVD calls: 1 if vectors are computed
    uint16_t curve;         // 1 + index of the cost curve used for the prediction, 0 = none
    float pred_cpu_s;       // predicted CPU time, 0 = no prediction
    float pred_gpu_s;       // predicted GPU time including transfers
    float gpu_in_s;         // GPU runs: host-to-device copy time
    float gpu_out_s;        // GPU runs: device-to-host copy time
} ofl_rec;
_Static_assert(sizeof(ofl_rec) == 168, "ofl_rec layout changed: update tools/ofl_report.py");

struct ofl_func {
    const char *name;     // exported symbol, e.g. "dgemm_"
    const char *fam;      // blas1 | blas2 | blas3 | lapack | fftw
    const char *op;       // operation without precision, e.g. "gemm"
    const char *iface;    // fortran | cblas | c
    const char *variant;  // symbol naming scheme, e.g. "lp64"
    char prec;            // s, d, c, z
};
OFL_HIDDEN extern const struct ofl_func ofl_funcs[];

// ---------------------------------------------------------------------------------------------
// GPU-capable operations
// ---------------------------------------------------------------------------------------------

enum ofl_op {
    OFL_OP_NONE = 0,
    OFL_OP_GEMM,
    OFL_OP_SYRK,
    OFL_OP_HERK,
    OFL_OP_TRSM,
    OFL_OP_GETRF,
    OFL_OP_GESV,
    OFL_OP_POTRF,
    OFL_OP_POSV,
    OFL_OP_SYEVD,  // symmetric / Hermitian eigenproblem (syev, syevd, heev, heevd)
    OFL_OP_GESVD,  // singular value decomposition (gesvd, gesdd)
    OFL_OP_COUNT
};
OFL_HIDDEN extern const char *const ofl_op_names[OFL_OP_COUNT];

// A call in a form the GPU backend understands, independent of interface and symbol naming.
// Matrices are column-major unless row_major is set (CBLAS row-major calls).
typedef struct {
    int op;
    char prec;          // s, d, c, z
    uint8_t variant;    // SYEVD / GESVD: 1 if vectors are computed, 0 for values only
    uint8_t row_major;
    uint8_t int_bytes;  // size of the integers behind ipiv / info (4 or 8)
    char trans_a, trans_b, side, uplo, diag;  // BLAS-style letters: N T C, L R, U L, N U
    char job_a, job_b;                        // jobz or jobu, jobvt
    int64_t m, n, k;                          // k is also the number of right-hand sides
    const void *alpha, *beta;
    const void *a, *b, *c;                    // input matrices
    int64_t lda, ldb, ldc;
    void *a_out, *b_out, *c_out;              // where results go; NULL = result not wanted
    void *w_out;                              // eigenvalues or singular values (real numbers)
    void *u_out, *vt_out;
    int64_t ldu, ldvt;
    const void *ipiv;                         // pivots, as input
    void *ipiv_out, *info_out;
    void *verify;                             // verify mode bookkeeping, owned by policy.c
} ofl_call;

// ---------------------------------------------------------------------------------------------
// Internal API
// ---------------------------------------------------------------------------------------------

OFL_HIDDEN extern int ofl_active;      // recording and / or offloading enabled in this process
OFL_HIDDEN extern int ofl_offloading;  // GPU dispatch enabled
OFL_HIDDEN extern int ofl_tracing;     // per-call trace file enabled
OFL_HIDDEN extern __thread int ofl_bypass;  // set while the GPU backend runs: wrappers pass straight through
OFL_HIDDEN extern double ofl_min_flops;     // calls smaller than this are never worth a GPU decision

// Finds the library function a wrapper stands in front of. `self` is the wrapper, `caller` is the
// return address of the intercepted call.
OFL_HIDDEN void *ofl_resolve(const char *sym, void *self, void *caller);

// CPU path: ofl_begin immediately before and ofl_end immediately after the real function.
// ofl_end is ofl_stop (stop the clock) followed by ofl_commit (store the record).
OFL_HIDDEN void ofl_begin(ofl_rec *r, unsigned func, void *callsite);
OFL_HIDDEN void ofl_stop(ofl_rec *r);
OFL_HIDDEN void ofl_commit(ofl_rec *r);
OFL_HIDDEN void ofl_end(ofl_rec *r);
OFL_HIDDEN uint64_t ofl_now_ns(void);

// Decides where a GPU-capable call runs. Returns 1 if the GPU produced the result (the record is
// already committed), 2 if the caller must run the CPU function and then call ofl_verify, and
// 0 if the caller must run the CPU function (r->reason says why).
OFL_HIDDEN int ofl_dispatch(ofl_rec *r, ofl_call *c, unsigned func, void *callsite);
OFL_HIDDEN void ofl_verify(ofl_rec *r, ofl_call *c);
OFL_HIDDEN void ofl_policy_init(void);
OFL_HIDDEN void ofl_policy_fork_child(void);
OFL_HIDDEN void ofl_policy_observe_cpu(const ofl_rec *r);
OFL_HIDDEN void ofl_policy_summary(void);
OFL_HIDDEN void ofl_policy_meta(FILE *f);  // writes the policy fields of the trace meta file

static inline uint64_t ofl_nb(int64_t elems, unsigned elem_bytes) {
    return elems > 0 ? (uint64_t)elems * elem_bytes : 0;
}

// CBLAS enum value -> the BLAS letter it stands for.
static inline char ofl_cblas_letter(int e) {
    switch (e) {
    case 111: return 'N';  // CblasNoTrans
    case 112: return 'T';  // CblasTrans
    case 113: return 'C';  // CblasConjTrans
    case 121: return 'U';  // CblasUpper
    case 122: return 'L';  // CblasLower
    case 131: return 'N';  // CblasNonUnit
    case 132: return 'U';  // CblasUnit
    case 141: return 'L';  // CblasLeft
    case 142: return 'R';  // CblasRight
    default: return 0;
    }
}

#endif
