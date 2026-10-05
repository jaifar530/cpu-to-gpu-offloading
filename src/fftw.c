// SPDX-License-Identifier: AGPL-3.0-only
// FFTW3 interception. Transform sizes are only known when a plan is created, so the planner
// wrappers remember each plan's geometry and the execute wrappers look it up.
#include "ofl.h"
#include "ofl_gen.h"

#include <math.h>
#include <pthread.h>
#include <string.h>

enum { K_C2C, K_R2C, K_C2R, K_R2R };

typedef struct {
    void *plan, *in, *out;
    uint64_t in_bytes, out_bytes;
    int64_t dims[4];  // first three transform dimensions, then the number of transforms
    double flops;
} plan_info;

#define NSLOT 8192  // power of two
#define TOMBSTONE ((void *)1)
static plan_info g_slots[NSLOT];
static pthread_mutex_t g_plan_lock = PTHREAD_MUTEX_INITIALIZER;

static unsigned slot_of(const void *plan) {
    return (unsigned)(((uintptr_t)plan >> 4) * 2654435761u) & (NSLOT - 1);
}

static void plan_put(const plan_info *pi) {
    pthread_mutex_lock(&g_plan_lock);
    unsigned i = slot_of(pi->plan);
    int target = -1;
    for (unsigned probe = 0; probe < NSLOT; probe++, i = (i + 1) & (NSLOT - 1)) {
        if (g_slots[i].plan == pi->plan) {
            target = (int)i;
            break;
        }
        if (g_slots[i].plan == TOMBSTONE) {
            if (target < 0) target = (int)i;
        } else if (!g_slots[i].plan) {
            if (target < 0) target = (int)i;
            break;
        }
    }
    if (target >= 0) g_slots[target] = *pi;  // a full table just loses the geometry of new plans
    pthread_mutex_unlock(&g_plan_lock);
}

// Returns the slot index of a plan, or -1. Caller holds g_plan_lock.
static int plan_find(const void *plan) {
    unsigned i = slot_of(plan);
    for (unsigned probe = 0; probe < NSLOT && g_slots[i].plan; probe++, i = (i + 1) & (NSLOT - 1))
        if (g_slots[i].plan == plan) return (int)i;
    return -1;
}

static void plan_del(const void *plan) {
    pthread_mutex_lock(&g_plan_lock);
    int i = plan_find(plan);
    if (i >= 0) g_slots[i].plan = TOMBSTONE;
    pthread_mutex_unlock(&g_plan_lock);
}

static void plan_register(void *plan, int kind, int rank, const int64_t *n, int64_t howmany, void *in, void *out,
                          unsigned real_bytes) {
    if (!plan || rank < 1) return;
    plan_info pi = {.plan = plan, .in = in, .out = out};
    double points = 1;
    for (int i = 0; i < rank; i++) {
        points *= (double)n[i];
        if (i < 3) pi.dims[i] = n[i];
    }
    if (howmany < 1) howmany = 1;
    pi.dims[3] = howmany;
    // A real-to-complex transform stores only n/2+1 complex values along its last dimension.
    double last = (double)n[rank - 1];
    double half = last > 0 ? points / last * (floor(last / 2) + 1) : 0;
    double in_elems, out_elems;  // in units of one real number
    switch (kind) {
    case K_C2C: in_elems = out_elems = 2 * points; break;
    case K_R2C: in_elems = points; out_elems = 2 * half; break;
    case K_C2R: in_elems = 2 * half; out_elems = points; break;
    default: in_elems = out_elems = points; break;
    }
    pi.in_bytes = (uint64_t)(in_elems * (double)howmany) * real_bytes;
    pi.out_bytes = (uint64_t)(out_elems * (double)howmany) * real_bytes;
    pi.flops = (kind == K_C2C ? 5.0 : 2.5) * points * log2(points > 2 ? points : 2) * (double)howmany;
    plan_put(&pi);
}

static void reg_n(void *plan, int kind, int rank, int n0, int n1, int n2, void *in, void *out, unsigned rb) {
    int64_t n[3] = {n0, n1, n2};
    plan_register(plan, kind, rank, n, 1, in, out, rb);
}

static void reg_v(void *plan, int kind, int rank, const int *nv, int howmany, void *in, void *out, unsigned rb) {
    int64_t n[8];
    if (rank < 1 || rank > 8 || !nv) return;
    for (int i = 0; i < rank; i++) n[i] = nv[i];
    plan_register(plan, kind, rank, n, howmany, in, out, rb);
}

typedef struct {
    int n, is, os;
} iodim;

static void reg_guru(void *plan, int kind, int rank, const iodim *dims, int hrank, const iodim *hdims, void *in,
                     void *out, unsigned rb) {
    int64_t n[8], howmany = 1;
    if (rank < 1 || rank > 8 || !dims) return;
    for (int i = 0; i < rank; i++) n[i] = dims[i].n;
    for (int i = 0; i < hrank && hdims; i++) howmany *= hdims[i].n;
    plan_register(plan, kind, rank, n, howmany, in, out, rb);
}

// Fills dims and flops of a planner record from the plan it produced.
static void plan_fill(const void *plan, ofl_rec *r) {
    pthread_mutex_lock(&g_plan_lock);
    int i = plan ? plan_find(plan) : -1;
    if (i >= 0) memcpy(r->dims, g_slots[i].dims, sizeof r->dims);
    pthread_mutex_unlock(&g_plan_lock);
}

// Fills an execute record. `in` / `out` override the planned arrays (new-array execute functions).
static void plan_describe(const void *plan, void *in, void *out, ofl_rec *r) {
    pthread_mutex_lock(&g_plan_lock);
    int i = plan_find(plan);
    if (i >= 0) {
        const plan_info *pi = &g_slots[i];
        if (!in) in = pi->in;
        if (!out) out = pi->out;
        memcpy(r->dims, pi->dims, sizeof r->dims);
        r->flops = pi->flops;
        r->buf_ptr[0] = (uint64_t)(uintptr_t)in;
        r->buf_bytes[0] = pi->in_bytes;
        if (in == out) {
            r->buf_role[0] = OFL_ROLE_RW;
        } else {
            r->buf_role[0] = OFL_ROLE_R;
            r->buf_ptr[1] = (uint64_t)(uintptr_t)out;
            r->buf_bytes[1] = pi->out_bytes;
            r->buf_role[1] = OFL_ROLE_W;
        }
    }
    pthread_mutex_unlock(&g_plan_lock);
}

#define STR_(x) #x
#define STR(x) STR_(x)
#define FXS(name) STR(FX(name))

#define PLANNER(name, PARAMS, ARGS, REGISTER)                                                                   \
    OFL_EXPORT void *FX(name) PARAMS {                                                                       \
        typedef void *(*fn_t) PARAMS;                                                                           \
        static fn_t real;                                                                                       \
        if (__builtin_expect(!real, 0))                                                                         \
            real = (fn_t)ofl_resolve(FXS(name), (void *)FX(name), __builtin_return_address(0));              \
        if (!ofl_tracing || ofl_bypass) return real ARGS;                                                                   \
        ofl_rec r = {0};                                                                                     \
        ofl_begin(&r, FN_PLAN, __builtin_return_address(0));                                                 \
        void *plan = real ARGS;                                                                                 \
        REGISTER;                                                                                               \
        plan_fill(plan, &r);                                                                                    \
        ofl_end(&r);                                                                                         \
        return plan;                                                                                            \
    }

#define EXEC(name, PARAMS, ARGS, IN, OUT)                                                                       \
    OFL_EXPORT void FX(name) PARAMS {                                                                        \
        typedef void (*fn_t) PARAMS;                                                                            \
        static fn_t real;                                                                                       \
        if (__builtin_expect(!real, 0))                                                                         \
            real = (fn_t)ofl_resolve(FXS(name), (void *)FX(name), __builtin_return_address(0));              \
        if (!ofl_tracing || ofl_bypass) {                                                                                   \
            real ARGS;                                                                                          \
            return;                                                                                             \
        }                                                                                                       \
        ofl_rec r = {0};                                                                                     \
        plan_describe(plan, IN, OUT, &r);                                                                       \
        ofl_begin(&r, FN_EXEC, __builtin_return_address(0));                                                 \
        real ARGS;                                                                                              \
        ofl_end(&r);                                                                                         \
    }

// Double precision: fftw_*
#define FX(name) fftw_##name
#define RB 8
#define FN_PLAN OFL_FN_FFTW_PLAN_D
#define FN_EXEC OFL_FN_FFTW_EXEC_D
#include "fftw.inc"
#undef FX
#undef RB
#undef FN_PLAN
#undef FN_EXEC

// Single precision: fftwf_*
#define FX(name) fftwf_##name
#define RB 4
#define FN_PLAN OFL_FN_FFTW_PLAN_S
#define FN_EXEC OFL_FN_FFTW_EXEC_S
#include "fftw.inc"
