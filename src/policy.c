// SPDX-License-Identifier: AGPL-3.0-only
// Decides, call by call, whether a GPU-capable operation runs on the GPU or stays on the CPU.
//
// The decision compares two predictions taken from the machine profile written by
// `gpu-offload calibrate`:
//     T_cpu = flops / rate_cpu(op, precision, flops)
//     T_gpu = copy-in + flops / rate_gpu(op, precision, flops) + copy-out
// and offloads when T_cpu >= min_speedup(level) * T_gpu, the GPU has room, and no other program
// is keeping it busy. Predictions are corrected while the program runs from the times actually
// observed, so a slower CPU library or a contended GPU shifts the decision without recalibrating.
#define _GNU_SOURCE
#include "backend.h"

#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const ofl_op_names[OFL_OP_COUNT] = {"none",  "gemm", "syrk",  "herk",  "trsm", "getrf",
                                                "gesv",  "potrf", "posv", "syevd", "gesvd"};
static const char *const reason_text[OFL_WHY_COUNT] = {
    "",
    "predicted gain too small",
    "GPU start-up not yet worth it",
    "not enough free GPU memory",
    "GPU busy with other programs",
    "arguments the GPU path does not handle",
    "GPU error, fell back to CPU",
    "machine not calibrated",
    "GPU unavailable",
    "forced to CPU",
    "re-checking the CPU's speed",
};

#define MAX_PTS 24
typedef struct {
    int n;
    double lf[MAX_PTS], lr[MAX_PTS];  // log(flops) -> log(flops per second)
} curve;
typedef struct {
    curve cpu, gpu;
    double corr_cpu, corr_gpu;  // observed / predicted, learned while the program runs
    unsigned since_cpu;         // GPU runs since the CPU's speed was last observed
} model;

// Calibration measures the CPU once, on a machine that may have been busy. Every PROBE_EVERY-th
// call that would go to the GPU runs on the CPU instead, so a CPU that is faster than the profile
// says is noticed and wins the work back.
#define PROBE_EVERY 16

#define NMODELS (OFL_OP_COUNT * 2 * 4)
static model g_models[NMODELS];
static const unsigned ESZ[4] = {4, 8, 8, 16}, RSZ[4] = {4, 8, 4, 8};

// A GPU call costs tens of microseconds before any work is done; a CPU finishes this many flops
// sooner than that, so smaller calls skip the decision entirely (about an 80 x 80 matrix product).
#define MIN_FLOPS 1e6
double ofl_min_flops;

enum { FORCE_NONE, FORCE_CPU, FORCE_GPU };
static int g_level = 5, g_force, g_verify, g_summary, g_have_profile;
static double g_min_speedup, g_util_limit, g_vram_reserve, g_warm_factor;
static double g_init_s = 1.0, g_h2d_fixed, g_h2d_bps = 1e10, g_d2h_fixed, g_d2h_bps = 1e10;

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_gpu_lock = PTHREAD_MUTEX_INITIALIZER;   // one GPU call at a time per process
static pthread_mutex_t g_stat_lock = PTHREAD_MUTEX_INITIALIZER;
static const struct ofl_backend *g_backend;
static int g_backend_state;  // 0 not loaded, 1 loaded, 2 initialised, -1 unusable
static double g_foregone_s, g_init_measured_s;
static uint64_t g_last_gpu_ns, g_util_checked_ns, g_mem_checked_ns, g_mem_free, g_mem_total;
static int g_util = -1;

static struct {
    uint64_t gpu_calls, cpu_calls[OFL_WHY_COUNT], verified, mismatched, unverifiable;
    double gpu_s, saved_s, max_rel;
} g_stats;

static int prec_index(char p) {
    return p == 's' ? 0 : p == 'd' ? 1 : p == 'c' ? 2 : 3;
}

static int model_index(int op, int variant, char prec) {
    return (op * 2 + (variant ? 1 : 0)) * 4 + prec_index(prec);
}

// ---------------------------------------------------------------------------------------------
// Configuration and machine profile
// ---------------------------------------------------------------------------------------------

static void set_level(int level) {
    static const double speedup[11] = {0, 3.0, 2.5, 2.0, 1.6, 1.3, 1.2, 1.1, 1.05, 1.0, 0.9};
    if (level < 1) level = 1;
    if (level > 10) level = 10;
    g_level = level;
    g_min_speedup = speedup[level];              // required predicted speedup
    g_util_limit = 30 + (level - 1) * 65.0 / 9;  // back off when other programs load the GPU above this (%)
    g_vram_reserve = 0.50 - (level - 1) * 0.05;  // fraction of GPU memory always left free
    g_warm_factor = 3.0 - (level - 1) * 0.25;    // start the GPU once the missed savings reach this x start-up cost
}

static int read_level_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char key[64];
    int value, found = 0;
    while (fscanf(f, "%63s %d", key, &value) == 2)
        if (!strcmp(key, "level")) found = value;
    fclose(f);
    return found;
}

static int load_profile(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char word[64];
    int curves = 0;
    while (fscanf(f, "%63s", word) == 1) {
        if (!strcmp(word, "param")) {
            char name[64];
            double v;
            if (fscanf(f, "%63s %lf", name, &v) != 2) break;
            if (!strcmp(name, "init_s")) g_init_s = v;
            else if (!strcmp(name, "h2d_fixed_s")) g_h2d_fixed = v;
            else if (!strcmp(name, "h2d_bps")) g_h2d_bps = v;
            else if (!strcmp(name, "d2h_fixed_s")) g_d2h_fixed = v;
            else if (!strcmp(name, "d2h_bps")) g_d2h_bps = v;
        } else if (!strcmp(word, "curve")) {
            char opname[32], prec[4], dev[8];
            int variant, n, op = 0;
            if (fscanf(f, "%31s %d %3s %7s %d", opname, &variant, prec, dev, &n) != 5 || n < 1 || n > MAX_PTS) break;
            for (int i = 1; i < OFL_OP_COUNT; i++)
                if (!strcmp(opname, ofl_op_names[i])) op = i;
            curve cv = {.n = n};
            int ok = 1;
            for (int i = 0; i < n; i++) {
                double flops, rate;
                if (fscanf(f, "%lf:%lf", &flops, &rate) != 2 || flops <= 0 || rate <= 0) {
                    ok = 0;
                    break;
                }
                cv.lf[i] = log(flops);
                cv.lr[i] = log(rate);
            }
            if (!ok) break;
            if (op) {
                model *m = &g_models[model_index(op, variant, prec[0])];
                if (!strcmp(dev, "cpu")) m->cpu = cv;
                else m->gpu = cv;
                curves++;
            }
        } else if (fscanf(f, "%*[^\n]") == EOF) {
            break;
        }
    }
    fclose(f);
    return curves > 0;
}

static void policy_load(void) {
    char path[PATH_MAX];
    const char *home = getenv("HOME"), *e;

    for (int i = 0; i < NMODELS; i++) {
        g_models[i].corr_cpu = g_models[i].corr_gpu = 1.0;
        g_models[i].since_cpu = PROBE_EVERY - 1;  // check the CPU once before trusting the profile
    }

    int level = 0;
    if ((e = getenv("GPU_OFFLOAD_LEVEL")) && atoi(e) > 0) level = atoi(e);
    if (!level && home) {
        snprintf(path, sizeof path, "%s/.config/gpu-offload/config", home);
        level = read_level_file(path);
    }
    if (!level) level = read_level_file("/etc/gpu-offload/config");
    set_level(level ? level : 5);

    if ((e = getenv("GPU_OFFLOAD_PROFILE")) && e[0]) {
        g_have_profile = load_profile(e);
    } else {
        if (home) {
            snprintf(path, sizeof path, "%s/.config/gpu-offload/profile", home);
            g_have_profile = load_profile(path);
        }
        if (!g_have_profile) g_have_profile = load_profile("/etc/gpu-offload/profile");
    }

    if ((e = getenv("GPU_OFFLOAD_FORCE"))) g_force = !strcmp(e, "gpu") ? FORCE_GPU : !strcmp(e, "cpu") ? FORCE_CPU : 0;
    g_verify = (e = getenv("GPU_OFFLOAD_VERIFY")) && atoi(e) > 0;
    g_summary = (e = getenv("GPU_OFFLOAD_SUMMARY")) && atoi(e) > 0;
    if (g_force == FORCE_NONE && !g_verify) ofl_min_flops = MIN_FLOPS;  // forced and verify runs take every call
    if ((e = getenv("GPU_OFFLOAD_MIN_FLOPS"))) ofl_min_flops = atof(e);
}

void ofl_policy_init(void) {
    pthread_once(&g_once, policy_load);
}

// A CUDA context does not survive fork(): the child keeps running on the CPU.
void ofl_policy_fork_child(void) {
    pthread_mutex_init(&g_gpu_lock, NULL);
    pthread_mutex_init(&g_stat_lock, NULL);
    if (g_backend_state == 2) g_backend_state = -1;
    memset(&g_stats, 0, sizeof g_stats);
}

// ---------------------------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------------------------

// Piecewise-linear interpolation in log-log space, flat outside the calibrated range.
static double curve_time(const curve *cv, double flops) {
    if (flops <= 0 || cv->n == 0) return 0;
    double x = log(flops), y;
    if (x <= cv->lf[0]) {
        y = cv->lr[0];
    } else if (x >= cv->lf[cv->n - 1]) {
        y = cv->lr[cv->n - 1];
    } else {
        int i = 1;
        while (x > cv->lf[i]) i++;
        double t = (x - cv->lf[i - 1]) / (cv->lf[i] - cv->lf[i - 1]);
        y = cv->lr[i - 1] + t * (cv->lr[i] - cv->lr[i - 1]);
    }
    return flops / exp(y);
}

// Moves a correction factor halfway (in ratio terms) towards the latest observed / predicted ratio.
static double learn(double corr, double ratio) {
    double v = sqrt(corr * ratio);
    return v < 0.02 ? 0.02 : v > 50 ? 50 : v;
}

void ofl_policy_observe_cpu(const ofl_rec *r) {
    if (!r->curve || r->curve > NMODELS || (r->flags & OFL_FLAG_WSQUERY)) return;
    model *m = &g_models[r->curve - 1];
    pthread_mutex_lock(&g_stat_lock);
    double raw = r->pred_cpu_s / m->corr_cpu;
    if (raw > 0) m->corr_cpu = learn(m->corr_cpu, r->dur_ns * 1e-9 / raw);
    m->since_cpu = 0;
    pthread_mutex_unlock(&g_stat_lock);
}

// ---------------------------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------------------------

static void backend_load(void) {
    char path[PATH_MAX];
    const char *e = getenv("GPU_OFFLOAD_BACKEND");
    Dl_info di;
    if (e && e[0]) {
        snprintf(path, sizeof path, "%s", e);
    } else if (dladdr((void *)backend_load, &di) && di.dli_fname) {
        snprintf(path, sizeof path, "%s", di.dli_fname);
        char *slash = strrchr(path, '/');
        snprintf(slash ? slash + 1 : path, sizeof path - (slash ? (size_t)(slash + 1 - path) : 0), "libgpuoffload-cuda.so");
    } else {
        g_backend_state = -1;
        return;
    }
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    ofl_backend_entry_fn entry = h ? (ofl_backend_entry_fn)dlsym(h, OFL_BACKEND_ENTRY) : NULL;
    g_backend = entry ? entry() : NULL;
    g_backend_state = g_backend && g_backend->abi == OFL_BACKEND_ABI ? 1 : -1;
}

// Returns 1 when the backend is loaded (not necessarily initialised). Caller holds g_gpu_lock.
static int backend_loaded(void) {
    if (g_backend_state == 0) {
        uint64_t t0 = ofl_now_ns();
        ofl_bypass++;
        backend_load();
        ofl_bypass--;
        g_init_measured_s += (ofl_now_ns() - t0) * 1e-9;  // loading the CUDA libraries is part of the start-up cost
    }
    return g_backend_state > 0;
}

// Caller holds g_gpu_lock.
static int backend_start(void) {
    if (g_backend_state == 2) return 1;
    if (g_backend_state != 1) return 0;
    uint64_t t0 = ofl_now_ns();
    ofl_bypass++;
    int rc = g_backend->init();
    ofl_bypass--;
    g_init_measured_s += (ofl_now_ns() - t0) * 1e-9;
    g_backend_state = rc == 0 ? 2 : -1;
    return rc == 0;
}

// True when programs other than this one are loading the GPU beyond the level's limit.
// While this process is itself using the GPU the utilisation figure says nothing about others.
static int others_busy(void) {
    uint64_t now = ofl_now_ns();
    if (g_last_gpu_ns && now - g_last_gpu_ns < 1000000000ull) return 0;
    if (!g_util_checked_ns || now - g_util_checked_ns > 1000000000ull) {
        g_util = g_backend->utilization();
        g_util_checked_ns = now;
    }
    return g_util >= 0 && g_util > g_util_limit;
}

// ---------------------------------------------------------------------------------------------
// Verify mode: run the call on both devices and compare
// ---------------------------------------------------------------------------------------------

typedef struct {
    void *tmp[2];
    const void *cpu[2];
    int64_t rows[2], cols[2], ld[2];
    int nmat;
    void *w_tmp;
    const void *w_cpu;
    int64_t w_n;
    void *ipiv_tmp;
    const void *ipiv_cpu;
    int64_t ipiv_n;
    int64_t info_tmp;
    const void *info_cpu;
} vstate;

// Gives the GPU a private copy of one output matrix; returns the copy.
static void *shadow(vstate *v, void *cpu_out, int64_t rows, int64_t cols, int64_t ld, unsigned esz) {
    size_t bytes = (size_t)(ld * (cols - 1) + rows) * esz;
    void *tmp = malloc(bytes);
    if (!tmp) return NULL;
    memcpy(tmp, cpu_out, bytes);
    int i = v->nmat++;
    v->tmp[i] = tmp;
    v->cpu[i] = cpu_out;
    v->rows[i] = rows;
    v->cols[i] = cols;
    v->ld[i] = ld;
    return tmp;
}

static int verify_setup(ofl_call *c) {
    vstate *v = calloc(1, sizeof *v);
    if (!v) return 0;
    int p = prec_index(c->prec);
    unsigned esz = ESZ[p];
    int64_t mn = c->m < c->n ? c->m : c->n;
    // Output geometry in column-major terms: a row-major m x n array is a column-major n x m one.
    int64_t rows = c->row_major ? c->n : c->m, cols = c->row_major ? c->m : c->n;
    c->verify = v;
    v->info_cpu = c->info_out;
    if (c->info_out) c->info_out = &v->info_tmp;

    switch (c->op) {
    case OFL_OP_GEMM: c->c_out = shadow(v, c->c_out, rows, cols, c->ldc, esz); break;
    case OFL_OP_SYRK:
    case OFL_OP_HERK: c->c_out = shadow(v, c->c_out, c->n, c->n, c->ldc, esz); break;
    case OFL_OP_TRSM: c->b_out = shadow(v, c->b_out, rows, cols, c->ldb, esz); break;
    case OFL_OP_GETRF:
        c->a_out = shadow(v, c->a_out, c->m, c->n, c->lda, esz);
        v->ipiv_cpu = c->ipiv_out;
        v->ipiv_n = mn;
        c->ipiv_out = v->ipiv_tmp = calloc((size_t)mn, c->int_bytes);
        break;
    case OFL_OP_GESV:
        c->a_out = NULL;  // pivoting can differ legitimately; the solution is what is compared
        c->ipiv_out = NULL;
        c->b_out = shadow(v, c->b_out, c->n, c->k, c->ldb, esz);
        break;
    case OFL_OP_POTRF: c->a_out = shadow(v, c->a_out, c->n, c->n, c->lda, esz); break;
    case OFL_OP_POSV:
        c->a_out = shadow(v, c->a_out, c->n, c->n, c->lda, esz);
        c->b_out = shadow(v, c->b_out, c->n, c->k, c->ldb, esz);
        break;
    case OFL_OP_SYEVD:
    case OFL_OP_GESVD:  // vectors are only defined up to sign: compare the values
        c->a_out = c->u_out = c->vt_out = NULL;
        v->w_cpu = c->w_out;
        v->w_n = c->op == OFL_OP_SYEVD ? c->n : mn;
        c->w_out = v->w_tmp = calloc((size_t)v->w_n, RSZ[p]);
        break;
    }
    return 1;
}

static void verify_free(ofl_call *c) {
    vstate *v = c->verify;
    if (!v) return;
    free(v->tmp[0]);
    free(v->tmp[1]);
    free(v->w_tmp);
    free(v->ipiv_tmp);
    free(v);
    c->verify = NULL;
}

// Largest difference relative to the largest CPU value, over `count` real numbers.
static void diff_reals(const void *x, const void *y, size_t count, int is_double, double *maxdiff, double *maxabs) {
    for (size_t i = 0; i < count; i++) {
        double a = is_double ? ((const double *)x)[i] : ((const float *)x)[i];
        double b = is_double ? ((const double *)y)[i] : ((const float *)y)[i];
        double d = fabs(a - b);
        // NaN here means a non-finite value: fine if both sides agree on it, a mismatch otherwise.
        if (isnan(d)) d = (isnan(a) && isnan(b)) || a == b ? 0 : INFINITY;
        if (d > *maxdiff) *maxdiff = d;
        if (fabs(a) > *maxabs) *maxabs = fabs(a);
    }
}

static int64_t read_int(const void *p, int bytes) {
    return bytes == 8 ? *(const int64_t *)p : *(const int32_t *)p;
}

void ofl_verify(ofl_rec *r, ofl_call *c) {
    vstate *v = c->verify;
    if (!v) return;
    int p = prec_index(c->prec), is_double = RSZ[p] == 8;
    size_t comps = ESZ[p] / RSZ[p];
    double maxdiff = 0, maxabs = 0;
    int comparable = 1, same_status = 1;

    if (v->info_cpu) {
        int64_t cpu_info = read_int(v->info_cpu, c->int_bytes), gpu_info = read_int(&v->info_tmp, c->int_bytes);
        same_status = cpu_info == gpu_info;
        if (cpu_info != 0 || gpu_info != 0) comparable = 0;  // failed factorisation: only the status is comparable
    }
    if (comparable && v->ipiv_tmp && memcmp(v->ipiv_tmp, v->ipiv_cpu, (size_t)v->ipiv_n * c->int_bytes))
        comparable = 0, same_status = -1;  // a different but equally valid pivot order
    if (comparable) {
        for (int i = 0; i < v->nmat; i++)
            for (int64_t col = 0; col < v->cols[i]; col++) {
                size_t off = (size_t)(col * v->ld[i]) * ESZ[p];
                diff_reals((const char *)v->cpu[i] + off, (const char *)v->tmp[i] + off, (size_t)v->rows[i] * comps,
                           is_double, &maxdiff, &maxabs);
            }
        if (v->w_tmp) diff_reals(v->w_cpu, v->w_tmp, (size_t)v->w_n, is_double, &maxdiff, &maxabs);
    }

    double rel = maxabs > 0 ? maxdiff / maxabs : maxdiff;
    double tol = is_double ? 1e-8 : 2e-3;
    pthread_mutex_lock(&g_stat_lock);
    if (same_status < 0) {
        g_stats.unverifiable++;
    } else if (!same_status || (comparable && !(rel <= tol))) {
        g_stats.mismatched++;
        r->flags |= OFL_FLAG_MISMATCH;
        fprintf(stderr, "gpu-offload: VERIFY MISMATCH in %s (%c): relative difference %.3e\n", ofl_op_names[c->op],
                c->prec, rel);
    } else {
        g_stats.verified++;
        r->flags |= OFL_FLAG_VERIFIED;
    }
    if (comparable && rel > g_stats.max_rel) g_stats.max_rel = rel;
    pthread_mutex_unlock(&g_stat_lock);
    verify_free(c);
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

static int stay_on_cpu(ofl_rec *r, int why) {
    r->reason = (uint8_t)why;
    pthread_mutex_lock(&g_stat_lock);
    g_stats.cpu_calls[why]++;
    pthread_mutex_unlock(&g_stat_lock);
    return 0;
}

int ofl_dispatch(ofl_rec *r, ofl_call *c, unsigned func, void *callsite) {
    pthread_once(&g_once, policy_load);
    int mi = model_index(c->op, c->variant, c->prec);
    model *m = &g_models[mi];
    int decide = g_force == FORCE_NONE && !g_verify;  // false: the device is dictated, skip the cost model

    double in_bytes = 0, out_bytes = 0, need = 0;
    for (int i = 0; i < 3; i++) {
        if (r->buf_role[i] & OFL_ROLE_R) in_bytes += (double)r->buf_bytes[i];
        if (r->buf_role[i] & OFL_ROLE_W) out_bytes += (double)r->buf_bytes[i];
        need += (double)r->buf_bytes[i];
    }
    // Device memory: the operands plus solver workspace (eigen and SVD solvers need several copies).
    need = need * (c->op == OFL_OP_SYEVD || c->op == OFL_OP_GESVD ? 4.0 : 1.5) + (64 << 20);

    double pc = 0, pg = 0;
    if (m->cpu.n && m->gpu.n) {
        pc = curve_time(&m->cpu, r->flops) * m->corr_cpu;
        pg = (g_h2d_fixed + in_bytes / g_h2d_bps + g_d2h_fixed + out_bytes / g_d2h_bps + curve_time(&m->gpu, r->flops)) *
             m->corr_gpu;
        r->curve = (uint16_t)(mi + 1);
        r->pred_cpu_s = (float)pc;
        r->pred_gpu_s = (float)pg;
    }

    if (g_force == FORCE_CPU) return stay_on_cpu(r, OFL_WHY_FORCED);
    if (g_backend_state < 0) return stay_on_cpu(r, OFL_WHY_NO_GPU);
    if (c->m <= 0 || c->n <= 0 || c->m > INT_MAX || c->n > INT_MAX || c->k > INT_MAX)
        return stay_on_cpu(r, OFL_WHY_ARGS);
    if (decide) {
        if (!(pc > 0 && pg > 0)) return stay_on_cpu(r, OFL_WHY_NO_PROFILE);
        if (pc < g_min_speedup * pg) return stay_on_cpu(r, OFL_WHY_SLOWER);
        if (g_backend_state == 2 && ++m->since_cpu >= PROBE_EVERY) return stay_on_cpu(r, OFL_WHY_PROBE);
    }

    pthread_mutex_lock(&g_gpu_lock);
    int why = 0;
    // Loading and starting the GPU libraries costs about g_init_s. Wait until the savings missed so
    // far would have paid for it; a process that does little heavy math then never pays that cost.
    if (g_backend_state != 2 && decide && (g_foregone_s += pc - pg) < g_init_s * g_warm_factor) why = OFL_WHY_WARMUP;
    else if (!backend_loaded()) why = OFL_WHY_NO_GPU;
    else if (decide && others_busy()) why = OFL_WHY_BUSY;
    else if (!backend_start()) why = OFL_WHY_NO_GPU;
    if (!why && decide) {
        uint64_t now = ofl_now_ns();
        if (!g_mem_checked_ns || now - g_mem_checked_ns > 100000000ull) {
            if (g_backend->mem_info(&g_mem_free, &g_mem_total)) g_mem_free = g_mem_total = 0;
            g_mem_checked_ns = now;
        }
        if (g_mem_total && need > (double)g_mem_free - g_vram_reserve * (double)g_mem_total) why = OFL_WHY_VRAM;
    }
    if (!why && g_verify && !verify_setup(c)) why = OFL_WHY_ERROR;
    if (why) {
        pthread_mutex_unlock(&g_gpu_lock);
        return stay_on_cpu(r, why);
    }

    ofl_gpu_times t = {0, 0, 0};
    ofl_begin(r, func, callsite);
    ofl_bypass++;
    int rc = g_backend->run(c, &t);
    ofl_bypass--;
    ofl_stop(r);
    g_last_gpu_ns = ofl_now_ns();
    g_mem_checked_ns = 0;
    pthread_mutex_unlock(&g_gpu_lock);

    if (g_verify) {  // the application gets the CPU result; ofl_verify compares the two afterwards
        if (rc) verify_free(c);
        r->dur_ns = 0;
        return rc ? stay_on_cpu(r, rc) : 2;
    }
    if (rc) {
        r->dur_ns = 0;
        return stay_on_cpu(r, rc);
    }

    double took = r->dur_ns * 1e-9;
    r->device = OFL_DEV_GPU;
    if (c->info_out && read_int(c->info_out, c->int_bytes) != 0) r->flags |= OFL_FLAG_FAILED;
    r->gpu_in_s = (float)t.in_s;
    r->gpu_out_s = (float)t.out_s;
    pthread_mutex_lock(&g_stat_lock);
    g_stats.gpu_calls++;
    g_stats.gpu_s += took;
    if (pc > 0) g_stats.saved_s += pc - took;
    if (pg > 0) m->corr_gpu = learn(m->corr_gpu, took / (pg / m->corr_gpu));
    pthread_mutex_unlock(&g_stat_lock);
    ofl_commit(r);
    return 1;
}

// ---------------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------------

void ofl_policy_meta(FILE *f) {
    fprintf(f, "  \"level\": %d,\n  \"profile\": %s,\n  \"gpu\": \"%s\",\n  \"gpu_init_s\": %.6f,\n", g_level,
            g_have_profile ? "true" : "false", g_backend_state == 2 ? g_backend->device_name() : "", g_init_measured_s);
}

void ofl_policy_summary(void) {
    if (!g_summary) return;
    uint64_t cpu = 0;
    for (int i = 0; i < OFL_WHY_COUNT; i++) cpu += g_stats.cpu_calls[i];
    if (!cpu && !g_stats.gpu_calls) return;

    fprintf(stderr, "\ngpu-offload: level %d", g_level);
    if (g_backend_state == 2) fprintf(stderr, " on %s (GPU start-up %.2f s)", g_backend->device_name(), g_init_measured_s);
    fprintf(stderr, "\n  GPU-capable calls: %llu on the GPU (%.2f s), %llu on the CPU\n",
            (unsigned long long)g_stats.gpu_calls, g_stats.gpu_s, (unsigned long long)cpu);
    if (g_stats.gpu_calls && g_have_profile)
        fprintf(stderr, "  estimated time saved by offloading: %.2f s\n", g_stats.saved_s - g_init_measured_s);
    for (int i = 1; i < OFL_WHY_COUNT; i++)
        if (g_stats.cpu_calls[i])
            fprintf(stderr, "  stayed on the CPU, %s: %llu\n", reason_text[i], (unsigned long long)g_stats.cpu_calls[i]);
    if (g_verify)
        fprintf(stderr, "  verify: %llu matched, %llu MISMATCHED, %llu not comparable (largest relative difference %.2e)\n",
                (unsigned long long)g_stats.verified, (unsigned long long)g_stats.mismatched,
                (unsigned long long)g_stats.unverifiable, g_stats.max_rel);
}
