// SPDX-License-Identifier: AGPL-3.0-only
#define _GNU_SOURCE
#include "ofl.h"
#include "ofl_gen.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define BUF_RECORDS 4096

int ofl_active, ofl_offloading, ofl_tracing;
__thread int ofl_bypass;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static ofl_rec g_buf[BUF_RECORDS];
static unsigned g_nbuf;
static int g_fd = -1;
static char g_dir[PATH_MAX - 128];
static const char *g_mode = "offload";
static uint64_t g_start_ns;
static uint64_t g_written_bytes, g_max_bytes;
static uint64_t g_records, g_dropped;
static uint64_t g_calls[OFL_NFUNCS], g_ns_all[OFL_NFUNCS], g_ns_top[OFL_NFUNCS];

static __thread unsigned tls_depth;
static __thread uint32_t tls_tid;

uint64_t ofl_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------------------------
// Finding the real function
// ---------------------------------------------------------------------------------------------

struct name_list {
    char *v[1024];
    int n;
};

static int collect_names(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    struct name_list *nl = data;
    if (info->dlpi_name && info->dlpi_name[0] && nl->n < 1024) nl->v[nl->n++] = strdup(info->dlpi_name);
    return 0;
}

static void *lookup_in(const char *file, const char *sym, void *self) {
    void *h = dlopen(file, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return NULL;
    void *p = dlsym(h, sym);
    return p == self ? NULL : p;
}

void *ofl_resolve(const char *sym, void *self, void *caller) {
    void *p = NULL;
    Dl_info di;

    // 1. What the calling module would have reached without us: itself and its dependencies.
    //    Needed for programs that load their math library in a private scope (NumPy, R packages).
    if (caller && dladdr(caller, &di) && di.dli_fname && di.dli_fname[0]) p = lookup_in(di.dli_fname, sym, self);

    // 2. The global scope after this library.
    if (!p) {
        p = dlsym(RTLD_NEXT, sym);
        if (p == self) p = NULL;
    }

    // 3. Any loaded object.
    if (!p) {
        struct name_list nl = {.n = 0};
        dl_iterate_phdr(collect_names, &nl);
        for (int i = 0; i < nl.n; i++) {
            if (!p) p = lookup_in(nl.v[i], sym, self);
            free(nl.v[i]);
        }
    }

    if (!p) {
        fprintf(stderr, "gpu-offload: cannot find the real %s\n", sym);
        abort();
    }
    return p;
}

// ---------------------------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------------------------

// Fingerprint of up to 8 evenly spaced 8-byte words. Cheap enough to run on every call, and enough
// to tell "same buffer, same contents" from "same address, new data". 0 means "not sampled".
static uint32_t fingerprint(uint64_t ptr, uint64_t bytes) {
    if (!ptr || bytes < 8) return 0;
    const unsigned char *p = (const unsigned char *)(uintptr_t)ptr;
    uint64_t words = bytes / 8, n = words < 8 ? words : 8;
    uint64_t h = 0x9E3779B97F4A7C15ull ^ bytes;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t idx = n == 1 ? 0 : i * (words - 1) / (n - 1), w;
        memcpy(&w, p + idx * 8, 8);
        h = (h ^ w) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    return (uint32_t)(h ^ (h >> 32)) | 1u;
}

static void write_all(int fd, const void *data, size_t len) {
    const char *p = data;
    while (len) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            return;
        }
        p += w;
        len -= (size_t)w;
    }
}

static void output_path(char *out, size_t cap, const char *ext) {
    snprintf(out, cap, "%s/%s.%d.%s", g_dir, program_invocation_short_name, (int)getpid(), ext);
}

// Caller holds g_lock.
static void flush_locked(void) {
    if (!g_nbuf) return;
    if (g_fd < 0) {
        char path[PATH_MAX];
        output_path(path, sizeof path, "trace");
        g_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (g_fd < 0) {
            g_dropped += g_nbuf;
            g_nbuf = 0;
            return;
        }
        struct {
            char magic[8];
            uint32_t version, record_size;
        } hdr = {OFL_MAGIC, OFL_VERSION, sizeof(ofl_rec)};
        write_all(g_fd, &hdr, sizeof hdr);
    }
    write_all(g_fd, g_buf, (size_t)g_nbuf * sizeof(ofl_rec));
    g_written_bytes += (uint64_t)g_nbuf * sizeof(ofl_rec);
    g_nbuf = 0;
}

void ofl_begin(ofl_rec *r, unsigned func, void *callsite) {
    r->func = (uint16_t)func;
    r->callsite = (uint64_t)(uintptr_t)callsite;
    r->depth = tls_depth < 255 ? (uint8_t)tls_depth : 255;
    tls_depth++;
    if (!tls_tid) tls_tid = (uint32_t)syscall(SYS_gettid);
    r->tid = tls_tid;
    if (ofl_tracing && !(r->flags & OFL_FLAG_WSQUERY))
        for (int i = 0; i < 3; i++)
            if (r->buf_role[i] & OFL_ROLE_R) r->sig_pre[i] = fingerprint(r->buf_ptr[i], r->buf_bytes[i]);
    r->t_ns = ofl_now_ns() - g_start_ns;
}

void ofl_stop(ofl_rec *r) {
    r->dur_ns = ofl_now_ns() - g_start_ns - r->t_ns;
    tls_depth--;
}

void ofl_commit(ofl_rec *r) {
    if (r->device == OFL_DEV_CPU && r->pred_cpu_s > 0) ofl_policy_observe_cpu(r);
    if (!ofl_tracing) return;
    if (!(r->flags & OFL_FLAG_WSQUERY))
        for (int i = 0; i < 3; i++)
            if (r->buf_role[i] & OFL_ROLE_W) r->sig_post[i] = fingerprint(r->buf_ptr[i], r->buf_bytes[i]);

    pthread_mutex_lock(&g_lock);
    g_calls[r->func]++;
    g_ns_all[r->func] += r->dur_ns;
    if (r->depth == 0) g_ns_top[r->func] += r->dur_ns;
    if (g_written_bytes >= g_max_bytes) {
        g_dropped++;  // trace size cap reached: keep the per-function totals only
    } else {
        g_buf[g_nbuf++] = *r;
        g_records++;
        if (g_nbuf == BUF_RECORDS) flush_locked();
    }
    pthread_mutex_unlock(&g_lock);
}

void ofl_end(ofl_rec *r) {
    ofl_stop(r);
    ofl_commit(r);
}

// ---------------------------------------------------------------------------------------------
// Start-up, fork and exit
// ---------------------------------------------------------------------------------------------

static void json_string(FILE *f, const char *s, size_t len) {
    fputc('"', f);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

static void write_meta(void) {
    char path[PATH_MAX];
    output_path(path, sizeof path, "meta.json");
    FILE *f = fopen(path, "w");
    if (!f) return;

    char cmd[4096];
    size_t n = 0;
    FILE *c = fopen("/proc/self/cmdline", "r");
    if (c) {
        n = fread(cmd, 1, sizeof cmd, c);
        fclose(c);
        while (n && cmd[n - 1] == 0) n--;
        for (size_t i = 0; i < n; i++)
            if (cmd[i] == 0) cmd[i] = ' ';
    }

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

    fprintf(f, "{\n  \"version\": %d,\n  \"record_size\": %zu,\n  \"pid\": %d,\n", OFL_VERSION, sizeof(ofl_rec),
            (int)getpid());
    fprintf(f, "  \"program\": ");
    json_string(f, program_invocation_short_name, strlen(program_invocation_short_name));
    fprintf(f, ",\n  \"cmdline\": ");
    json_string(f, cmd, n);
    fprintf(f, ",\n  \"mode\": \"%s\"", g_mode);
    fprintf(f, ",\n  \"wall_s\": %.6f,\n  \"user_s\": %.6f,\n  \"sys_s\": %.6f,\n", (ofl_now_ns() - g_start_ns) * 1e-9,
            ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6, ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6);
    fprintf(f, "  \"cpus\": %ld,\n  \"records\": %llu,\n  \"dropped\": %llu,\n", sysconf(_SC_NPROCESSORS_ONLN),
            (unsigned long long)g_records, (unsigned long long)g_dropped);
    ofl_policy_meta(f);

    fprintf(f, "  \"funcs\": [");
    const char *sep = "\n";
    for (unsigned i = 0; i < OFL_NFUNCS; i++) {
        if (!g_calls[i]) continue;
        const struct ofl_func *fn = &ofl_funcs[i];
        fprintf(f,
                "%s    {\"id\": %u, \"name\": \"%s\", \"fam\": \"%s\", \"op\": \"%s\", \"prec\": \"%c\", "
                "\"iface\": \"%s\", \"variant\": \"%s\", \"calls\": %llu, \"ns_all\": %llu, \"ns_top\": %llu}",
                sep, i, fn->name, fn->fam, fn->op, fn->prec, fn->iface, fn->variant, (unsigned long long)g_calls[i],
                (unsigned long long)g_ns_all[i], (unsigned long long)g_ns_top[i]);
        sep = ",\n";
    }
    fprintf(f, "\n  ],\n");

    // Executable mappings, so call-site addresses can be turned into module+offset offline.
    fprintf(f, "  \"maps\": [");
    sep = "\n";
    FILE *m = fopen("/proc/self/maps", "r");
    if (m) {
        char line[PATH_MAX + 128];
        while (fgets(line, sizeof line, m)) {
            unsigned long long start, end, off;
            char perms[8];
            int pos = 0;
            if (sscanf(line, "%llx-%llx %7s %llx %*s %*s %n", &start, &end, perms, &off, &pos) < 4) continue;
            if (!strchr(perms, 'x') || !pos || line[pos] != '/') continue;
            size_t len = strcspn(line + pos, "\n");
            fprintf(f, "%s    {\"start\": %llu, \"end\": %llu, \"off\": %llu, \"path\": ", sep, start, end, off);
            json_string(f, line + pos, len);
            fputc('}', f);
            sep = ",\n";
        }
        fclose(m);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

static void mkdir_p(const char *dir) {
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s", dir);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        mkdir(tmp, 0755);
        *p = '/';
    }
    mkdir(tmp, 0755);
}

static void reset_state(void) {
    g_nbuf = 0;
    g_fd = -1;
    g_written_bytes = g_records = g_dropped = 0;
    memset(g_calls, 0, sizeof g_calls);
    memset(g_ns_all, 0, sizeof g_ns_all);
    memset(g_ns_top, 0, sizeof g_ns_top);
    g_start_ns = ofl_now_ns();
}

// A forked child gets its own trace; the parent keeps the records it had buffered.
static void atfork_child(void) {
    int fd = g_fd;
    pthread_mutex_init(&g_lock, NULL);
    reset_state();
    if (fd >= 0) close(fd);
    tls_tid = 0;
    tls_depth = 0;
    ofl_bypass = 0;
    ofl_policy_fork_child();
}

// GPU_OFFLOAD_MODE = offload (default) | measure | off
// GPU_OFFLOAD_TRACE = directory for per-call traces (always on in measure mode)
__attribute__((constructor)) static void ofl_init(void) {
    const char *mode = getenv("GPU_OFFLOAD_MODE");
    const char *dir = getenv("GPU_OFFLOAD_TRACE");
    reset_state();
    if (mode && !strcmp(mode, "off")) {
        g_mode = "off";
        return;
    }
    if (mode && !strcmp(mode, "measure")) {
        g_mode = "measure";
        if (!dir || !dir[0]) dir = "/tmp/gpu-offload-trace";
    } else {
        ofl_offloading = 1;
    }

    if (dir && dir[0]) {
        snprintf(g_dir, sizeof g_dir, "%s", dir);
        mkdir_p(g_dir);
        const char *mb = getenv("GPU_OFFLOAD_TRACE_MAX_MB");
        g_max_bytes = (mb && atoll(mb) > 0 ? (uint64_t)atoll(mb) : 2048) << 20;
        ofl_tracing = access(g_dir, W_OK) == 0;
        if (!ofl_tracing) fprintf(stderr, "gpu-offload: cannot write to %s, tracing disabled\n", g_dir);
    }

    pthread_atfork(NULL, NULL, atfork_child);
    ofl_active = ofl_tracing || ofl_offloading;
}

__attribute__((destructor)) static void ofl_fini(void) {
    if (!ofl_active) return;
    pthread_mutex_lock(&g_lock);
    int tracing = ofl_tracing;
    ofl_active = ofl_offloading = ofl_tracing = 0;
    if (tracing) {
        flush_locked();
        write_meta();  // also for processes that made no calls: "0% interceptable" is a result too
        if (g_fd >= 0) close(g_fd);
        g_fd = -1;
    }
    pthread_mutex_unlock(&g_lock);
    ofl_policy_summary();
}
