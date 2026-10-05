#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generates the BLAS / LAPACK interception wrappers.

usage: gen_wrappers.py OUT.c OUT.h

Every operation is described once in OPS and expanded for each precision (s, d, c, z), each
interface (Fortran, CBLAS) and each symbol naming scheme in VARIANTS.
"""
import sys

# precision letter -> (element bytes, C type of the real part, is complex)
PREC = {"s": (4, "float", False), "d": (8, "double", False), "c": (8, "float", True), "z": (16, "double", True)}

# (tag, Fortran prefix, Fortran suffix, CBLAS prefix, CBLAS suffix, integer type)
VARIANTS = [
    ("lp64", "", "_", "cblas_", "", "int32_t"),
    # OpenBLAS builds bundled in the NumPy / SciPy wheels from PyPI.
    ("scipy32", "scipy_", "_", "scipy_cblas_", "", "int32_t"),
    ("scipy64", "scipy_", "_64_", "scipy_cblas_", "64_", "int64_t"),
]

# Functions written by hand in fftw.c: (id macro, symbol family name, op, precision)
MANUAL = [
    ("FFTW_PLAN_D", "fftw_plan_*", "plan", "d"),
    ("FFTW_EXEC_D", "fftw_execute*", "fft", "d"),
    ("FFTW_PLAN_S", "fftwf_plan_*", "plan", "s"),
    ("FFTW_EXEC_S", "fftwf_execute*", "fft", "s"),
]

OPS = []


def op(name, fam, precs, f=None, c=None, dims="", bufs="", flops="0.0", preds="", pre="", ret=None, gpu=None):
    """Describes one operation.

    f / c : Fortran / CBLAS argument lists as "kind:name" tokens.
            Fortran kinds: c char, i integer, S scalar of the op's precision, R real scalar, x other.
            CBLAS kinds:   e enum, i integer, s scalar (by value if real, pointer if complex),
                           r real scalar by value, p pointer.
    dims  : integer arguments stored in the record (up to 4).
    bufs  : "role:arg:elements" for up to 3 operand buffers; role is r, w, rw, or wb
            (written, and also read unless beta == 0).
    preds : "var=arg:CHAR:ENUM" booleans, e.g. side_l=side:L:141 is true for side 'L' / CblasLeft.
    pre   : extra C statements that define helper values used by bufs / flops.
    ret   : "real" if the function returns a real scalar.
    gpu   : "OP field=source ..." fills an ofl_call so the call can run on the GPU. Sources are
            argument names; "variant" takes a C expression over the preds.
    """
    OPS.append(dict(name=name, fam=fam, precs=precs, f=f, c=c, dims=dims, bufs=bufs, flops=flops, preds=preds,
                    pre=pre, ret=ret, gpu=gpu))


KA = "int64_t ka = side_l ? m : n;"
MNMX = "int64_t mn = m < n ? m : n, mx = m < n ? n : m;"

# ---- BLAS level 3 ----
op("{p}gemm", "blas3", "sdcz",
   f="c:ta c:tb i:m i:n i:k S:alpha x:a i:lda x:b i:ldb S:beta x:c i:ldc",
   c="e:order e:ta e:tb i:m i:n i:k s:alpha p:a i:lda p:b i:ldb s:beta p:c i:ldc",
   dims="m n k", bufs="r:a:m*k r:b:k*n wb:c:m*n", flops="2.0*m*n*k",
   gpu="GEMM trans_a=ta trans_b=tb m=m n=n k=k alpha=alpha beta=beta a=a lda=lda b=b ldb=ldb c=c ldc=ldc c_out=c")
for nm, precs in (("{p}symm", "sdcz"), ("{p}hemm", "cz")):
    op(nm, "blas3", precs,
       f="c:side c:uplo i:m i:n S:alpha x:a i:lda x:b i:ldb S:beta x:c i:ldc",
       c="e:order e:side e:uplo i:m i:n s:alpha p:a i:lda p:b i:ldb s:beta p:c i:ldc",
       dims="m n", bufs="r:a:ka*ka r:b:m*n wb:c:m*n", flops="2.0*m*n*ka", preds="side_l=side:L:141", pre=KA)
RK_GPU = "uplo=uplo trans_a=trans m=n n=n k=k alpha=alpha beta=beta a=a lda=lda c=c ldc=ldc c_out=c"
op("{p}syrk", "blas3", "sdcz",
   f="c:uplo c:trans i:n i:k S:alpha x:a i:lda S:beta x:c i:ldc",
   c="e:order e:uplo e:trans i:n i:k s:alpha p:a i:lda s:beta p:c i:ldc",
   dims="n k", bufs="r:a:n*k wb:c:n*n", flops="1.0*n*n*k", gpu="SYRK " + RK_GPU)
op("{p}herk", "blas3", "cz",
   f="c:uplo c:trans i:n i:k R:alpha x:a i:lda R:beta x:c i:ldc",
   c="e:order e:uplo e:trans i:n i:k r:alpha p:a i:lda r:beta p:c i:ldc",
   dims="n k", bufs="r:a:n*k wb:c:n*n", flops="1.0*n*n*k", gpu="HERK " + RK_GPU)
op("{p}syr2k", "blas3", "sdcz",
   f="c:uplo c:trans i:n i:k S:alpha x:a i:lda x:b i:ldb S:beta x:c i:ldc",
   c="e:order e:uplo e:trans i:n i:k s:alpha p:a i:lda p:b i:ldb s:beta p:c i:ldc",
   dims="n k", bufs="r:a:n*k r:b:n*k wb:c:n*n", flops="2.0*n*n*k")
op("{p}her2k", "blas3", "cz",
   f="c:uplo c:trans i:n i:k S:alpha x:a i:lda x:b i:ldb R:beta x:c i:ldc",
   c="e:order e:uplo e:trans i:n i:k s:alpha p:a i:lda p:b i:ldb r:beta p:c i:ldc",
   dims="n k", bufs="r:a:n*k r:b:n*k wb:c:n*n", flops="2.0*n*n*k")
for nm, gpu in (("{p}trmm", None),
                ("{p}trsm", "TRSM side=side uplo=uplo trans_a=transa diag=diag m=m n=n alpha=alpha a=a lda=lda "
                            "b=b ldb=ldb b_out=b")):
    op(nm, "blas3", "sdcz",
       f="c:side c:uplo c:transa c:diag i:m i:n S:alpha x:a i:lda x:b i:ldb",
       c="e:order e:side e:uplo e:transa e:diag i:m i:n s:alpha p:a i:lda p:b i:ldb",
       dims="m n", bufs="r:a:ka*ka rw:b:m*n", flops="1.0*m*n*ka", preds="side_l=side:L:141", pre=KA, gpu=gpu)

# ---- BLAS level 2 ----
op("{p}gemv", "blas2", "sdcz",
   f="c:trans i:m i:n S:alpha x:a i:lda x:x i:incx S:beta x:y i:incy",
   c="e:order e:trans i:m i:n s:alpha p:a i:lda p:x i:incx s:beta p:y i:incy",
   dims="m n", bufs="r:a:m*n r:x:(trans_n?n:m) wb:y:(trans_n?m:n)", flops="2.0*m*n", preds="trans_n=trans:N:111")
op("{p}ger", "blas2", "sd",
   f="i:m i:n S:alpha x:x i:incx x:y i:incy x:a i:lda",
   c="e:order i:m i:n s:alpha p:x i:incx p:y i:incy p:a i:lda",
   dims="m n", bufs="r:x:m r:y:n rw:a:m*n", flops="2.0*m*n")

# ---- BLAS level 1 ----
op("{p}axpy", "blas1", "sdcz", f="i:n S:alpha x:x i:incx x:y i:incy", c="i:n s:alpha p:x i:incx p:y i:incy",
   dims="n", bufs="r:x:n rw:y:n", flops="2.0*n")
op("{p}scal", "blas1", "sdcz", f="i:n S:alpha x:x i:incx", c="i:n s:alpha p:x i:incx",
   dims="n", bufs="rw:x:n", flops="1.0*n")
op("{p}dot", "blas1", "sd", f="i:n x:x i:incx x:y i:incy", c="i:n p:x i:incx p:y i:incy",
   dims="n", bufs="r:x:n r:y:n", flops="2.0*n", ret="real")
op("{p}nrm2", "blas1", "sd", f="i:n x:x i:incx", c="i:n p:x i:incx",
   dims="n", bufs="r:x:n", flops="2.0*n", ret="real")

# ---- LAPACK (Fortran interface; LAPACKE and SciPy end up here) ----
LU = "mn*mn*(mx - mn/3.0)"
op("{p}getrf", "lapack", "sdcz", f="i:m i:n x:a i:lda x:ipiv x:info",
   dims="m n", bufs="rw:a:m*n", flops=LU, pre=MNMX,
   gpu="GETRF m=m n=n a=a lda=lda a_out=a ipiv_out=ipiv info_out=info")
op("{p}getrs", "lapack", "sdcz", f="c:trans i:n i:nrhs x:a i:lda x:ipiv x:b i:ldb x:info",
   dims="n nrhs", bufs="r:a:n*n rw:b:n*nrhs", flops="2.0*n*n*nrhs")
op("{p}gesv", "lapack", "sdcz", f="i:n i:nrhs x:a i:lda x:ipiv x:b i:ldb x:info",
   dims="n nrhs", bufs="rw:a:n*n rw:b:n*nrhs", flops="2.0/3*n*n*n + 2.0*n*n*nrhs",
   gpu="GESV m=n n=n k=nrhs a=a lda=lda a_out=a b=b ldb=ldb b_out=b ipiv_out=ipiv info_out=info")
op("{p}getri", "lapack", "sdcz", f="i:n x:a i:lda x:ipiv x:work i:lwork x:info",
   dims="n", bufs="rw:a:n*n", flops="4.0/3*n*n*n")
op("{p}potrf", "lapack", "sdcz", f="c:uplo i:n x:a i:lda x:info",
   dims="n", bufs="rw:a:n*n", flops="1.0/3*n*n*n",
   gpu="POTRF uplo=uplo m=n n=n a=a lda=lda a_out=a info_out=info")
op("{p}potrs", "lapack", "sdcz", f="c:uplo i:n i:nrhs x:a i:lda x:b i:ldb x:info",
   dims="n nrhs", bufs="r:a:n*n rw:b:n*nrhs", flops="2.0*n*n*nrhs")
op("{p}posv", "lapack", "sdcz", f="c:uplo i:n i:nrhs x:a i:lda x:b i:ldb x:info",
   dims="n nrhs", bufs="rw:a:n*n rw:b:n*nrhs", flops="1.0/3*n*n*n + 2.0*n*n*nrhs",
   gpu="POSV uplo=uplo m=n n=n k=nrhs a=a lda=lda a_out=a b=b ldb=ldb b_out=b info_out=info")
op("{p}geqrf", "lapack", "sdcz", f="i:m i:n x:a i:lda x:tau x:work i:lwork x:info",
   dims="m n", bufs="rw:a:m*n", flops="2.0*" + LU, pre=MNMX)

GELSD = "i:m i:n i:nrhs x:a i:lda x:b i:ldb x:s x:rcond x:rank x:work i:lwork {rw}x:iwork x:info"
SYEV = "c:jobz c:uplo i:n x:a i:lda x:w x:work i:lwork {rw}x:info"
SYEVD = "c:jobz c:uplo i:n x:a i:lda x:w x:work i:lwork {rwl}x:iwork i:liwork x:info"
SYEVR = ("c:jobz c:range c:uplo i:n x:a i:lda x:vl x:vu x:il x:iu x:abstol x:m x:w x:z i:ldz x:isuppz "
         "x:work i:lwork {rwl}x:iwork i:liwork x:info")
GESDD = "c:jobz i:m i:n x:a i:lda x:s x:u i:ldu x:vt i:ldvt x:work i:lwork {rw}x:iwork x:info"
GESVD = "c:jobu c:jobvt i:m i:n x:a i:lda x:s x:u i:ldu x:vt i:ldvt x:work i:lwork {rw}x:info"
EIG = "(jobz_v ? 9.0 : 4.0/3)*n*n*n"
EIG_GPU = "SYEVD job_a=jobz uplo=uplo m=n n=n a=a lda=lda a_out=a w_out=w info_out=info variant=jobz_v"
SVD_GPU = "m=m n=n a=a lda=lda w_out=s u_out=u ldu=ldu vt_out=vt ldvt=ldvt info_out=info"
# The complex routines take an extra real workspace argument; the names differ for sy/he.
for precs, sy, rw, rwl in (("sd", "sy", "", ""), ("cz", "he", "x:rwork ", "x:rwork i:lrwork ")):
    op("{p}gelsd", "lapack", precs, f=GELSD.format(rw=rw), dims="m n nrhs",
       bufs="rw:a:m*n rw:b:mx*nrhs", flops="4.0*mx*mn*mn", pre=MNMX)
    op("{p}%sev" % sy, "lapack", precs, f=SYEV.format(rw=rw), dims="n", bufs="rw:a:n*n", flops=EIG,
       preds="jobz_v=jobz:V:0", gpu=EIG_GPU)
    op("{p}%sevd" % sy, "lapack", precs, f=SYEVD.format(rwl=rwl), dims="n", bufs="rw:a:n*n", flops=EIG,
       preds="jobz_v=jobz:V:0", gpu=EIG_GPU)
    op("{p}%sevr" % sy, "lapack", precs, f=SYEVR.format(rwl=rwl), dims="n", bufs="rw:a:n*n", flops=EIG,
       preds="jobz_v=jobz:V:0")
    op("{p}gesdd", "lapack", precs, f=GESDD.format(rw=rw), dims="m n", bufs="rw:a:m*n",
       flops="(jobz_n ? 4.0 : 12.0)*mx*mn*mn", preds="jobz_n=jobz:N:0", pre=MNMX,
       gpu="GESVD job_a=jobz job_b=jobz variant=!jobz_n " + SVD_GPU)
    op("{p}gesvd", "lapack", precs, f=GESVD.format(rw=rw), dims="m n", bufs="rw:a:m*n",
       flops="((jobu_n && jobvt_n) ? 4.0 : 12.0)*mx*mn*mn", preds="jobu_n=jobu:N:0 jobvt_n=jobvt:N:0", pre=MNMX,
       gpu="GESVD job_a=jobu job_b=jobvt variant=!(jobu_n&&jobvt_n) " + SVD_GPU)
GEEV_FLOPS = "((jobvl_n && jobvr_n) ? 10.0 : 25.0)*n*n*n"
GEEV_PREDS = "jobvl_n=jobvl:N:0 jobvr_n=jobvr:N:0"
op("{p}geev", "lapack", "sd",
   f="c:jobvl c:jobvr i:n x:a i:lda x:wr x:wi x:vl i:ldvl x:vr i:ldvr x:work i:lwork x:info",
   dims="n", bufs="rw:a:n*n", flops=GEEV_FLOPS, preds=GEEV_PREDS)
op("{p}geev", "lapack", "cz",
   f="c:jobvl c:jobvr i:n x:a i:lda x:w x:vl i:ldvl x:vr i:ldvr x:work i:lwork x:rwork x:info",
   dims="n", bufs="rw:a:n*n", flops=GEEV_FLOPS, preds=GEEV_PREDS)

ROLES = {"r": "OFL_ROLE_R", "w": "OFL_ROLE_W", "rw": "OFL_ROLE_RW"}
CHAR_FIELDS = {"trans_a", "trans_b", "side", "uplo", "diag", "job_a", "job_b"}
INT_FIELDS = {"m", "n", "k", "lda", "ldb", "ldc", "ldu", "ldvt"}
SCALAR_FIELDS = {"alpha", "beta"}
QUERY_ARGS = ("lwork", "liwork", "lrwork")


def parse(spec):
    return [tuple(tok.split(":", 1)) for tok in spec.split()]


def gpu_fields(o, p, fortran, INT, names):
    """C statements that fill `ofl_call c` from the wrapper's arguments."""
    esz, real_t, cplx = PREC[p]
    opname, *fields = o["gpu"].split()
    lines = [f"c.op = OFL_OP_{opname};", f"c.prec = '{p}';", f"c.int_bytes = sizeof({INT});", "c.variant = r.variant;"]
    if not fortran and "order" in names:
        lines.append("c.row_major = p_order == 101;")
    for tok in fields:
        field, src = tok.split("=", 1)
        if field == "variant":
            continue  # already in the record
        if field in CHAR_FIELDS:
            lines.append(f"c.{field} = (char)((*(const char *)p_{src}) & 0xDF);" if fortran
                         else f"c.{field} = ofl_cblas_letter(p_{src});")
        elif field in INT_FIELDS:
            lines.append(f"c.{field} = {src};")
        elif field in SCALAR_FIELDS:
            by_value = not fortran and (names[src] == "r" or (names[src] == "s" and not cplx))
            lines.append(f"c.{field} = {'&' if by_value else ''}p_{src};")
        else:
            lines.append(f"c.{field} = p_{src};")
    return lines


def emit(out, table, variant, o, p, iface):
    tag, fpre, fsuf, cpre, csuf, INT = variant
    esz, real_t, cplx = PREC[p]
    base = o["name"].format(p=p)
    args = parse(o[iface[0]])
    fortran = iface == "fortran"
    sym = (fpre + base + fsuf) if fortran else (cpre + base + csuf)
    fid = len(table)
    table.append((sym, o["fam"], o["name"].format(p=""), iface, tag, p))

    # Parameter list.
    params = []
    for kind, name in args:
        if fortran or kind == "p" or (kind == "s" and cplx):
            params.append(f"void *p_{name}")
        elif kind == "e":
            params.append(f"int p_{name}")
        elif kind == "i":
            params.append(f"{INT} p_{name}")
        else:  # real scalar by value
            params.append(f"{real_t} p_{name}")
    call = [f"p_{name}" for _, name in args]
    if fortran:  # hidden string lengths that Fortran compilers append for character arguments
        nchar = sum(1 for kind, _ in args if kind == "c")
        params += [f"size_t l{i}" for i in range(nchar)]
        call += [f"l{i}" for i in range(nchar)]
    ret_t = real_t if o["ret"] == "real" else "void"
    params_s, call_s = ", ".join(params), ", ".join(call)
    ra = "__builtin_return_address(0)"

    w = out.append
    w(f"OFL_EXPORT {ret_t} {sym}({params_s}) {{")
    w(f"    typedef {ret_t} (*fn_t)({params_s});")
    w("    static fn_t real;")
    w(f'    if (__builtin_expect(!real, 0)) real = (fn_t)ofl_resolve("{sym}", (void *){sym}, {ra});')
    # Calls without a GPU path only matter when tracing; GPU-capable ones also when offloading.
    idle = "!ofl_active || ofl_bypass" if o["gpu"] else "!ofl_tracing || ofl_bypass"
    if ret_t == "void":
        w(f"    if ({idle}) {{ real({call_s}); return; }}")
    else:
        w(f"    if ({idle}) return real({call_s});")
    w("    ofl_rec r = {0};")

    # Integer arguments as 64-bit locals, so size products cannot overflow.
    names = {name: kind for kind, name in args}
    for kind, name in args:
        if kind == "i":
            w(f"    int64_t {name} = " + (f"*(const {INT} *)p_{name};" if fortran else f"p_{name};"))
    for pred in o["preds"].split():
        var, rest = pred.split("=")
        arg, ch, enum = rest.split(":")
        if fortran:
            w(f"    int {var} = ((*(const char *)p_{arg}) & 0xDF) == '{ch}';")
        else:
            w(f"    int {var} = p_{arg} == {enum};")
    if o["pre"]:
        w(f"    {o['pre']}")

    for i, d in enumerate(o["dims"].split()):
        w(f"    r.dims[{i}] = {d};")
    for i, b in enumerate(o["bufs"].split()):
        role, arg, elems = b.split(":", 2)
        if role == "wb":
            kind = names["beta"]
            if fortran or (kind == "s" and cplx):
                parts = 2 if (cplx and kind in ("S", "s")) else 1
                zero = " && ".join(f"((const {real_t} *)p_beta)[{j}] == 0" for j in range(parts))
            else:
                zero = "p_beta == 0"
            role_c = f"({zero}) ? OFL_ROLE_W : OFL_ROLE_RW"
        else:
            role_c = ROLES[role]
        w(f"    r.buf_ptr[{i}] = (uint64_t)(uintptr_t)p_{arg};")
        w(f"    r.buf_bytes[{i}] = ofl_nb({elems}, {esz});")
        w(f"    r.buf_role[{i}] = {role_c};")
    w(f"    r.flops = {'4.0' if cplx else '1.0'} * ({o['flops']});")
    queries = [q for q in QUERY_ARGS if q in names]
    if queries:
        w(f"    if ({' || '.join(q + ' == -1' for q in queries)}) r.flags |= OFL_FLAG_WSQUERY;")

    done = "return;" if ret_t == "void" else "return ret;"
    # LAPACK reports failure through info; a failed call returns early and must not look fast.
    failed = f"    if (*(const {INT} *)p_info != 0) r.flags |= OFL_FLAG_FAILED;" if "info" in names else None
    if o["gpu"]:
        variant_expr = [t.split("=", 1)[1] for t in o["gpu"].split() if t.startswith("variant=")]
        if variant_expr:
            w(f"    r.variant = ({variant_expr[0]}) != 0;")
        w("    if (ofl_offloading && r.flops >= ofl_min_flops && !(r.flags & OFL_FLAG_WSQUERY)) {")
        w("        ofl_call c = {0};")
        for line in gpu_fields(o, p, fortran, INT, names):
            w(f"        {line}")
        w(f"        int gpu = ofl_dispatch(&r, &c, {fid}, {ra});")
        w("        if (gpu == 1) return;")
        w(f"        ofl_begin(&r, {fid}, {ra});")
        w(f"        real({call_s});")
        w("        ofl_stop(&r);")
        if failed:
            w("    " + failed)
        w("        if (gpu == 2) ofl_verify(&r, &c);")
        w("        ofl_commit(&r);")
        w("        return;")
        w("    }")
        w(f"    if (!ofl_tracing) {{ real({call_s}); return; }}")

    w(f"    ofl_begin(&r, {fid}, {ra});")
    if ret_t == "void":
        w(f"    real({call_s});")
    else:
        w(f"    {ret_t} ret = real({call_s});")
    if failed:
        w(failed)
    w("    ofl_end(&r);")
    w(f"    {done}")
    w("}")
    w("")


def main():
    out_c, out_h = sys.argv[1], sys.argv[2]
    body, table = [], []
    for variant in VARIANTS:
        for o in OPS:
            for p in o["precs"]:
                emit(body, table, variant, o, p, "fortran")
                if o["c"]:
                    emit(body, table, variant, o, p, "cblas")

    manual_ids = {}
    for macro, name, opname, p in MANUAL:
        manual_ids[macro] = len(table)
        table.append((name, "fftw", opname, "c", "fftw3", p))

    with open(out_h, "w") as h:
        h.write("// Generated by tools/gen_wrappers.py. Do not edit.\n#ifndef OFL_GEN_H\n#define OFL_GEN_H\n")
        h.write(f"#define OFL_NFUNCS {len(table)}\n")
        for macro, fid in manual_ids.items():
            h.write(f"#define OFL_FN_{macro} {fid}\n")
        h.write("#endif\n")

    with open(out_c, "w") as c:
        c.write("// Generated by tools/gen_wrappers.py. Do not edit.\n")
        c.write('#include "ofl.h"\n#include "ofl_gen.h"\n\n')
        c.write("const struct ofl_func ofl_funcs[OFL_NFUNCS] = {\n")
        for name, fam, opname, iface, tag, p in table:
            c.write(f'    {{"{name}", "{fam}", "{opname}", "{iface}", "{tag}", \'{p}\'}},\n')
        c.write("};\n\n")
        c.write("\n".join(body))


if __name__ == "__main__":
    main()
