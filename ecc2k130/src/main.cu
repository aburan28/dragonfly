// ECC2K-130 client: Pollard rho with the Frobenius-based iteration function,
// bitsliced over the permuted type-II optimal normal basis.
//
// Build as CUDA (nvcc/clang) or as plain C++ with -DECC_NO_CUDA; the walk code
// is identical in both cases, only the word width differs (32 bits on the
// device, 64 on the host).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

#include "../include/curveparams.h"
#include "../include/kernel.h"
#include "../include/ref.h"
#include "../include/solver.h"

#ifndef ECC_NO_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(x)                                                                   \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e_));                                            \
            exit(2);                                                                    \
        }                                                                               \
    } while (0)
typedef unsigned int DeviceWord;
#endif

#ifdef _OPENMP
#include <omp.h>
#endif



static double nowSeconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct Options {
    int curve = 131;
    int threads = 0;
    int blocks = 0;
    int steps = 64;
    int dpWeight = -1;
    long launches = 0;
    int instance = -1;
    unsigned runId = 1;
    u64 maxIters = 0;
    bool bench = false;
    bool test = false;
    bool selfCheck = false;
    unsigned dpCap = 1u << 16;
    int device = 0;
    int verify = 8;
    std::string dpFile;
};

// ---------------------------------------------------------------------------
// host backend
// ---------------------------------------------------------------------------
template <class Cfg>
struct HostEngine {
    typedef u64 W;
    typedef Kernel<Cfg, W> K;
    static const int M = Cfg::M;
    static const int LANES = WordTraits<W>::LANES;
    static const int BATCH = ECC_BATCH;

    std::vector<W> x, y, pchain, dead;
    std::vector<u64> seed, startIter;
    std::vector<DpRecord> dp;
    unsigned dpCount = 0;
    WalkParams<W> P;
    std::vector<u64> px, py, qx, qy;

    void setup(const Options &o, const unsigned long long *cpx, const unsigned long long *cpy,
               const unsigned long long *cqx, const unsigned long long *cqy) {
        const size_t T = (size_t)o.threads;
        x.assign(T * BATCH * M, 0);
        y.assign(T * BATCH * M, 0);
        pchain.assign(T * BATCH * M, 0);
        dead.assign(T * BATCH, 0);
        seed.assign(T * BATCH * LANES, 0);
        startIter.assign(T * BATCH * LANES, 0);
        dp.assign(o.dpCap, DpRecord());
        px.assign(cpx, cpx + 3);
        py.assign(cpy, cpy + 3);
        qx.assign(cqx, cqx + 3);
        qy.assign(cqy, cqy + 3);
        P.threads = o.threads;
        P.steps = o.steps;
        P.dpWeight = o.dpWeight;
        P.runId = o.runId;
        P.maxIters = o.maxIters;
        P.iterBase = 0;
        P.x = x.data();
        P.y = y.data();
        P.pchain = pchain.data();
        P.seed = seed.data();
        P.startIter = startIter.data();
        P.dead = dead.data();
        P.dp = dp.data();
        P.dpCount = &dpCount;
        P.dpCap = o.dpCap;
        P.consts.px = px.data();
        P.consts.py = py.data();
        P.consts.qx = qx.data();
        P.consts.qy = qy.data();
        const WalkParams<W> pp = P;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < o.threads; ++t) K::init(t, pp);
    }

    void launch(u64 iterBase) {
        P.iterBase = iterBase;
        const WalkParams<W> pp = P;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < pp.threads; ++t) K::run(t, pp);
    }

    // revive the lanes that reported during the last launch
    void reseed(u64 iterBase) {
        P.iterBase = iterBase;
        const WalkParams<W> pp = P;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < pp.threads; ++t) K::reseed(t, pp);
    }

    unsigned fetch(std::vector<DpRecord> &out) {
        const unsigned n = dpCount;
        const unsigned m = n < P.dpCap ? n : P.dpCap;
        out.assign(dp.begin(), dp.begin() + m);
        dpCount = 0;
        return n;
    }
    const char *name() const { return "cpu"; }
    u64 walksPerLaunch() const { return (u64)P.threads * BATCH * LANES; }
};

#ifndef ECC_NO_CUDA
// ---------------------------------------------------------------------------
// device backend
// ---------------------------------------------------------------------------
template <class Cfg>
struct CudaEngine {
    typedef DeviceWord W;
    static const int M = Cfg::M;
    static const int LANES = WordTraits<W>::LANES;
    static const int BATCH = ECC_BATCH;
    WalkParams<W> P;
    std::vector<DpRecord> staging;

    void setup(const Options &o, const unsigned long long *cpx, const unsigned long long *cpy,
               const unsigned long long *cqx, const unsigned long long *cqy) {
        const size_t T = (size_t)o.threads;
        const size_t fw = T * BATCH * M * sizeof(W);
        const size_t lw = T * BATCH * LANES * sizeof(u64);
        CUDA_CHECK(cudaMalloc(&P.x, fw));
        CUDA_CHECK(cudaMalloc(&P.y, fw));
        CUDA_CHECK(cudaMalloc(&P.pchain, fw));
        CUDA_CHECK(cudaMalloc(&P.dead, T * BATCH * sizeof(W)));
        CUDA_CHECK(cudaMemset(P.dead, 0, T * BATCH * sizeof(W)));
        CUDA_CHECK(cudaMalloc(&P.seed, lw));
        CUDA_CHECK(cudaMalloc(&P.startIter, lw));
        CUDA_CHECK(cudaMalloc(&P.dp, (size_t)o.dpCap * sizeof(DpRecord)));
        CUDA_CHECK(cudaMalloc(&P.dpCount, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(P.dpCount, 0, sizeof(unsigned)));
        unsigned long long *dk;
        CUDA_CHECK(cudaMalloc(&dk, 12 * sizeof(u64)));
        u64 hk[12];
        memcpy(hk + 0, cpx, 3 * sizeof(u64));
        memcpy(hk + 3, cpy, 3 * sizeof(u64));
        memcpy(hk + 6, cqx, 3 * sizeof(u64));
        memcpy(hk + 9, cqy, 3 * sizeof(u64));
        CUDA_CHECK(cudaMemcpy(dk, hk, sizeof hk, cudaMemcpyHostToDevice));
        P.consts.px = dk;
        P.consts.py = dk + 3;
        P.consts.qx = dk + 6;
        P.consts.qy = dk + 9;
        P.threads = o.threads;
        P.steps = o.steps;
        P.dpWeight = o.dpWeight;
        P.runId = o.runId;
        P.maxIters = o.maxIters;
        P.iterBase = 0;
        P.dpCap = o.dpCap;
        staging.resize(o.dpCap);
        const int blocks = (o.threads + ECC_THREADS - 1) / ECC_THREADS;
        eccInitKernel<Cfg, W><<<blocks, ECC_THREADS>>>(P);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void launch(u64 iterBase) {
        P.iterBase = iterBase;
        const int blocks = (P.threads + ECC_THREADS - 1) / ECC_THREADS;
        eccWalkKernel<Cfg, W><<<blocks, ECC_THREADS>>>(P);
        CUDA_CHECK(cudaGetLastError());
    }

    void reseed(u64 iterBase) {
        P.iterBase = iterBase;
        const int blocks = (P.threads + ECC_THREADS - 1) / ECC_THREADS;
        eccReseedKernel<Cfg, W><<<blocks, ECC_THREADS>>>(P);
        CUDA_CHECK(cudaGetLastError());
    }

    unsigned fetch(std::vector<DpRecord> &out) {
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned n = 0;
        CUDA_CHECK(cudaMemcpy(&n, P.dpCount, sizeof n, cudaMemcpyDeviceToHost));
        const unsigned m = n < P.dpCap ? n : P.dpCap;
        out.resize(m);
        if (m) CUDA_CHECK(cudaMemcpy(out.data(), P.dp, (size_t)m * sizeof(DpRecord), cudaMemcpyDeviceToHost));
        if (n) CUDA_CHECK(cudaMemset(P.dpCount, 0, sizeof(unsigned)));
        return n;
    }
    const char *name() const { return "cuda"; }
    u64 walksPerLaunch() const { return (u64)P.threads * BATCH * LANES; }
};
#endif

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------
static int gFail = 0;
static void report(const char *what, bool ok) {
    printf("  %-46s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) gFail++;
}

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 1) {}
    u64 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

template <class Cfg>
static typename Ref<Cfg>::Elem randomElem(Rng &rng) {
    typename Ref<Cfg>::Elem e = Ref<Cfg>::zero();
    for (int i = 0; i < Cfg::M; ++i)
        if (rng.next() & 1) Ref<Cfg>::setBit(e, i);
    return e;
}

// random point of order l on y^2 + xy = x^3 + 1
template <class Cfg>
static typename Ref<Cfg>::Point randomPoint(Rng &rng, const U192 &ell) {
    typedef Ref<Cfg> R;
    for (;;) {
        const typename R::Elem x = randomElem<Cfg>(rng);
        if (R::isZero(x)) continue;
        const typename R::Elem c = R::add(x, R::inv(R::mul(x, x)));
        if (R::trace(c)) continue;
        const typename R::Elem z = R::halfTrace(c);
        typename R::Point p = R::make(x, R::mul(x, z));
        if (!R::onCurve(p)) continue;
        p = R::dbl(R::dbl(p));
        if (p.inf) continue;
        if (!R::scalarMul(p, ell).inf) continue;
        return p;
    }
}

template <class Cfg>
static void testField(Rng &rng) {
    typedef Ref<Cfg> R;
    typedef FieldBs<Cfg, u64> F;
    const int M = Cfg::M;
    const int LANES = 64;
    std::vector<typename R::Elem> a(LANES), b(LANES);
    std::vector<u64> ba(M, 0), bb(M, 0), bc(M, 0), bd(M, 0);
    for (int l = 0; l < LANES; ++l) {
        a[l] = randomElem<Cfg>(rng);
        b[l] = randomElem<Cfg>(rng);
        F::setLane(ba.data(), l, a[l].v);
        F::setLane(bb.data(), l, b[l].v);
    }
    bool okMul = true, okSqr = true, okInv = true, okHw = true, okSig = true;
    F::mul(ba.data(), bb.data(), bc.data());
    for (int l = 0; l < LANES; ++l) {
        typename R::Elem got;
        F::getLane(bc.data(), l, got.v);
        if (!(got == R::mul(a[l], b[l]))) okMul = false;
    }
    F::sqr(ba.data(), bc.data());
    for (int l = 0; l < LANES; ++l) {
        typename R::Elem got;
        F::getLane(bc.data(), l, got.v);
        if (!(got == R::sqr(a[l]))) okSqr = false;
    }
    F::inv(ba.data(), bc.data());
    for (int l = 0; l < LANES; ++l) {
        typename R::Elem got;
        F::getLane(bc.data(), l, got.v);
        if (!R::isZero(a[l]) && !(got == R::inv(a[l]))) okInv = false;
    }
    F::sigmaRun(ba.data(), 7, bc.data());
    for (int l = 0; l < LANES; ++l) {
        typename R::Elem got;
        F::getLane(bc.data(), l, got.v);
        if (!(got == R::sigma(a[l], 7))) okSig = false;
    }
    std::vector<u64> hb(Cfg::HWBITS > 4 ? Cfg::HWBITS : 4, 0);
    Cfg::hamming(ba.data(), hb.data());
    for (int l = 0; l < LANES; ++l) {
        int got = 0;
        for (int i = 0; i < Cfg::HWBITS; ++i) got |= (int)((hb[i] >> l) & 1) << i;
        if (got != R::weight(a[l])) okHw = false;
    }
    report("bitsliced multiply matches reference", okMul);
    report("bitsliced squaring matches reference", okSqr);
    report("bitsliced inversion matches reference", okInv);
    report("bitsliced sigma^7 matches reference", okSig);
    report("bitsliced Hamming weight matches reference", okHw);
}

template <class Cfg>
static void testOrbit(Rng &rng, const U192 &ell) {
    typedef Ref<Cfg> R;
    bool okFrob = true, okNeg = true, okWeight = true, okTrace = true;
    for (int t = 0; t < 8; ++t) {
        const typename R::Point p = randomPoint<Cfg>(rng, ell);
        const int hw = R::weight(p.x);
        const typename R::Point f = R::step(p, hw);
        const int c = (int)(rng.next() % Cfg::M);
        const typename R::Point pc = R::frob(p, c);
        if (!R::eq(R::step(pc, R::weight(pc.x)), R::frob(f, c))) okFrob = false;
        const typename R::Point pn = R::neg(p);
        if (!R::eq(R::step(pn, R::weight(pn.x)), R::neg(f))) okNeg = false;
        if (R::weight(pc.x) != hw) okWeight = false;
        if (R::trace(p.x) != 0) okTrace = false;
    }
    report("iteration commutes with Frobenius", okFrob);
    report("iteration commutes with negation", okNeg);
    report("weight is constant on an orbit", okWeight);
    report("subgroup x-coordinates have even weight", okTrace);
}

template <class Cfg>
static void testStartPoint(Rng &rng, Solver<Cfg> &sol) {
    typedef Ref<Cfg> R;
    typedef FieldBs<Cfg, u64> F;
    typedef Walk<Cfg, u64> WK;
    const int M = Cfg::M;
    u64 seeds[64];
    for (int l = 0; l < 64; ++l) seeds[l] = rng.next();
    std::vector<u64> x(M, 0), y(M, 0);
    CurveConsts K;
    K.px = sol.basis.x.v;
    K.py = sol.basis.y.v;
    K.qx = sol.target.x.v;
    K.qy = sol.target.y.v;
    WK::startPoint(seeds, K, x.data(), y.data());
    bool ok = true, okAlpha = true;
    for (int l = 0; l < 64; ++l) {
        typename R::Elem gx, gy;
        F::getLane(x.data(), l, gx.v);
        F::getLane(y.data(), l, gy.v);
        U192 alpha;
        const typename R::Point want = R::startPoint(seeds[l], sol.basis, sol.target, &alpha, sol.ell, sol.spow);
        if (!(gx == want.x) || !(gy == want.y)) ok = false;
        // the tracked scalar must satisfy start = [alpha] P + Q
        const typename R::Point chk = R::addPt(R::scalarMul(sol.basis, alpha), sol.target);
        if (!R::eq(chk, want)) okAlpha = false;
    }
    report("bitsliced start point matches reference", ok);
    report("tracked start scalar reproduces the point", okAlpha);
}

template <class Cfg>
static void testSolveAlgebra(Rng &rng, const Solver<Cfg> &base) {
    typedef Ref<Cfg> R;
    // Plant a discrete log of our own, synthesise two walks that meet with
    // known exponents, and check the solver recovers it.
    Solver<Cfg> sol = base;
    U192 knownK = u192_zero();
    knownK.v[0] = rng.next();
    knownK.v[1] = rng.next();
    knownK = mod_reduce(knownK, sol.ell);
    if (u192_is_zero(knownK)) knownK = u192_from(7);
    sol.target = R::scalarMul(sol.basis, knownK);
    bool ok = true;
    for (int t = 0; t < 4 && ok; ++t) {
        typename Solver<Cfg>::WalkResult A, B;
        A.ok = B.ok = true;
        for (int j = 0; j < 8; ++j) {
            A.counts[j] = rng.next() % 7;
            B.counts[j] = rng.next() % 7;
        }
        const U192 muA = sol.multiplier(A.counts), muB = sol.multiplier(B.counts);
        U192 beta = u192_zero();
        beta.v[0] = rng.next();
        beta.v[1] = rng.next() & 0xFFFF;
        beta = mod_reduce(beta, sol.ell);
        const int c = (int)(rng.next() % Cfg::M);
        const int eps = (rng.next() & 1) ? 1 : -1;
        U192 sc = sol.spow[c];
        if (eps < 0) sc = mod_neg(sc, sol.ell);
        // choose alpha so that  mu_A (alpha + k) = eps s^c mu_B (beta + k)
        const U192 rhs = mod_mul(mod_mul(sc, muB, sol.ell), mod_add(beta, knownK, sol.ell), sol.ell);
        const U192 alpha = mod_sub(mod_mul(rhs, mod_inv(muA, sol.ell), sol.ell), knownK, sol.ell);
        A.alpha0 = alpha;
        B.alpha0 = beta;
        A.endPoint = R::scalarMul(R::addPt(R::scalarMul(sol.basis, alpha), sol.target), muA);
        B.endPoint = R::scalarMul(R::addPt(R::scalarMul(sol.basis, beta), sol.target), muB);
        if (!R::eq(A.endPoint, eps > 0 ? R::frob(B.endPoint, c) : R::neg(R::frob(B.endPoint, c)))) {
            ok = false;
            break;
        }
        U192 got;
        std::string why;
        if (!sol.solve(A, B, &got, &why) || !u192_eq(got, knownK)) ok = false;
    }
    report("collision solver recovers a known discrete log", ok);
}

// ---------------------------------------------------------------------------
// search loop
// ---------------------------------------------------------------------------
template <class Cfg, class Engine>
static int runSearch(const Options &o, Engine &eng, Solver<Cfg> &sol, const U192 *knownK) {
    typedef Ref<Cfg> R;
    std::vector<DpRecord> recs;
    u64 iterBase = 0;
    u64 totalDp = 0, lost = 0, verified = 0, verifyBudget = (u64)(o.verify < 0 ? 0 : o.verify);
    const double t0 = nowSeconds();
    double lastPrint = t0;
    FILE *dpOut = o.dpFile.empty() ? NULL : fopen(o.dpFile.c_str(), "a");
    for (long launch = 0; o.launches == 0 || launch < o.launches; ++launch) {
        eng.launch(iterBase);
        const unsigned n = eng.fetch(recs);
        iterBase += (u64)o.steps;
        if (n) eng.reseed(iterBase);
        if (n > recs.size()) lost += n - recs.size();
        totalDp += recs.size();
        for (size_t i = 0; i < recs.size(); ++i) {
            const DpRecord &rec = recs[i];
            if (verifyBudget) {
                --verifyBudget;
                const typename Solver<Cfg>::WalkResult w = sol.rewalk(rec.seed);
                const bool same = w.ok && w.iters == rec.iters &&
                                  w.endPoint.x == R::fromLimbs(rec.x) &&
                                  w.endPoint.y == R::fromLimbs(rec.y);
                if (!same) {
                    printf("MISMATCH: seed %016llx was not reproduced by the reference walk\n",
                           (unsigned long long)rec.seed);
                    return 3;
                }
                ++verified;
            }
            if (dpOut) {
                const typename R::Elem cx = R::canonical(R::fromLimbs(rec.x));
                fprintf(dpOut, "%016llx %016llx\n", (unsigned long long)rec.seed,
                        (unsigned long long)R::hashPoint(cx));
            }
            typename Solver<Cfg>::Entry other;
            if (!sol.insert(rec, &other)) continue;
            printf("collision: seeds %016llx and %016llx meet after %llu and %llu steps\n",
                   (unsigned long long)rec.seed, (unsigned long long)other.seed,
                   (unsigned long long)rec.iters, (unsigned long long)other.iters);
            const double tr = nowSeconds();
            const typename Solver<Cfg>::WalkResult A = sol.rewalk(rec.seed);
            const typename Solver<Cfg>::WalkResult B = sol.rewalk(other.seed);
            U192 k;
            std::string why;
            if (!sol.solve(A, B, &k, &why)) {
                printf("  unusable (%s), continuing\n", why.c_str());
                continue;
            }
            printf("  recomputed both walks in %.2f s\n", nowSeconds() - tr);
            printf("  k = %s\n", u192_to_dec(k).c_str());
            printf("  verified [k]P == Q\n");
            if (knownK) printf("  matches the planted discrete log: %s\n",
                               u192_eq(k, *knownK) ? "yes" : "NO");
            const double el = nowSeconds() - t0;
            printf("  solved after %llu iterations of %llu walks in %.2f s (%llu distinguished points)\n",
                   (unsigned long long)iterBase, (unsigned long long)eng.walksPerLaunch(), el,
                   (unsigned long long)totalDp);
            if (dpOut) fclose(dpOut);
            return (knownK && !u192_eq(k, *knownK)) ? 4 : 0;
        }
        const double now = nowSeconds();
        if (now - lastPrint > 2.0 || (o.launches && launch + 1 == o.launches)) {
            const double el = now - t0;
            const double it = (double)iterBase * (double)eng.walksPerLaunch();
            printf("  %8.1f s  %10.3f M it/s  %10llu iterations  %8llu dp  %8llu stored\n",
                   el, it / el / 1e6, (unsigned long long)it, (unsigned long long)totalDp,
                   (unsigned long long)sol.inserted);
            fflush(stdout);
            lastPrint = now;
        }
    }
    const double el = nowSeconds() - t0;
    const double it = (double)iterBase * (double)eng.walksPerLaunch();
    printf("  finished: %.3f M it/s, %llu distinguished points (%llu verified against the reference, %llu dropped)\n",
           it / el / 1e6, (unsigned long long)totalDp, (unsigned long long)verified, (unsigned long long)lost);
    if (dpOut) fclose(dpOut);
    return 0;
}

// ---------------------------------------------------------------------------
template <class Cfg>
static int runCurve(const Options &oIn, const unsigned long long *px, const unsigned long long *py,
                    const unsigned long long *qx, const unsigned long long *qy, const char *ellDec,
                    const char *sDec, int defaultW, const char *knownKDec) {
    Options o = oIn;
    if (o.dpWeight < 0) o.dpWeight = defaultW;
    Solver<Cfg> sol;
    sol.setup(px, py, qx, qy, ellDec, sDec, o.dpWeight,
              o.maxIters ? o.maxIters : (u64)1 << 40);
    std::string why;
    if (!sol.checkSetup(&why)) {
        printf("parameter check failed: %s\n", why.c_str());
        return 5;
    }
    U192 knownK;
    bool haveK = false;
    if (knownKDec) {
        knownK = u192_from_dec(knownKDec);
        haveK = true;
    }

    if (o.test) {
        printf("GF(2^%d), n = %d, l = %s\n", Cfg::M, Cfg::NRING, ellDec);
        report("published parameters are consistent", true);
        Rng rng(0x1234567 + Cfg::M);
        testField<Cfg>(rng);
        testOrbit<Cfg>(rng, sol.ell);
        testStartPoint<Cfg>(rng, sol);
        testSolveAlgebra<Cfg>(rng, sol);
        return 0;
    }

#ifndef ECC_NO_CUDA
    CudaEngine<Cfg> eng;
#else
    HostEngine<Cfg> eng;
#endif
    eng.setup(o, px, py, qx, qy);
    printf("backend %s: %d threads x %d slots x %d lanes = %llu walks, dp weight %d, %d steps per launch\n",
           eng.name(), o.threads, (int)ECC_BATCH, (int)WordTraits<typename decltype(eng)::W>::LANES,
           (unsigned long long)eng.walksPerLaunch(), o.dpWeight, o.steps);
    return runSearch<Cfg>(o, eng, sol, haveK ? &knownK : NULL);
}

// ---------------------------------------------------------------------------
static void usage() {
    printf(
        "ecc2k130 - Pollard rho client for the Certicom ECC2K-130 challenge\n"
        "\n"
        "  --curve M        131 (the challenge), or 83 / 41 / 23 test curves\n"
        "  --instance I     use planted test instance I (small curves only)\n"
        "  --threads T      worker threads (device: total threads)\n"
        "  --steps S        iterations per launch (default 64)\n"
        "  --launches L     stop after L launches (0 = until solved)\n"
        "  --dp-weight W    distinguished point when the normal-basis weight is <= W\n"
        "  --max-iters N    restart a walk that has run N steps without a report\n"
        "  --run-id R       16-bit salt making seeds unique across processes\n"
        "  --verify N       recompute the first N reported points with the reference\n"
        "  --dp-file F      append (seed, hash) records to F\n"
        "  --bench          throughput only, no distinguished-point handling\n"
        "  --test           run the validation suite and exit\n"
        "  --device D       CUDA device index\n"
        "\n"
        "Build-time: ECC_BATCH=%d slots per thread, ECC_THREADS block size.\n",
        (int)ECC_BATCH);
}

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char *nx = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (a == "--curve" && nx) o.curve = atoi(argv[++i]);
        else if (a == "--instance" && nx) o.instance = atoi(argv[++i]);
        else if (a == "--threads" && nx) o.threads = atoi(argv[++i]);
        else if (a == "--steps" && nx) o.steps = atoi(argv[++i]);
        else if (a == "--launches" && nx) o.launches = atol(argv[++i]);
        else if (a == "--dp-weight" && nx) o.dpWeight = atoi(argv[++i]);
        else if (a == "--max-iters" && nx) o.maxIters = strtoull(argv[++i], NULL, 10);
        else if (a == "--run-id" && nx) o.runId = (unsigned)atoi(argv[++i]);
        else if (a == "--verify" && nx) o.verify = atoi(argv[++i]);
        else if (a == "--dp-cap" && nx) o.dpCap = (unsigned)atoi(argv[++i]);
        else if (a == "--dp-file" && nx) o.dpFile = argv[++i];
        else if (a == "--device" && nx) o.device = atoi(argv[++i]);
        else if (a == "--bench") o.bench = true;
        else if (a == "--test") o.test = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { printf("unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (o.threads <= 0) {
#ifndef ECC_NO_CUDA
        cudaDeviceProp prop;
        CUDA_CHECK(cudaSetDevice(o.device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
        o.threads = prop.multiProcessorCount * ECC_THREADS * 2;
        printf("device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
#else
#ifdef _OPENMP
        o.threads = omp_get_max_threads();
#else
        o.threads = 1;
#endif
#endif
    }
#ifndef ECC_NO_CUDA
    CUDA_CHECK(cudaSetDevice(o.device));
#endif
    if (o.bench) o.dpWeight = 0;

    if (o.test) {
        printf("ECC2K-130 validation suite\n");
        int rc = 0;
        rc |= runCurve<CfgF23>(o, eccF23::PX, eccF23::PY, eccF23::QX, eccF23::QY,
                               eccF23::ELL_DEC, eccF23::S_DEC, eccF23::DP_WEIGHT, NULL);
        rc |= runCurve<CfgF41>(o, eccF41::PX, eccF41::PY, eccF41::QX, eccF41::QY,
                               eccF41::ELL_DEC, eccF41::S_DEC, eccF41::DP_WEIGHT, NULL);
        rc |= runCurve<CfgF83>(o, eccF83::PX, eccF83::PY, eccF83::QX, eccF83::QY,
                               eccF83::ELL_DEC, eccF83::S_DEC, eccF83::DP_WEIGHT, NULL);
        rc |= runCurve<CfgF131>(o, eccF131::PX, eccF131::PY, eccF131::QX, eccF131::QY,
                                eccF131::ELL_DEC, eccF131::S_DEC, eccF131::DP_WEIGHT, NULL);
        printf("%s (%d failures)\n", gFail ? "VALIDATION FAILED" : "all checks passed", gFail);
        return (rc || gFail) ? 1 : 0;
    }

#define ECC_DISPATCH(NS, CFG)                                                                 \
    do {                                                                                      \
        const unsigned long long *px = NS::PX, *py = NS::PY, *qx = NS::QX, *qy = NS::QY;       \
        const char *kk = NULL;                                                                 \
        if (o.instance >= 0 && o.instance < NS::NUM_INSTANCES) {                              \
            px = NS::INSTANCE_PX[o.instance];                                                  \
            py = NS::INSTANCE_PY[o.instance];                                                  \
            qx = NS::INSTANCE_QX[o.instance];                                                  \
            qy = NS::INSTANCE_QY[o.instance];                                                  \
            kk = NS::INSTANCE_K[o.instance];                                                   \
        } else if (o.instance >= 0) {                                                          \
            printf("curve %d has %d planted instances\n", o.curve, NS::NUM_INSTANCES);        \
            return 1;                                                                          \
        }                                                                                      \
        return runCurve<CFG>(o, px, py, qx, qy, NS::ELL_DEC, NS::S_DEC, NS::DP_WEIGHT, kk);   \
    } while (0)

    if (o.curve == 131) ECC_DISPATCH(eccF131, CfgF131);
    if (o.curve == 83) ECC_DISPATCH(eccF83, CfgF83);
    if (o.curve == 41) ECC_DISPATCH(eccF41, CfgF41);
    if (o.curve == 23) ECC_DISPATCH(eccF23, CfgF23);
    printf("unsupported curve %d\n", o.curve);
    return 1;
}
