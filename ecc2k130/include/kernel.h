// Batched walk driver shared by the CUDA kernel and the CPU backend.
//
// State layout is structure-of-arrays indexed by thread so that lanes of a warp
// touch consecutive words:
//
//     x[(slot * M + word) * threads + tid]
//
// Only x, y and the Montgomery product chain move through memory each
// iteration; the addition denominators are recomputed in pass 2 from three
// stored weight bits.
#pragma once

#include "walk.h"

#ifndef ECC_BATCH
#define ECC_BATCH 16
#endif

#ifndef ECC_GUARD_PERIOD
#define ECC_GUARD_PERIOD 4096
#endif

template <class W>
struct WalkParams {
    int threads;
    int steps;
    int dpWeight;
    unsigned runId;
    unsigned long long maxIters;      // 0 disables the cycle guard
    unsigned long long iterBase;
    W *x;
    W *y;
    W *pchain;
    unsigned long long *seed;         // one per lane
    unsigned long long *startIter;    // one per lane
    W *dead;                          // one word per slot: lanes awaiting a restart
    DpRecord *dp;
    unsigned *dpCount;
    unsigned dpCap;
    CurveConsts consts;
};

ECC_HD unsigned eccAtomicInc(unsigned *p) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(p, 1u);
#elif defined(_OPENMP) || defined(ECC_HOST_ATOMIC)
    return __sync_fetch_and_add(p, 1u);
#else
    unsigned v = *p;
    *p = v + 1;
    return v;
#endif
}

template <class Cfg, class W>
struct Kernel {
    typedef Walk<Cfg, W> WK;
    typedef FieldBs<Cfg, W> F;
    static const int M = Cfg::M;
    static const int LANES = WordTraits<W>::LANES;
    static const int HWBITS = Cfg::HWBITS;
    static const int BATCH = ECC_BATCH;

    static ECC_HD size_t fieldIndex(int slot, int word, int tid, int threads) {
        return ((size_t)(slot * M + word) * (size_t)threads) + (size_t)tid;
    }
    static ECC_HD size_t laneIndex(int slot, int lane, int tid, int threads) {
        return ((size_t)(slot * LANES + lane) * (size_t)threads) + (size_t)tid;
    }

    static ECC_HD void load(const W *src, int slot, int tid, int threads, W *dst) {
#pragma unroll
        for (int i = 0; i < M; ++i) dst[i] = src[fieldIndex(slot, i, tid, threads)];
    }
    static ECC_HD void store(W *dst, int slot, int tid, int threads, const W *src) {
#pragma unroll
        for (int i = 0; i < M; ++i) dst[fieldIndex(slot, i, tid, threads)] = src[i];
    }

    // ---- start every walk of this thread ------------------------------
    static ECC_HD void init(int tid, const WalkParams<W> &P) {
        W x[M], y[M];
        unsigned long long seeds[LANES];
        for (int slot = 0; slot < BATCH; ++slot) {
            for (int lane = 0; lane < LANES; ++lane) {
                const unsigned long long walkIndex =
                    ((unsigned long long)tid * BATCH + slot) * LANES + lane;
                seeds[lane] = eccSeedFor(P.runId, walkIndex);
                P.seed[laneIndex(slot, lane, tid, P.threads)] = seeds[lane];
                P.startIter[laneIndex(slot, lane, tid, P.threads)] = 0;
            }
            WK::startPoint(seeds, P.consts, x, y);
            store(P.x, slot, tid, P.threads, x);
            store(P.y, slot, tid, P.threads, y);
            P.dead[(size_t)slot * (size_t)P.threads + (size_t)tid] = ECC_ZERO;
        }
    }

    // Report every lane flagged in `mask` and mark it for restart.  Restarting
    // is deferred to the reseed kernel: computing a fresh start point needs 128
    // point additions, and keeping that call chain out of the hot kernel is
    // worth 5.8 KB of per-thread stack frame.  A marked lane keeps walking, but
    // its reports are suppressed until it is revived.
    static ECC_HD void handleDistinguished(int tid, int slot, W mask, const WalkParams<W> &P,
                                           unsigned long long now, const W *x, const W *y) {
        for (int lane = 0; lane < LANES; ++lane) {
            if (!laneBit(mask, lane)) continue;
            const size_t li = laneIndex(slot, lane, tid, P.threads);
            DpRecord rec;
            rec.seed = P.seed[li];
            rec.iters = now - P.startIter[li];
            F::getLane(x, lane, rec.x);
            F::getLane(y, lane, rec.y);
            const unsigned slotIdx = eccAtomicInc(P.dpCount);
            if (slotIdx < P.dpCap) P.dp[slotIdx] = rec;
        }
    }

    // ---- revive every marked lane (rare; launched between walk kernels) ----
    static ECC_HD void reseed(int tid, const WalkParams<W> &P) {
        W x[M], y[M];
        unsigned long long seeds[LANES];
        for (int slot = 0; slot < BATCH; ++slot) {
            const size_t di = (size_t)slot * (size_t)P.threads + (size_t)tid;
            const W mask = P.dead[di];
            if (mask == ECC_ZERO) continue;
            load(P.x, slot, tid, P.threads, x);
            load(P.y, slot, tid, P.threads, y);
            for (int lane = 0; lane < LANES; ++lane) {
                const size_t li = laneIndex(slot, lane, tid, P.threads);
                unsigned long long sd = P.seed[li];
                if (laneBit(mask, lane)) {
                    sd += 1;
                    P.seed[li] = sd;
                    P.startIter[li] = P.iterBase;
                }
                seeds[lane] = sd;
            }
            WK::reseedLanes(mask, seeds, P.consts, x, y);
            store(P.x, slot, tid, P.threads, x);
            store(P.y, slot, tid, P.threads, y);
            P.dead[di] = ECC_ZERO;
        }
    }

    // ---- one full iteration over the batch ----------------------------
    static ECC_HD void run(int tid, const WalkParams<W> &P) {
        W x[M], y[M], t[M], u[M], prod[M], inv[M];
        W hb[HWBITS < 4 ? 4 : HWBITS];
        W jbits[BATCH][4];

#pragma unroll 1
        for (int step = 0; step < P.steps; ++step) {
            const unsigned long long now = P.iterBase + (unsigned long long)step;
            const bool guard = P.maxIters != 0 && (now % ECC_GUARD_PERIOD) == 0;

#pragma unroll 1
            for (int slot = 0; slot < BATCH; ++slot) {
                load(P.x, slot, tid, P.threads, x);
                const size_t di = (size_t)slot * (size_t)P.threads + (size_t)tid;
                WK::hamming(x, hb);
                W dp = WK::dpMask(hb, P.dpWeight);
                if (guard) {
                    for (int lane = 0; lane < LANES; ++lane) {
                        const size_t li = laneIndex(slot, lane, tid, P.threads);
                        if (now - P.startIter[li] >= P.maxIters) dp |= laneMask<W>(lane);
                    }
                }
                const W alreadyDead = P.dead[di];
                dp &= ~alreadyDead;
                if (dp != ECC_ZERO) {
                    load(P.y, slot, tid, P.threads, y);
                    handleDistinguished(tid, slot, dp, P, now, x, y);
                    P.dead[di] = alreadyDead | dp;
                }
                jbits[slot][1] = hb[1];
                jbits[slot][2] = hb[2];
                jbits[slot][3] = hb[3];
                WK::sigmaJ(x, jbits[slot], t);
                F::add(x, t, u);                       // u = d
                if (slot == 0) {
                    F::copy(prod, u);
                } else {
                    F::mul(prod, u, t);
                    F::copy(prod, t);
                }
                store(P.pchain, slot, tid, P.threads, prod);
            }

            F::inv(prod, inv);

#pragma unroll 1
            for (int slot = BATCH - 1; slot >= 0; --slot) {
                load(P.x, slot, tid, P.threads, x);
                load(P.y, slot, tid, P.threads, y);
                W d[M], e[M], ii[M], lam[M];
                WK::sigmaJ(x, jbits[slot], t);
                F::add(x, t, d);
                WK::sigmaJ(y, jbits[slot], t);
                F::add(y, t, e);
                if (slot > 0) {
                    load(P.pchain, slot - 1, tid, P.threads, u);
                    F::mul(inv, u, ii);
                    F::mul(inv, d, t);
                    F::copy(inv, t);
                } else {
                    F::copy(ii, inv);
                }
                F::mul(e, ii, lam);
                F::sqr(lam, t);
#pragma unroll
                for (int i = 0; i < M; ++i) t[i] = ECC_XOR3(t[i], lam[i], d[i]);   // x3
                F::add(x, t, u);
                F::mul(lam, u, e);
#pragma unroll
                for (int i = 0; i < M; ++i) y[i] = ECC_XOR3(e[i], t[i], y[i]);
                store(P.x, slot, tid, P.threads, t);
                store(P.y, slot, tid, P.threads, y);
            }
        }
    }
};

#if defined(__CUDACC__)
#ifndef ECC_THREADS
#define ECC_THREADS 128
#endif
template <class Cfg, class W>
__global__ void __launch_bounds__(ECC_THREADS) eccWalkKernel(WalkParams<W> P) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.threads) return;
    Kernel<Cfg, W>::run(tid, P);
}
template <class Cfg, class W>
__global__ void __launch_bounds__(ECC_THREADS) eccInitKernel(WalkParams<W> P) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.threads) return;
    Kernel<Cfg, W>::init(tid, P);
}
template <class Cfg, class W>
__global__ void __launch_bounds__(ECC_THREADS) eccReseedKernel(WalkParams<W> P) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.threads) return;
    Kernel<Cfg, W>::reseed(tid, P);
}
#endif
