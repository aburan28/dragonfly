// The ECC2K-130 iteration function, bitsliced.
//
//     R_{i+1} = sigma^j(R_i) + R_i,     j = 3 + ((HW(x_{R_i}) / 2) mod 8)
//
// HW is the Hamming weight of the x coordinate in normal-basis representation,
// which is invariant under both sigma and negation, so the iteration is well
// defined on the orbits of size 2m and no canonical representative is ever
// needed inside the loop.
//
// Each thread owns BATCH slots; a slot is W walks in the bit lanes of one word.
// One iteration is two passes over the batch:
//
//   pass 1  weight, distinguished-point test, sigma^j(x), and the running
//           product of the addition denominators (Montgomery's trick)
//   pass 2  one inversion for the whole batch, then the affine additions
//
// Pass 2 recomputes sigma^j(x) and sigma^j(y) from the three stored weight bits
// rather than reading stored denominators back: on current hardware this loop
// is close to memory bound, so trading 786 instructions for 1048 bytes of
// traffic per slot is worth it.
#pragma once

#include "fieldbs.h"

struct DpRecord {
    unsigned long long seed;
    unsigned long long iters;
    unsigned long long x[3];
    unsigned long long y[3];
};

ECC_HD unsigned long long eccPrf(unsigned long long seed, int idx) {
    unsigned long long z = seed + 0x9E3779B97F4A7C15ull * (unsigned long long)(idx + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// walk index -> initial seed;  a walk that finishes just increments its seed
ECC_HD unsigned long long eccSeedFor(unsigned runId, unsigned long long walkIndex) {
    return ((unsigned long long)runId << 48) | ((walkIndex & 0xFFFFFFFFull) << 16);
}

// Curve constants live in device memory and are passed in, so the generated
// tables need no __constant__ plumbing.
struct CurveConsts {
    const unsigned long long *px;
    const unsigned long long *py;
    const unsigned long long *qx;
    const unsigned long long *qy;
};

template <class Cfg, class W>
struct Walk {
    typedef FieldBs<Cfg, W> F;
    static const int M = Cfg::M;
    static const int HWBITS = Cfg::HWBITS;
    static const int LANES = WordTraits<W>::LANES;
    static const int STARTTERMS = 128;

    // ---- weight and distinguished-point test ---------------------------
    static ECC_HD void hamming(const W *x, W *hb) { Cfg::hamming(x, hb); }

    // Mask of the lanes whose weight is at most w.  Scanning from the top,
    // `eq` stays set while the weight matches w bit for bit and `gt` latches as
    // soon as the weight is decided larger; the answer is ~gt.
    static ECC_HD W dpMask(const W *hb, int w) {
        W gt = ECC_ZERO;
        W eq = ~ECC_ZERO;
        for (int i = HWBITS - 1; i >= 0; --i) {
            const W h = hb[i];
            if ((w >> i) & 1) {
                eq = eq & h;
            } else {
                gt = gt | (eq & h);
                eq = eq & ~h;
            }
        }
        return ~gt;
    }

    // ---- sigma^j, j = 3 + b0 + 2 b1 + 4 b2, branch free ----------------
    static ECC_BIG void sigmaJ(const W *a, const W *hb, W *out) {
        W r[M], s[M];
        F::template sigma<3>(a, r);
        F::template sigma<1>(r, s);
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = ECC_SEL(hb[1], s[i], r[i]);
        F::template sigma<2>(r, s);
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = ECC_SEL(hb[2], s[i], r[i]);
        F::template sigma<4>(r, s);
#pragma unroll
        for (int i = 0; i < M; ++i) out[i] = ECC_SEL(hb[3], s[i], r[i]);
    }

    // ---- one affine addition R = R + S (all lanes) ---------------------
    static ECC_BIG void pointAdd(const W *x1, const W *y1, const W *x2, const W *y2, W *x3, W *y3) {
        W d[M], e[M], di[M], lam[M], t[M];
        F::add(x1, x2, d);
        F::add(y1, y2, e);
        F::inv(d, di);
        F::mul(e, di, lam);
        F::sqr(lam, t);
#pragma unroll
        for (int i = 0; i < M; ++i) t[i] = ECC_XOR3(t[i], lam[i], d[i]);   // x3 = lam^2 + lam + d
        F::add(x1, t, d);                                                  // d = x1 + x3
        F::mul(lam, d, e);
#pragma unroll
        for (int i = 0; i < M; ++i) y3[i] = ECC_XOR3(e[i], t[i], y1[i]);
        F::copy(x3, t);
    }

    // ---- fresh starting point:  R = Q + sum_i c_i sigma^i(P) -----------
    // c is a 128-bit string derived from the walk seed, per lane.
    static ECC_BIG void startPoint(const unsigned long long *seeds, const CurveConsts &K, W *x, W *y) {
        const int NG = LANES / 64 > 0 ? LANES / 64 : 1;
        unsigned long long cw[STARTTERMS][(LANES / 64) > 0 ? (LANES / 64) : 1];
#pragma unroll 1
        for (int i = 0; i < STARTTERMS; ++i)
            for (int g = 0; g < NG; ++g) cw[i][g] = 0;
        for (int lane = 0; lane < LANES; ++lane) {
            const unsigned long long c0 = eccPrf(seeds[lane], 0);
            const unsigned long long c1 = eccPrf(seeds[lane], 1);
            const int g = (LANES >= 64) ? (lane >> 6) : 0;
            const int b = (LANES >= 64) ? (lane & 63) : lane;
            for (int i = 0; i < STARTTERMS; ++i) {
                const unsigned long long bit = (i < 64) ? (c0 >> i) : (c1 >> (i - 64));
                cw[i][g] |= (bit & 1ull) << b;
            }
        }
        W cbit[STARTTERMS];
#pragma unroll 1
        for (int i = 0; i < STARTTERMS; ++i) cbit[i] = eccWordFromLimbs<W>(cw[i]);
        F::broadcast(K.qx, x);
        F::broadcast(K.qy, y);
        W sx[M], sy[M], ax[M], ay[M];
        for (int i = 0; i < STARTTERMS; ++i) {
            F::broadcastSigma(K.px, i % M, sx);
            F::broadcastSigma(K.py, i % M, sy);
            pointAdd(x, y, sx, sy, ax, ay);
#pragma unroll
            for (int t = 0; t < M; ++t) {
                x[t] = ECC_SEL(cbit[i], ax[t], x[t]);
                y[t] = ECC_SEL(cbit[i], ay[t], y[t]);
            }
        }
    }

    // replace only the lanes selected by `mask` with fresh walks
    static ECC_BIG void reseedLanes(W mask, unsigned long long *seeds, const CurveConsts &K, W *x, W *y) {
        W nx[M], ny[M];
        startPoint(seeds, K, nx, ny);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            x[i] = ECC_SEL(mask, nx[i], x[i]);
            y[i] = ECC_SEL(mask, ny[i], y[i]);
        }
    }
};
