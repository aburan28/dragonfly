// Bitsliced GF(2^m) arithmetic in the permuted type-II optimal normal basis.
//
// An element is M words, word i holding the coefficient of gamma_{i+1} for each
// of the W lanes.  Squaring is the index permutation i -> fold(2i), which the
// compiler resolves at code-generation time when the loop is unrolled, so
// squarings and the whole sigma^j sequence cost no arithmetic at all.
//
// Multiplication follows Bernstein-Lange: convert both operands to the optimal
// polynomial basis (multPrep, generated), one 131x131 polynomial multiplication
// over GF(2), then convert the 261-coefficient product back (toOnb, generated).
// The polynomial multiplication is Karatsuba, recursing on the size chain in
// the generated header down to a leaf that is itself a generated straight-line
// DAG.  Keeping the leaf a real function is what holds the instruction cache
// footprint to a few hundred instructions instead of the ~9000 a fully inlined
// multiplication would need.
#pragma once

#include "bitslice.h"

// ---------------------------------------------------------------------------
// Karatsuba recursion.  N is the operand length in words; the result has
// 2N-1 words.  Operand halves are ceil(N/2), the upper half zero padded.
// ---------------------------------------------------------------------------
template <class Cfg, class W, int N>
struct Karat {
    static const int H = (N + 1) / 2;
    static const int SUB = 2 * H - 1;

    static ECC_HD void mul(const W *a, const W *b, W *r) {
        W am[H], bm[H];
        W a1[H], b1[H];
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const W ah = (H + i < N) ? a[H + i] : ECC_ZERO;
            const W bh = (H + i < N) ? b[H + i] : ECC_ZERO;
            a1[i] = ah;
            b1[i] = bh;
            am[i] = a[i] ^ ah;
            bm[i] = b[i] ^ bh;
        }
        W p0[SUB], p1[SUB], pm[SUB];
        Karat<Cfg, W, H>::mul(a, b, p0);
        Karat<Cfg, W, H>::mul(a1, b1, p1);
        Karat<Cfg, W, H>::mul(am, bm, pm);
#pragma unroll
        for (int i = 0; i < 2 * N - 1; ++i) r[i] = ECC_ZERO;
#pragma unroll
        for (int i = 0; i < SUB; ++i) {
            if (i < 2 * N - 1) r[i] ^= p0[i];
        }
#pragma unroll
        for (int i = 0; i < SUB; ++i) {
            if (H + i < 2 * N - 1) r[H + i] ^= ECC_XOR3(pm[i], p0[i], p1[i]);
        }
#pragma unroll
        for (int i = 0; i < SUB; ++i) {
            if (2 * H + i < 2 * N - 1) r[2 * H + i] ^= p1[i];
        }
    }
};

// The leaf is the generated straight-line multiplier.
#define ECC_DEFINE_LEAF(CFG, LEAFSIZE)                                       \
    template <class W>                                                       \
    struct Karat<CFG, W, LEAFSIZE> {                                         \
        static ECC_HD void mul(const W *a, const W *b, W *r) {                \
            CFG::mulLeaf(a, b, r);                                            \
        }                                                                     \
    };

// ---------------------------------------------------------------------------
template <class Cfg, class W>
struct FieldBs {
    static const int M = Cfg::M;
    static const int NRING = Cfg::NRING;
    static const int PRODLEN = 2 * Cfg::M - 1;
    static const int LANES = WordTraits<W>::LANES;

    static ECC_HD void setZero(W *r) {
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = ECC_ZERO;
    }

    static ECC_HD void copy(W *r, const W *a) {
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = a[i];
    }

    static ECC_HD void add(const W *a, const W *b, W *r) {
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = a[i] ^ b[i];
    }

    static ECC_HD void addTo(W *r, const W *a) {
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] ^= a[i];
    }

    // 1 = sum of all basis elements
    static ECC_HD void setOne(W *r) {
#pragma unroll
        for (int i = 0; i < M; ++i) r[i] = ~ECC_ZERO;
    }

    static ECC_HD bool isZeroLane(const W *a, int lane) {
        W acc = ECC_ZERO;
#pragma unroll
        for (int i = 0; i < M; ++i) acc |= a[i];
        return laneBit(acc, lane) == 0;
    }

    // ---- sigma (Frobenius) -------------------------------------------
    // The coefficient of gamma_{i+1} moves to gamma_{fold(2^k (i+1))}.
    static constexpr int pow2Mod(int k) {
        int e = 1;
        for (int j = 0; j < k; ++j) e = (2 * e) % NRING;
        return e;
    }
    static constexpr int foldIdx(int t) { return t <= M ? t : NRING - t; }
    static constexpr int sigmaDest(int i, int e) {
        return foldIdx((int)(((long long)(i + 1) * (long long)e) % (long long)NRING)) - 1;
    }

    template <int K>
    static ECC_HD void sigma(const W *a, W *r) {
        constexpr int e = pow2Mod(K);
#pragma unroll
        for (int i = 0; i < M; ++i) r[sigmaDest(i, e)] = a[i];
    }

    // runtime exponent, used by the inversion chain and by the host
    static ECC_HD void sigmaRun(const W *a, int k, W *r) {
        int e = 1;
        for (int j = 0; j < k; ++j) e = (2 * e) % NRING;
        for (int i = 0; i < M; ++i) {
            long long t = (long long)(i + 1) * (long long)e % (long long)NRING;
            int d = (int)(t <= M ? t : NRING - t) - 1;
            r[d] = a[i];
        }
    }

    static ECC_HD void sqr(const W *a, W *r) { sigma<1>(a, r); }

    // ---- multiplication -----------------------------------------------
    static ECC_BIG void mul(const W *a, const W *b, W *r) {
        W pa[M], pb[M], h[PRODLEN];
        Cfg::multPrep(a, pa);
        Cfg::multPrep(b, pb);
        Karat<Cfg, W, Cfg::M>::mul(pa, pb, h);
        Cfg::toOnb(h, r);
    }

    static ECC_HD void sqrMul(const W *a, const W *b, W *r) { mul(a, b, r); }

    // ---- inversion: Itoh-Tsujii, squarings are free -------------------
    // a^(2^m - 2) = (a^(2^(m-1) - 1))^2 with an addition chain for m-1.
    static ECC_BIG void inv(const W *a, W *r) {
        W acc[M], t[M], u[M];
        copy(acc, a);
        int k = 1;
        const int e = M - 1;
        int hb = 0;
        while ((1 << (hb + 1)) <= e) ++hb;
        for (int bit = hb - 1; bit >= 0; --bit) {
            sigmaRun(acc, k, t);
            mul(t, acc, u);
            copy(acc, u);
            k <<= 1;
            if ((e >> bit) & 1) {
                sigmaRun(acc, 1, t);
                mul(t, a, u);
                copy(acc, u);
                k += 1;
            }
        }
        sigmaRun(acc, 1, r);
    }

    // ---- lane access (rare paths only) --------------------------------
    static ECC_HD void getLane(const W *a, int lane, unsigned long long *out3) {
        out3[0] = out3[1] = out3[2] = 0;
        for (int i = 0; i < M; ++i) {
            if (laneBit(a[i], lane)) out3[i >> 6] |= 1ull << (i & 63);
        }
    }

    static ECC_HD void setLane(W *a, int lane, const unsigned long long *in3) {
        const W m1 = laneMask<W>(lane);
        for (int i = 0; i < M; ++i) {
            const W bit = ((in3[i >> 6] >> (i & 63)) & 1ull) ? m1 : ECC_ZERO;
            a[i] = (a[i] & ~m1) | bit;
        }
    }

    // broadcast a scalar element (same value in every lane)
    static ECC_HD void broadcast(const unsigned long long *in3, W *a) {
#pragma unroll
        for (int i = 0; i < M; ++i) {
            a[i] = ((in3[i >> 6] >> (i & 63)) & 1ull) ? ~ECC_ZERO : ECC_ZERO;
        }
    }

    // broadcast sigma^k of a scalar element, without materialising the source
    static ECC_HD void broadcastSigma(const unsigned long long *in3, int k, W *a) {
        int e = 1;
        for (int j = 0; j < k; ++j) e = (2 * e) % NRING;
        for (int i = 0; i < M; ++i) {
            long long t = (long long)(i + 1) * (long long)e % (long long)NRING;
            int d = (int)(t <= M ? t : NRING - t) - 1;
            a[d] = ((in3[i >> 6] >> (i & 63)) & 1ull) ? ~ECC_ZERO : ECC_ZERO;
        }
    }
};
