// Independent host-side reference implementation.
//
// This deliberately shares no code with the generated bitsliced arithmetic: an
// element is the symmetric vector over Z/n it stands for, and multiplication is
// the plain cyclic convolution that defines the type-II normal basis, rather
// than the Bernstein-Lange polynomial-basis route the fast path takes.  It is
// used to differential-test the fast path, to recompute walks when the server
// resolves a collision, and to verify a recovered discrete logarithm.
#pragma once

#include <stdint.h>
#include <string.h>
#include <string>
#include "bigmod.h"

typedef unsigned long long u64;

template <class Cfg>
struct Ref {
    static const int M = Cfg::M;
    static const int NRING = Cfg::NRING;
    static const int NL = 3;                    // limbs for m-bit coordinates
    static const int SL = (Cfg::NRING + 63) / 64;   // limbs for the n-bit lift

    struct Elem {
        u64 v[NL];
        bool operator==(const Elem &o) const {
            for (int i = 0; i < NL; ++i)
                if (v[i] != o.v[i]) return false;
            return true;
        }
        bool operator!=(const Elem &o) const { return !(*this == o); }
    };

    static Elem zero() { Elem e; memset(e.v, 0, sizeof e.v); return e; }
    static bool isZero(const Elem &a) {
        for (int i = 0; i < NL; ++i)
            if (a.v[i]) return false;
        return true;
    }
    static Elem one() {
        Elem e = zero();
        for (int i = 0; i < M; ++i) e.v[i >> 6] |= 1ull << (i & 63);
        return e;
    }
    static Elem add(const Elem &a, const Elem &b) {
        Elem r;
        for (int i = 0; i < NL; ++i) r.v[i] = a.v[i] ^ b.v[i];
        return r;
    }
    static int bit(const Elem &a, int i) { return (int)((a.v[i >> 6] >> (i & 63)) & 1); }
    static void setBit(Elem &a, int i) { a.v[i >> 6] |= 1ull << (i & 63); }
    static int weight(const Elem &a) {
        int w = 0;
        for (int i = 0; i < NL; ++i) w += __builtin_popcountll(a.v[i]);
        return w;
    }
    static Elem fromLimbs(const unsigned long long *p) {
        Elem e;
        for (int i = 0; i < NL; ++i) e.v[i] = p[i];
        return e;
    }

    // ---- lift to the symmetric n-bit vector and back -------------------
    struct Lift { u64 v[SL]; };

    static Lift lift(const Elem &a) {
        Lift u;
        memset(u.v, 0, sizeof u.v);
        for (int i = 1; i <= M; ++i) {
            if (bit(a, i - 1)) {
                u.v[i >> 6] |= 1ull << (i & 63);
                const int j = NRING - i;
                u.v[j >> 6] |= 1ull << (j & 63);
            }
        }
        return u;
    }
    static Elem unlift(Lift u) {
        if (u.v[0] & 1) {                      // normalise: add the all-ones vector
            for (int i = 0; i < NRING; ++i) u.v[i >> 6] ^= 1ull << (i & 63);
        }
        Elem e = zero();
        for (int i = 1; i <= M; ++i)
            if ((u.v[i >> 6] >> (i & 63)) & 1) setBit(e, i - 1);
        return e;
    }
    static Lift rot(const Lift &u, int k) {
        Lift r;
        memset(r.v, 0, sizeof r.v);
        k %= NRING;
        for (int i = 0; i < NRING; ++i) {
            if ((u.v[i >> 6] >> (i & 63)) & 1) {
                const int j = (i + k) % NRING;
                r.v[j >> 6] |= 1ull << (j & 63);
            }
        }
        return r;
    }

    static Elem mul(const Elem &a, const Elem &b) {
        const Lift ua = lift(a);
        const Lift ub = lift(b);
        Lift acc;
        memset(acc.v, 0, sizeof acc.v);
        for (int i = 0; i < NRING; ++i) {
            if (!((ub.v[i >> 6] >> (i & 63)) & 1)) continue;
            const Lift t = rot(ua, i);
            for (int j = 0; j < SL; ++j) acc.v[j] ^= t.v[j];
        }
        return unlift(acc);
    }

    // squaring is the coordinate permutation i -> fold(2i)
    static Elem sigma(const Elem &a, int k) {
        int e = 1;
        for (int j = 0; j < k % M; ++j) e = (2 * e) % NRING;
        Elem r = zero();
        for (int i = 1; i <= M; ++i) {
            if (!bit(a, i - 1)) continue;
            int t = (int)((long long)i * e % NRING);
            if (t > M) t = NRING - t;
            setBit(r, t - 1);
        }
        return r;
    }
    static Elem sqr(const Elem &a) { return sigma(a, 1); }

    // Itoh-Tsujii; squarings are free so this is 8 multiplications for m=131
    static Elem inv(const Elem &a) {
        Elem acc = a;
        int k = 1;
        const int e = M - 1;
        int hb = 0;
        while ((1 << (hb + 1)) <= e) ++hb;
        for (int b = hb - 1; b >= 0; --b) {
            acc = mul(sigma(acc, k), acc);
            k <<= 1;
            if ((e >> b) & 1) {
                acc = mul(sqr(acc), a);
                k += 1;
            }
        }
        return sqr(acc);
    }
    static Elem pow(const Elem &a, u64 e) {
        Elem r = one(), b = a;
        while (e) {
            if (e & 1) r = mul(r, b);
            b = mul(b, b);
            e >>= 1;
        }
        return r;
    }
    static int trace(const Elem &a) { return weight(a) & 1; }

    // ---- polynomial-basis interoperability -----------------------------
    static Elem fromPolyBasis(const unsigned long long *pb, const unsigned long long ztab[][3]) {
        Elem r = zero();
        for (int i = 0; i < M; ++i) {
            if ((pb[i >> 6] >> (i & 63)) & 1) {
                for (int l = 0; l < NL; ++l) r.v[l] ^= ztab[i][l];
            }
        }
        return r;
    }
    static void toPolyBasis(const Elem &a, const unsigned long long gtab[][3], unsigned long long *out) {
        out[0] = out[1] = out[2] = 0;
        for (int i = 0; i < M; ++i) {
            if (bit(a, i)) {
                for (int l = 0; l < NL; ++l) out[l] ^= gtab[i][l];
            }
        }
    }

    // ---- curve ---------------------------------------------------------
    struct Point { Elem x, y; bool inf; };
    static Point infinity() { Point p; p.x = zero(); p.y = zero(); p.inf = true; return p; }
    static Point make(const Elem &x, const Elem &y) { Point p; p.x = x; p.y = y; p.inf = false; return p; }
    static bool eq(const Point &a, const Point &b) {
        if (a.inf || b.inf) return a.inf == b.inf;
        return a.x == b.x && a.y == b.y;
    }
    static bool onCurve(const Point &p) {
        if (p.inf) return true;
        return add(mul(p.y, p.y), mul(p.x, p.y)) == add(mul(mul(p.x, p.x), p.x), one());
    }
    static Point neg(const Point &p) { return p.inf ? p : make(p.x, add(p.x, p.y)); }
    static Point dbl(const Point &p) {
        if (p.inf || isZero(p.x)) return infinity();
        const Elem lam = add(p.x, mul(p.y, inv(p.x)));
        const Elem x3 = add(mul(lam, lam), lam);
        const Elem y3 = add(mul(p.x, p.x), mul(add(lam, one()), x3));
        return make(x3, y3);
    }
    static Point addPt(const Point &a, const Point &b) {
        if (a.inf) return b;
        if (b.inf) return a;
        if (a.x == b.x) return (a.y == b.y) ? dbl(a) : infinity();
        const Elem d = add(a.x, b.x);
        const Elem lam = mul(add(a.y, b.y), inv(d));
        const Elem x3 = add(add(mul(lam, lam), lam), d);
        const Elem y3 = add(add(mul(lam, add(a.x, x3)), x3), a.y);
        return make(x3, y3);
    }
    static Point frob(const Point &p, int k) {
        if (p.inf) return p;
        return make(sigma(p.x, k), sigma(p.y, k));
    }
    static Point scalarMul(const Point &p, const U192 &k) {
        Point r = infinity();
        for (int i = u192_bits(k) - 1; i >= 0; --i) {
            r = dbl(r);
            if (u192_bit(k, i)) r = addPt(r, p);
        }
        return r;
    }
    static Elem halfTrace(const Elem &a) {
        Elem acc = a, t = a;
        for (int i = 1; i <= (M - 1) / 2; ++i) {
            t = sigma(t, 2);
            acc = add(acc, t);
        }
        return acc;
    }

    // ---- the iteration function ----------------------------------------
    static int jOf(int hw) { return 3 + ((hw >> 1) & 7); }
    static Point step(const Point &p, int hw) { return addPt(p, frob(p, jOf(hw))); }

    // canonical representative of the orbit under sigma (negation leaves x fixed)
    static Elem canonical(const Elem &x) {
        Elem best = x, cur = x;
        for (int k = 1; k < M; ++k) {
            cur = sqr(cur);
            for (int i = NL - 1; i >= 0; --i) {
                if (cur.v[i] < best.v[i]) { best = cur; break; }
                if (cur.v[i] > best.v[i]) break;
            }
        }
        return best;
    }
    static u64 hashPoint(const Elem &canonX) {
        u64 h = 0xcbf29ce484222325ull;
        for (int i = 0; i < NL; ++i) {
            h ^= canonX.v[i];
            h *= 0x100000001b3ull;
        }
        return h ^ (h >> 29);
    }

    // start point Q + sum c_i sigma^i(P), matching the device derivation
    static Point startPoint(u64 seed, const Point &basis, const Point &target, U192 *alphaOut, const U192 &ell, const U192 *spow) {
        const u64 c0 = eccPrfHost(seed, 0);
        const u64 c1 = eccPrfHost(seed, 1);
        Point r = target;
        U192 alpha = u192_zero();
        for (int i = 0; i < 128; ++i) {
            const u64 b = (i < 64) ? (c0 >> i) : (c1 >> (i - 64));
            if (b & 1) {
                r = addPt(r, frob(basis, i % M));
                alpha = mod_add(alpha, spow[i % M], ell);
            }
        }
        if (alphaOut) *alphaOut = alpha;
        return r;
    }

    static u64 eccPrfHost(u64 seed, int idx) {
        u64 z = seed + 0x9E3779B97F4A7C15ull * (u64)(idx + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};
