// Distinguished-point store and collision resolution.
//
// Clients report only (seed, endpoint); no linear combination of P and Q is
// tracked inside the walk, exactly as in the ECC2K-130 design.  When two seeds
// reach the same orbit the server recomputes both walks, this time counting how
// often each sigma^j + 1 was applied, which gives
//
//     endpoint = [mu] (alpha0 P + Q),   mu = prod_j (1 + s^j)^{n_j}
//
// so endpoint = a P + b Q with a = mu alpha0 and b = mu.  Matching the two
// endpoints up to Frobenius and negation, endA = eps sigma^c endB, yields
//
//     k = (a_A - eps s^c a_B) / (eps s^c b_B - b_A)   mod l
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "ref.h"
#include "walk.h"

template <class Cfg>
struct Solver {
    typedef Ref<Cfg> R;
    typedef typename R::Elem Elem;
    typedef typename R::Point Point;
    static const int M = Cfg::M;

    U192 ell;
    U192 s;
    U192 spow[256];
    Point basis;
    Point target;
    int dpWeight;
    unsigned long long maxIters;

    void setup(const unsigned long long *px, const unsigned long long *py,
               const unsigned long long *qx, const unsigned long long *qy,
               const char *ellDec, const char *sDec, int w, unsigned long long maxIt) {
        ell = u192_from_dec(ellDec);
        s = u192_from_dec(sDec);
        basis = R::make(R::fromLimbs(px), R::fromLimbs(py));
        target = R::make(R::fromLimbs(qx), R::fromLimbs(qy));
        dpWeight = w;
        maxIters = maxIt;
        spow[0] = u192_from(1);
        for (int i = 1; i < 256; ++i) spow[i] = mod_mul(spow[i - 1], s, ell);
    }

    bool checkSetup(std::string *why) const {
        if (!R::onCurve(basis)) { *why = "P is not on the curve"; return false; }
        if (!R::onCurve(target)) { *why = "Q is not on the curve"; return false; }
        if (!R::scalarMul(basis, ell).inf) { *why = "P does not have order l"; return false; }
        if (!R::scalarMul(target, ell).inf) { *why = "Q does not have order l"; return false; }
        if (!R::eq(R::frob(basis, 1), R::scalarMul(basis, s))) { *why = "sigma(P) != [s]P"; return false; }
        if (!u192_is_zero(mod_add(mod_add(mod_mul(s, s, ell), s, ell), u192_from(2), ell))) {
            *why = "s does not satisfy s^2 + s + 2 = 0"; return false;
        }
        return true;
    }

    struct WalkResult {
        bool ok;
        Point endPoint;
        unsigned long long iters;
        unsigned long long counts[8];
        U192 alpha0;
        u64 seed;
    };

    // Recompute a walk from its seed, counting the Frobenius powers used.
    WalkResult rewalk(u64 seed) const {
        WalkResult out;
        out.ok = false;
        out.seed = seed;
        out.iters = 0;
        for (int i = 0; i < 8; ++i) out.counts[i] = 0;
        Point p = R::startPoint(seed, basis, target, &out.alpha0, ell, spow);
        for (unsigned long long it = 0;; ++it) {
            const int hw = R::weight(p.x);
            if (hw <= dpWeight) {
                out.ok = true;
                out.endPoint = p;
                out.iters = it;
                return out;
            }
            if (it >= maxIters) return out;
            out.counts[R::jOf(hw) - 3]++;
            p = R::step(p, hw);
        }
    }

    // mu = prod_j (1 + s^j)^{n_j}
    U192 multiplier(const unsigned long long *counts) const {
        U192 mu = u192_from(1);
        for (int j = 0; j < 8; ++j) {
            if (!counts[j]) continue;
            const U192 f = mod_add(u192_from(1), spow[j + 3], ell);
            U192 e = u192_zero();
            e.v[0] = counts[j];
            mu = mod_mul(mu, mod_pow(f, e, ell), ell);
        }
        return mu;
    }

    bool solve(const WalkResult &A, const WalkResult &B, U192 *kOut, std::string *why) const {
        if (!A.ok || !B.ok) { *why = "a walk did not reach a distinguished point"; return false; }
        const U192 muA = multiplier(A.counts), muB = multiplier(B.counts);
        const U192 aA = mod_mul(muA, A.alpha0, ell), bA = muA;
        const U192 aB = mod_mul(muB, B.alpha0, ell), bB = muB;
        for (int c = 0; c < M; ++c) {
            const Point rot = R::frob(B.endPoint, c);
            int eps = 0;
            if (R::eq(rot, A.endPoint)) eps = 1;
            else if (R::eq(R::neg(rot), A.endPoint)) eps = -1;
            else continue;
            U192 sc = spow[c % M];
            if (eps < 0) sc = mod_neg(sc, ell);
            const U192 num = mod_sub(aA, mod_mul(sc, aB, ell), ell);
            const U192 den = mod_sub(mod_mul(sc, bB, ell), bA, ell);
            if (u192_is_zero(den)) { *why = "degenerate collision (identical walk)"; return false; }
            const U192 k = mod_mul(num, mod_inv(den, ell), ell);
            if (!R::eq(R::scalarMul(basis, k), target)) { *why = "candidate k failed verification"; return false; }
            *kOut = k;
            return true;
        }
        *why = "endpoints are not in the same orbit";
        return false;
    }

    // ---- store ---------------------------------------------------------
    struct Key {
        u64 v[3];
        bool operator==(const Key &o) const { return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2]; }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            u64 h = k.v[0] * 0x9E3779B97F4A7C15ull;
            h ^= (k.v[1] + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull;
            h ^= (k.v[2] + 0x94D049BB133111EBull) * 0x2545F4914F6CDD1Dull;
            return (size_t)(h ^ (h >> 29));
        }
    };
    struct Entry { u64 seed; unsigned long long iters; };

    std::unordered_map<Key, Entry, KeyHash> store;
    size_t duplicates = 0;
    size_t inserted = 0;

    static Key keyOf(const Elem &canonX) {
        Key k;
        k.v[0] = canonX.v[0];
        k.v[1] = canonX.v[1];
        k.v[2] = canonX.v[2];
        return k;
    }

    // Returns true and fills `other` when this point collides with a stored one.
    bool insert(const DpRecord &rec, Entry *other) {
        const Elem x = R::fromLimbs(rec.x);
        const Key key = keyOf(R::canonical(x));
        auto it = store.find(key);
        if (it != store.end()) {
            if (it->second.seed == rec.seed) { ++duplicates; return false; }
            *other = it->second;
            return true;
        }
        Entry e;
        e.seed = rec.seed;
        e.iters = rec.iters;
        store.emplace(key, e);
        ++inserted;
        return false;
    }
};
