// Minimal 192-bit unsigned integers and arithmetic modulo the (130-bit) group
// order.  Host only; performance is irrelevant here.
#pragma once
#include <stdint.h>
#include <string>
#include <stdexcept>

typedef unsigned long long u64;
typedef unsigned int u32;

struct U192 { u64 v[3]; };

inline U192 u192_from(u64 a) { U192 r; r.v[0] = a; r.v[1] = 0; r.v[2] = 0; return r; }
inline U192 u192_zero() { return u192_from(0); }
inline bool u192_is_zero(const U192& a) { return (a.v[0] | a.v[1] | a.v[2]) == 0; }
inline int u192_cmp(const U192& a, const U192& b) {
    for (int i = 2; i >= 0; --i) { if (a.v[i] < b.v[i]) return -1; if (a.v[i] > b.v[i]) return 1; }
    return 0;
}
inline bool u192_eq(const U192& a, const U192& b) { return u192_cmp(a, b) == 0; }
inline U192 u192_add(const U192& a, const U192& b) {
    U192 r; unsigned __int128 c = 0;
    for (int i = 0; i < 3; ++i) { c += (unsigned __int128)a.v[i] + b.v[i]; r.v[i] = (u64)c; c >>= 64; }
    return r;
}
inline U192 u192_sub(const U192& a, const U192& b) {   // requires a >= b
    U192 r; u64 br = 0;
    for (int i = 0; i < 3; ++i) {
        const __int128 t = (__int128)a.v[i] - b.v[i] - br;
        r.v[i] = (u64)t; br = t < 0 ? 1 : 0;
    }
    return r;
}
inline int u192_bit(const U192& a, int i) { return (int)((a.v[i >> 6] >> (i & 63)) & 1); }
inline int u192_bits(const U192& a) {
    for (int i = 191; i >= 0; --i) if (u192_bit(a, i)) return i + 1;
    return 0;
}
inline U192 u192_shl(const U192& a, int k) {
    U192 r = u192_zero();
    for (int i = 0; i < 192; ++i) if (i + k < 192 && u192_bit(a, i)) r.v[(i + k) >> 6] |= 1ull << ((i + k) & 63);
    return r;
}
inline U192 u192_mul_small(const U192& a, u64 k) {
    U192 r; unsigned __int128 c = 0;
    for (int i = 0; i < 3; ++i) { c += (unsigned __int128)a.v[i] * k; r.v[i] = (u64)c; c >>= 64; }
    return r;
}
inline U192 u192_div_small(const U192& a, u64 k, u64* rem) {
    U192 r; unsigned __int128 c = 0;
    for (int i = 2; i >= 0; --i) { c = (c << 64) | a.v[i]; r.v[i] = (u64)(c / k); c %= k; }
    if (rem) *rem = (u64)c;
    return r;
}
inline U192 u192_add_small(const U192& a, u64 k) { return u192_add(a, u192_from(k)); }

inline U192 mod_add(const U192& a, const U192& b, const U192& m) {
    U192 r = u192_add(a, b);
    if (u192_cmp(r, m) >= 0) r = u192_sub(r, m);
    return r;
}
inline U192 mod_sub(const U192& a, const U192& b, const U192& m) {
    if (u192_cmp(a, b) >= 0) return u192_sub(a, b);
    return u192_sub(u192_add(a, m), b);
}
inline U192 mod_neg(const U192& a, const U192& m) { return u192_is_zero(a) ? a : u192_sub(m, a); }
inline U192 mod_mul(const U192& a, const U192& b, const U192& m) {
    U192 r = u192_zero();
    for (int i = u192_bits(b) - 1; i >= 0; --i) {
        r = mod_add(r, r, m);
        if (u192_bit(b, i)) r = mod_add(r, a, m);
    }
    return r;
}
inline U192 mod_pow(const U192& a, const U192& e, const U192& m) {
    U192 r = u192_from(1);
    for (int i = u192_bits(e) - 1; i >= 0; --i) {
        r = mod_mul(r, r, m);
        if (u192_bit(e, i)) r = mod_mul(r, a, m);
    }
    return r;
}
inline U192 mod_inv(const U192& a, const U192& m) { return mod_pow(a, u192_sub(m, u192_from(2)), m); }
inline U192 mod_reduce(const U192& a, const U192& m) {   // a < 2^191 arbitrary
    U192 r = u192_zero();
    for (int i = 191; i >= 0; --i) { r = mod_add(r, r, m); if (u192_bit(a, i)) r = mod_add(r, u192_from(1), m); }
    return r;
}

inline U192 u192_from_dec(const std::string& s) {
    U192 r = u192_zero();
    for (char ch : s) {
        if (ch < '0' || ch > '9') throw std::runtime_error("bad decimal");
        r = u192_add_small(u192_mul_small(r, 10), (u64)(ch - '0'));
    }
    return r;
}
inline std::string u192_to_dec(U192 a) {
    if (u192_is_zero(a)) return "0";
    std::string s;
    while (!u192_is_zero(a)) { u64 rem; a = u192_div_small(a, 10, &rem); s.push_back((char)('0' + rem)); }
    return std::string(s.rbegin(), s.rend());
}
inline U192 u192_from_hex(const std::string& s) {
    U192 r = u192_zero();
    for (char ch : s) {
        int d;
        if (ch >= '0' && ch <= '9') d = ch - '0';
        else if (ch >= 'a' && ch <= 'f') d = ch - 'a' + 10;
        else if (ch >= 'A' && ch <= 'F') d = ch - 'A' + 10;
        else if (ch == ' ' || ch == '_') continue;
        else throw std::runtime_error("bad hex");
        r = u192_add_small(u192_mul_small(r, 16), (u64)d);
    }
    return r;
}
inline std::string u192_to_hex(const U192& a) {
    static const char* hx = "0123456789abcdef";
    std::string s;
    for (int i = 47; i >= 0; --i) s.push_back(hx[(a.v[i >> 4] >> ((i & 15) * 4)) & 15]);
    size_t p = s.find_first_not_of('0');
    return p == std::string::npos ? "0" : s.substr(p);
}

// Tonelli-Shanks square root modulo an odd prime.  Returns false if a is a
// non-residue.
inline bool mod_sqrt(const U192& a0, const U192& p, U192* out) {
    const U192 one = u192_from(1);
    const U192 a = mod_reduce(a0, p);
    if (u192_is_zero(a)) { *out = a; return true; }
    const U192 pm1 = u192_sub(p, one);
    const U192 half = u192_div_small(pm1, 2, nullptr);
    if (!u192_eq(mod_pow(a, half, p), one)) return false;
    U192 q = pm1; int s = 0;
    while (!u192_bit(q, 0)) { q = u192_div_small(q, 2, nullptr); ++s; }
    U192 z = u192_from(2);
    while (u192_eq(mod_pow(z, half, p), one)) z = u192_add_small(z, 1);
    int m = s;
    U192 c = mod_pow(z, q, p);
    U192 t = mod_pow(a, q, p);
    U192 r = mod_pow(a, u192_div_small(u192_add_small(q, 1), 2, nullptr), p);
    while (!u192_eq(t, one)) {
        int i = 0; U192 tt = t;
        while (!u192_eq(tt, one)) { tt = mod_mul(tt, tt, p); ++i; if (i >= m) return false; }
        U192 b = c;
        for (int j = 0; j < m - i - 1; ++j) b = mod_mul(b, b, p);
        m = i; c = mod_mul(b, b, p); t = mod_mul(t, c, p); r = mod_mul(r, b, p);
    }
    *out = r;
    return true;
}
