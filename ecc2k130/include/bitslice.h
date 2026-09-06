// Word types and three-input logic for bitsliced GF(2^m) arithmetic.
//
// Every value is one machine word holding W independent bit lanes, so one word
// operation advances W walks at once.  Three word types are supported and the
// arithmetic source is identical for all of them:
//
//   unsigned      32 lanes, CUDA devices
//   uint64_t      64 lanes, portable host fallback
//   Bits256/512  256/512 lanes, AVX2 / AVX-512 hosts
//
// ECC_XOR3 / ECC_XORAND / ECC_MAJ / ECC_SEL are single instructions on both of
// the interesting targets: LOP3.LUT on sm_50 and later, and vpternlogd on
// AVX-512.  They use the same truth-table constants, computed for
// a = 0xF0, b = 0xCC, c = 0xAA.
#pragma once

#include <stdint.h>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define ECC_HD __host__ __device__ __forceinline__
#define ECC_DEV __device__ __forceinline__
// Routines big enough that one shared copy beats inlining: keeping these out of
// line is what holds the kernel to a few thousand instructions instead of the
// ~285000 a fully inlined batch expands to, which no instruction cache can hold
// and which ptxas takes many minutes to register-allocate.
#define ECC_BIG __host__ __device__ __noinline__
#else
#define ECC_HD inline
#define ECC_DEV inline
#define ECC_BIG inline
#endif

#define ECC_ZERO ((W)0)

#define ECC_LUT_XOR3 0x96
#define ECC_LUT_XORAND 0x78
#define ECC_LUT_MAJ 0xe8
#define ECC_LUT_SEL 0xca

// ---- generic (scalar) implementations -------------------------------------
template <class W>
ECC_HD W eccXor3(const W &a, const W &b, const W &c) {
    return a ^ b ^ c;
}
template <class W>
ECC_HD W eccXorAnd(const W &a, const W &b, const W &c) {
    return a ^ (b & c);
}
template <class W>
ECC_HD W eccMaj(const W &a, const W &b, const W &c) {
    return (a & b) | (c & (a ^ b));
}
template <class W>
ECC_HD W eccSel(const W &a, const W &b, const W &c) {
    return (a & b) | (~a & c);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
#define ECC_LOP3_FN(NAME, LUT)                                                       \
    __device__ __forceinline__ unsigned NAME(const unsigned &a, const unsigned &b,    \
                                             const unsigned &c) {                     \
        unsigned r;                                                                    \
        asm("lop3.b32 %0, %1, %2, %3, " #LUT ";" : "=r"(r) : "r"(a), "r"(b), "r"(c));   \
        return r;                                                                      \
    }
ECC_LOP3_FN(eccXor3, 0x96)
ECC_LOP3_FN(eccXorAnd, 0x78)
ECC_LOP3_FN(eccMaj, 0xe8)
ECC_LOP3_FN(eccSel, 0xca)
#endif

#define ECC_XOR3(a, b, c) eccXor3((a), (b), (c))
#define ECC_XORAND(a, b, c) eccXorAnd((a), (b), (c))
#define ECC_MAJ(a, b, c) eccMaj((a), (b), (c))
// select: bits of a choose b, else c
#define ECC_SEL(a, b, c) eccSel((a), (b), (c))

template <class W>
struct WordTraits {
    static const int LANES = (int)(sizeof(W) * 8);
};

template <class W>
ECC_HD int laneBit(const W &w, int lane) {
    return (int)((w >> lane) & (W)1);
}

template <class W>
ECC_HD W laneMask(int lane) {
    return ((W)1) << lane;
}

// build a word from its 64-bit limbs (lane group g holds lanes 64g..64g+63)
template <class W>
ECC_HD W eccWordFromLimbs(const unsigned long long *p) {
    return (W)p[0];
}

// ---------------------------------------------------------------------------
// wide host words
// ---------------------------------------------------------------------------
#if !defined(__CUDACC__) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>

#if defined(__AVX512F__)
struct alignas(64) Bits512 {
    __m512i v;
    Bits512() {}
    Bits512(int x) { v = x ? _mm512_set1_epi32(-1) : _mm512_setzero_si512(); }
    explicit Bits512(__m512i w) : v(w) {}
    Bits512 operator^(const Bits512 &o) const { return Bits512(_mm512_xor_si512(v, o.v)); }
    Bits512 operator&(const Bits512 &o) const { return Bits512(_mm512_and_si512(v, o.v)); }
    Bits512 operator|(const Bits512 &o) const { return Bits512(_mm512_or_si512(v, o.v)); }
    Bits512 operator~() const { return Bits512(_mm512_ternarylogic_epi32(v, v, v, 0x0f)); }
    Bits512 &operator^=(const Bits512 &o) { v = _mm512_xor_si512(v, o.v); return *this; }
    Bits512 &operator&=(const Bits512 &o) { v = _mm512_and_si512(v, o.v); return *this; }
    Bits512 &operator|=(const Bits512 &o) { v = _mm512_or_si512(v, o.v); return *this; }
    bool operator==(const Bits512 &o) const {
        return _mm512_cmpneq_epi64_mask(v, o.v) == 0;
    }
    bool operator!=(const Bits512 &o) const { return !(*this == o); }
    unsigned long long limb(int i) const {
        unsigned long long tmp[8];
        _mm512_storeu_si512((void *)tmp, v);
        return tmp[i];
    }
    static Bits512 fromLimbs(const unsigned long long *p) { return Bits512(_mm512_loadu_si512((const void *)p)); }
};
#define ECC_LOG3_512(NAME, LUT)                                                         \
    inline Bits512 NAME(const Bits512 &a, const Bits512 &b, const Bits512 &c) {          \
        return Bits512(_mm512_ternarylogic_epi32(a.v, b.v, c.v, LUT));                   \
    }
ECC_LOG3_512(eccXor3, ECC_LUT_XOR3)
ECC_LOG3_512(eccXorAnd, ECC_LUT_XORAND)
ECC_LOG3_512(eccMaj, ECC_LUT_MAJ)
ECC_LOG3_512(eccSel, ECC_LUT_SEL)
template <>
struct WordTraits<Bits512> {
    static const int LANES = 512;
};
template <>
inline int laneBit<Bits512>(const Bits512 &w, int lane) {
    return (int)((w.limb(lane >> 6) >> (lane & 63)) & 1ull);
}
template <>
inline Bits512 laneMask<Bits512>(int lane) {
    unsigned long long tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    tmp[lane >> 6] = 1ull << (lane & 63);
    return Bits512::fromLimbs(tmp);
}
template <>
inline Bits512 eccWordFromLimbs<Bits512>(const unsigned long long *p) {
    return Bits512::fromLimbs(p);
}
#endif  // __AVX512F__

struct alignas(32) Bits256 {
    __m256i v;
    Bits256() {}
    Bits256(int x) { v = x ? _mm256_set1_epi32(-1) : _mm256_setzero_si256(); }
    explicit Bits256(__m256i w) : v(w) {}
    Bits256 operator^(const Bits256 &o) const { return Bits256(_mm256_xor_si256(v, o.v)); }
    Bits256 operator&(const Bits256 &o) const { return Bits256(_mm256_and_si256(v, o.v)); }
    Bits256 operator|(const Bits256 &o) const { return Bits256(_mm256_or_si256(v, o.v)); }
    Bits256 operator~() const { return Bits256(_mm256_xor_si256(v, _mm256_set1_epi32(-1))); }
    Bits256 &operator^=(const Bits256 &o) { v = _mm256_xor_si256(v, o.v); return *this; }
    Bits256 &operator&=(const Bits256 &o) { v = _mm256_and_si256(v, o.v); return *this; }
    Bits256 &operator|=(const Bits256 &o) { v = _mm256_or_si256(v, o.v); return *this; }
    bool operator==(const Bits256 &o) const {
        return _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, o.v)) == -1;
    }
    bool operator!=(const Bits256 &o) const { return !(*this == o); }
    unsigned long long limb(int i) const {
        unsigned long long tmp[4];
        _mm256_storeu_si256((__m256i *)tmp, v);
        return tmp[i];
    }
    static Bits256 fromLimbs(const unsigned long long *p) { return Bits256(_mm256_loadu_si256((const __m256i *)p)); }
};
#if defined(__AVX512VL__)
#define ECC_LOG3_256(NAME, LUT)                                                         \
    inline Bits256 NAME(const Bits256 &a, const Bits256 &b, const Bits256 &c) {          \
        return Bits256(_mm256_ternarylogic_epi32(a.v, b.v, c.v, LUT));                   \
    }
ECC_LOG3_256(eccXor3, ECC_LUT_XOR3)
ECC_LOG3_256(eccXorAnd, ECC_LUT_XORAND)
ECC_LOG3_256(eccMaj, ECC_LUT_MAJ)
ECC_LOG3_256(eccSel, ECC_LUT_SEL)
#endif
template <>
struct WordTraits<Bits256> {
    static const int LANES = 256;
};
template <>
inline int laneBit<Bits256>(const Bits256 &w, int lane) {
    return (int)((w.limb(lane >> 6) >> (lane & 63)) & 1ull);
}
template <>
inline Bits256 laneMask<Bits256>(int lane) {
    unsigned long long tmp[4] = {0, 0, 0, 0};
    tmp[lane >> 6] = 1ull << (lane & 63);
    return Bits256::fromLimbs(tmp);
}
template <>
inline Bits256 eccWordFromLimbs<Bits256>(const unsigned long long *p) {
    return Bits256::fromLimbs(p);
}
#endif  // AVX

// Host word selection: widest available unless overridden with ECC_HOST_WORD.
#if !defined(ECC_HOST_WORD)
#if defined(__AVX512F__) && !defined(__CUDACC__)
#define ECC_HOST_WORD Bits512
#elif defined(__AVX2__) && !defined(__CUDACC__)
#define ECC_HOST_WORD Bits256
#else
#define ECC_HOST_WORD unsigned long long
#endif
#endif
