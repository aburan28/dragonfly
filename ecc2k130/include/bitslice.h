// Word type and three-input logic for bitsliced GF(2^m) arithmetic.
//
// Every value is one machine word holding W independent bit lanes, so one word
// operation advances W walks at once.  On the device W = 32 (uint32_t), on the
// host W = 64 (uint64_t); the arithmetic source is identical.
//
// ECC_XOR3 / ECC_XORAND / ECC_MAJ / ECC_SEL each map to a single LOP3.LUT
// instruction on sm_50 and later.  That is where the 1.5-1.8x instruction
// reduction over a two-input rendering of the straight-line code comes from.
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
// a = 0xF0, b = 0xCC, c = 0xAA
__device__ __forceinline__ unsigned lop3_(unsigned a, unsigned b, unsigned c, unsigned lut) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(0), "n"(0));
    return r;
}
#define ECC_LOP3_DECL(NAME, LUT)                                                       \
    __device__ __forceinline__ unsigned NAME(unsigned a, unsigned b, unsigned c) {     \
        unsigned r;                                                                    \
        asm("lop3.b32 %0, %1, %2, %3, " #LUT ";" : "=r"(r) : "r"(a), "r"(b), "r"(c));   \
        return r;                                                                      \
    }
ECC_LOP3_DECL(eccXor3_, 0x96)
ECC_LOP3_DECL(eccXorAnd_, 0x78)
ECC_LOP3_DECL(eccMaj_, 0xe8)
ECC_LOP3_DECL(eccSel_, 0xca)
#define ECC_XOR3(a, b, c) eccXor3_((a), (b), (c))
#define ECC_XORAND(a, b, c) eccXorAnd_((a), (b), (c))
#define ECC_MAJ(a, b, c) eccMaj_((a), (b), (c))
// select: bits of a choose b, else c
#define ECC_SEL(a, b, c) eccSel_((a), (b), (c))
#else
#define ECC_XOR3(a, b, c) ((a) ^ (b) ^ (c))
#define ECC_XORAND(a, b, c) ((a) ^ ((b) & (c)))
#define ECC_MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) ^ (b))))
#define ECC_SEL(a, b, c) (((a) & (b)) | (~(a) & (c)))
#endif

template <class W>
struct WordTraits {
    static const int LANES = (int)(sizeof(W) * 8);
};

// lane extraction / insertion used only on the rare distinguished-point path
template <class W>
ECC_HD int laneBit(W w, int lane) {
    return (int)((w >> lane) & (W)1);
}

template <class W>
ECC_HD W laneMask(int lane) {
    return ((W)1) << lane;
}
