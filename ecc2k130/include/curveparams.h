// Adapters turning each generated namespace into a Cfg type for the templates.
#pragma once

#include "bitslice.h"
#include "fieldbs.h"
#include "../generated/eccF131.h"
#include "../generated/eccF83.h"
#include "../generated/eccF41.h"
#include "../generated/eccF23.h"

#define ECC_MAKE_CFG(NAME, NS)                                                        \
    struct NAME {                                                                     \
        static const int M = NS::M;                                                   \
        static const int NRING = NS::NRING;                                           \
        static const int HWBITS = NS::HWBITS;                                         \
        static const int LEAF = NS::LEAF;                                             \
        static const int PRODLEN = NS::PRODLEN;                                       \
        static const int DP_WEIGHT = NS::DP_WEIGHT;                                   \
        template <class W> static ECC_BIG void multPrep(const W *a, W *o) { NS::multPrep<W>(a, o); }   \
        template <class W> static ECC_BIG void toOnb(const W *h, W *o) { NS::toOnb<W>(h, o); }         \
        template <class W> static ECC_BIG void mulLeaf(const W *a, const W *b, W *o) { NS::mulLeaf<W>(a, b, o); } \
        template <class W> static ECC_BIG void hamming(const W *x, W *o) { NS::hamming<W>(x, o); }     \
    };

ECC_MAKE_CFG(CfgF131, eccF131)
ECC_MAKE_CFG(CfgF83, eccF83)
ECC_MAKE_CFG(CfgF41, eccF41)
ECC_MAKE_CFG(CfgF23, eccF23)

ECC_DEFINE_LEAF(CfgF131, eccF131::LEAF)
ECC_DEFINE_LEAF(CfgF83, eccF83::LEAF)
ECC_DEFINE_LEAF(CfgF41, eccF41::LEAF)
ECC_DEFINE_LEAF(CfgF23, eccF23::LEAF)
