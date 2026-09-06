// Device-only translation unit used to measure register pressure, local memory
// traffic and code size with ptxas, without needing a GPU.
#include "../include/curveparams.h"
#include "../include/kernel.h"

template __global__ void eccWalkKernel<CfgF131, unsigned>(WalkParams<unsigned>);
template __global__ void eccInitKernel<CfgF131, unsigned>(WalkParams<unsigned>);
template __global__ void eccReseedKernel<CfgF131, unsigned>(WalkParams<unsigned>);
