#include <cuda.h>
#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) { b[idx] *= a[idx]; }
}

