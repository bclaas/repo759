#include "reduce.cuh"
#include <cuda.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0;
    if (idx < n) sum += g_idata[idx];
    if (idx + blockDim.x < n) sum += g_idata[idx + blockDim.x];
    sdata[tid] = sum;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    size_t smem_size = threads_per_block * sizeof(float);

    float *in = *input;
    float *out = *output;

    while (blocks > 1) {
        reduce_kernel<<<blocks, threads_per_block, smem_size>>>(in, out, N);
        cudaDeviceSynchronize();

        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

        float *tmp = in;
        in = out;
        out = tmp;
    }
    cudaMemcpy(*input, out, sizeof(float), cudaMemcpyDeviceToDevice);
}
