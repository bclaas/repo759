#include <cuda.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <random>
#include <curand_kernel.h>
#include "stencil.cuh"

void stencil(const float* image, const float* output, float* mask, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = (threads_per_block + 2 * R) * sizeof(float);
    stencil_kernel<<<blocks, threads_per_block, shared_mem_size>>>(image, output, mask, n, R);
}

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int shared_index = tid + R;

    if (gid < n) {
        shared[shared_index] = image[gid];
    } else {
        shared[shared_index] = 1.0f;
    }

    if (tid < R) {
        int left_idx = gid - R;
        shared[tid] = (left_idx < 0) ? 1.0f : image[left_idx];
    }

    if (tid >= blockDim.x - R) {
        int right_idx = gid + R;
        shared[shared_index + R] = (right_idx >= n) ? 1.0f : image[right_idx];
    }

    __syncthreads();

    if (gid < n) {
        float sum = 0.0f;
        for (int j = -R; j <= R; j++) {
            sum += shared[shared_index + j] * mask[j + R];
        }
        output[gid] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int n = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);
    int total_size = n * n;
    size_t bytes = total_size * sizeof(float);
    int mask_len = 2 * R + 1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Host allocations
    float* h_image = new float[n];
    float* h_mask = new float[mask_len];
    float* h_output = new float[n];

<<<<<<< HEAD
    for (:_t i = 0; i < size; ++i) {
=======
    for (int i = 0; i < n; ++i) {
>>>>>>> 8096a76 (Euler task2 debug)
        h_image[i] = dist(gen);
        h_mask[i] = dist(gen);
    }    

    // Device allocations
    float *image, *output, *mask;
    cudaMalloc(&image, n * sizeof(float));
    cudaMalloc(&output, n * sizeof(float));
    cudaMalloc(&mask, mask_len * sizeof(float));

    cudaMemcpy(image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, h_mask, mask_len * sizeof(float), cudaMemcpyHostToDevice);

    stencil(image, output, mask, n, R, threads_per_block);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", h_output[n]);
    printf("%f\n", milliseconds);


    // Cleanup
    delete[] h_image;
    delete[] h_output;
    delete[] h_mask;
    cudaFree(image);
    cudaFree(output);
    cudaFree(mask);

    return 0;
}
