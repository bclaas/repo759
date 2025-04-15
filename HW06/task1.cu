#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <curand_kernel.h>
#include "matmul.cuh"
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    size_t total_threads = n * n;
    size_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__global__ void fill_random(float* A, float* B, int total_size, unsigned int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total_size) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    float r1 = curand_uniform(&state); // [0, 1)
    float r2 = curand_uniform(&state); // [0, 1)

    A[i] = r1 * 2.0f - 1.0f;  // [-1, 1)
    B[i] = r2 * 2.0f - 1.0f;  // [-1, 1)
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n * n) return;

    int row = idx / n;
    int col = idx % n;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    int n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);
    int total_size = n * n;
    size_t bytes = total_size * sizeof(float);

    float *d_A, *d_B, *d_C;
    float *h_A = new float[total_size];
    float *h_B = new float[total_size];
    float *h_C = new float[total_size];

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill matrices A and B with random values between -1 and 1
    int fill_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    fill_random<<<fill_blocks, threads_per_block>>>(d_A, d_B, total_size, time(NULL));
    cudaDeviceSynchronize();

    // Multiply A and B -> store result in C
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Copy result back to host
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << h_C[n * n - 1] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    std::cout << duration_sec.count() << std::endl;

    return 0;
}