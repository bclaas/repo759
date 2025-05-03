#include <cuda.h>
#include <stdio.h>
#include <ctime>
#include "matmul.cuh"

__global__ void matmul1_kernel(const int *A, const int *B, int *C, unsigned int n) {
    extern __shared__ int s[];
    int *As = s;
    int *Bs = s + blockDim.x * blockDim.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int t = 0; t < n / blockDim.x; ++t) {
        As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + t * blockDim.x + threadIdx.x];
        Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    int *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared = 2 * block_dim * block_dim * sizeof(int);
    matmul1_kernel<<<blocks, threads, shared>>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmul2_kernel(const float *A, const float *B, float *C, unsigned int n) {
    extern __shared__ int s[];
    int *As = s;
    int *Bs = s + blockDim.x * blockDim.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int t = 0; t < n / blockDim.x; ++t) {
        As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + t * blockDim.x + threadIdx.x];
        Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared = 2 * block_dim * block_dim * sizeof(int);
    matmul2_kernel<<<blocks, threads, shared>>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmul3_kernel(const double *A, const double *B, double *C, unsigned int n) {
    extern __shared__ int s[];
    int *As = s;
    int *Bs = s + blockDim.x * blockDim.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int t = 0; t < n / blockDim.x; ++t) {
        As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + t * blockDim.x + threadIdx.x];
        Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared = 2 * block_dim * block_dim * sizeof(double);
    matmul3_kernel<<<blocks, threads, shared>>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
