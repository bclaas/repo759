#include <cuda_runtime.h>
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

__global__ void matmul2_kernel(const int *A, const int *B, int *C, unsigned int n) {
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

void matmul_2(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
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
    matmul3_kernel<<<blocks, threads, shared>>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <n> <block_dim>\n", argv[0]);
        return 1;
    }
    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);
    size_t size_float = n * n * sizeof(float);
    size_t size_int = n * n * sizeof(int);

    // MATMUL 1    
    int *A = (int*)malloc(size_int);
    int *B = (int*)malloc(size_int);
    int *C = (int*)malloc(size_int);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    matmul_1(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);
    printf("%f\n", milliseconds);

    // MATMUL 2
    int *B_old = B;
    int *A_old = A;
    int *C_old = C;

    B = (float*)malloc(n * n * sizeof(float));
    A = (float*)malloc(n * n * sizeof(float));
    C = (float*)malloc(n * n * sizeof(float));

    for (unsigned int i = 0; i < n * n; ++i) {
        B[i] = (float)B_old[i];
        A[i] = (float)A_old[i];
        C[i] = (float)C_old[i];
    }
    
    //cudaEvent_t start, stop;
    cudaEventCreate(&start);
    matmul_2(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);
    printf("%f\n", milliseconds);

    free(A_old);
    free(B_old);
    free(C_old);

    // MATMUL 3
    float *B_old = B;
    float *A_old = A;
    float *C_old = C;

    B = (double*)malloc(n * n * sizeof(double));
    A = (double*)malloc(n * n * sizeof(double));
    C = (double*)malloc(n * n * sizeof(double));

    for (unsigned int i = 0; i < n * n; ++i) {
        B[i] = (double)B_old[i];
        A[i] = (double)A_old[i];
        C[i] = (double)C_old[i];
    }

    //cudaEvent_t start, stop;
    cudaEventCreate(&start);
    matmul_3(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);
    printf("%f\n", milliseconds);

    free(A_old);
    free(B_old);
    free(C_old);

    return 0;
}