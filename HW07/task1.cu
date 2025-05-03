#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>
#include "matmul.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <n> <block_dim>\n", argv[0]);
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);
    size_t size_float = n * n * sizeof(float);
    size_t size_int = n * n * sizeof(int);
    size_t size_double = n * n * sizeof(double);

    int *A = (int*)malloc(size_int);
    int *B = (int*)malloc(size_int);
    int *C = (int*)malloc(size_int);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    matmul_1(A, B, C, n, block_dim);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);

    printf("%d\n", C[0]);
    printf("%d\n", C[n * n - 1]);
    printf("%f\n", milliseconds);

    float *A2 = (float*)malloc(size_float);
    float *B2 = (float*)malloc(size_float);
    float *C2 = (float*)malloc(size_float);

    for (unsigned int i = 0; i < n * n; ++i) {
        A2[i] = (float)A[i];
        B2[i] = (float)B[i];
        C2[i] = (float)C[i];
    }

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    matmul_2(A2, B2, C2, n, block_dim);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start2, stop2);

    printf("%f\n", C2[0]);
    printf("%f\n", C2[n * n - 1]);
    printf("%f\n", milliseconds);

    double *A3 = (double*)malloc(size_double);
    double *B3 = (double*)malloc(size_double);
    double *C3 = (double*)malloc(size_double);

    for (unsigned int i = 0; i < n * n; ++i) {
        A3[i] = (double)A2[i];
        B3[i] = (double)B2[i];
        C3[i] = (double)C2[i];
    }

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    matmul_3(A3, B3, C3, n, block_dim);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds, start2, stop2);

    printf("%f\n", C3[0]);
    printf("%f\n", C3[n * n - 1]);
    printf("%f\n", milliseconds);

    return 0;
}
