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

    int *A_old = A;
    int *B_old = B;
    int *C_old = C;

    free(A);
    free(B);
    free(C);

    A = (float*)malloc(size_float);
    B = (float*)malloc(size_float);
    C = (float*)malloc(size_float);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = (float)A_old[i];
        B[i] = (float)B_old[i];
        C[i] = (float)C_old[i];
    }

    free(A_old);
    free(B_old);
    free(C_old);

    matmul_2(A, B, C, n, block_dim);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);

    float *A_old2 = A;
    float *B_old2 = B;
    float *C_old2 = C;

    free(A);
    free(B);
    free(C);

    A = (double*)malloc(size_double);
    B = (double*)malloc(size_double);
    C = (double*)malloc(size_double);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = (double)A_old2[i];
        B[i] = (double)B_old2[i];
        C[i] = (double)C_old2[i];
    }

    free(A_old2);
    free(B_old2);
    free(C_old2);

    cudaEventCreate(&start);
    matmul_3(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", C[0]);
    printf("%f\n", C[n * n - 1]);
    printf("%f\n", milliseconds);

    free(A);
    free(B);
    free(C);

    return 0;
}
