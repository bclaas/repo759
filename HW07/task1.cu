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

    free(A);
    free(B);
    free(C);
    float *A;
    float *B;
    float *C;

    A = (float*)malloc(size_float);
    B = (float*)malloc(size_float);
    C = (float*)malloc(size_float);

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
    *B_old = B;
    *A_old = A;
    *C_old = C;

    free(A);
    free(B);
    free(C);
    double *A;
    double *B;
    double *C;

    A = (double*)malloc(size_double);
    B = (double*)malloc(size_double);
    C = (double*)malloc(size_double);

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