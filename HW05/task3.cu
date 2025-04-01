#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "vscale.cuh"

void randomizeArray(float *arr, int n, float min, float max) {
    for (int i = 0; i < n; i++) {
        arr[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    srand(time(NULL));
    
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    randomizeArray(h_a, n, -10.0f, 10.0f);
    randomizeArray(h_b, n, 0.0f, 1.0f);
    
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    vscale<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%f\n", milliseconds);
    printf("%f\n", h_b[0]);
    printf("%f\n", h_b[n - 1]);
    
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    
    return 0;
}
