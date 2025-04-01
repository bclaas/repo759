#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#define N 16

__global__ void computeKernel(int *dA, int a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        dA[idx] = a * threadIdx.x + blockIdx.x;
    }
}

int main(){
    srand(time(NULL));
    int a = rand() % 10 + 1;
    int *dA, hA[N];
    cudaMalloc((void**)&dA, N * sizeof(int));
    computeKernel<<<2, 8>>>(dA, a);
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) { printf("%d ", hA[i]); }
    printf("\n");
    cudaFree(dA);
    return 0;
}

