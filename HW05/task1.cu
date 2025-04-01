#include <cstdio>

__global__ void factorialKernel() {
    int idx = threadIdx.x + 1;
    if (idx > 8) return;
    
    int fact = 1;
    for (int i = 1; i <= idx; i++) {
        fact *= i;
    }
    
    printf("%d!=%d\n", idx, fact);
}

int main() {
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
