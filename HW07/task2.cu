#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "reduce.cuh"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <threads_per_block>\n";
        return 1;
    }

    size_t N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    // Allocate and initialize host array with random values in range [-1, 1]
    std::vector<float> h_data(N);
    srand(time(NULL));
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
    }

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    size_t blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_out, blocks * sizeof(float));

    cudaMemcpy(d_in, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&d_in, &d_out, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the result back to host
    float h_result;
    cudaMemcpy(&h_result, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Sum: " << h_result << "\n";
    std::cout << "Time: " << ms << " ms\n";

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
