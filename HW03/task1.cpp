#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "matmul.h"

void printResults(const std::string& functionName, const float* A, const float* B, float* C, std::size_t n, void(*mmul)(const float*, const float*, float*, std::size_t)) {
    clock_t start = clock();
    
    mmul(A, B, C, n);
    
    clock_t end = clock();
    float duration = static_cast<float>(end - start) / CLOCKS_PER_SEC * 1000;
    
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << duration << std::endl;
    
}

int main(int argc, char* argv[]) {
    // Check if user provided an argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    std::size_t n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: matrix size must be a positive integer.\n";
        return 1;
    }

    int t = std::atoi(argv[2]);
    omp_set_num_threads(t);

    std::vector<float> A(n * n), B(n * n), C(n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A[i * n + j] = rand() % 1000;
            B[i * n + j] = rand() % 1000;
        }
    }

    std::cout << "Matrix dimension: " << n << "x" << n << std::endl;

    printResults("mmul", A.data(), B.data(), C.data(), n, mmul);

    return 0;
}