#include <iostream>
#include <vector>
#include <ctime>

void printResults(const std::string& functionName, const double* A, const double* B, double* C, std::size_t n, void(*mmul)(const double*, const double*, double*, std::size_t)) {
    clock_t start = clock();
    
    mmul(A, B, C, n);
    
    clock_t end = clock();
    double duration = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
    
    std::cout << functionName << " time (ms): " << duration << std::endl;
    std::cout << "Last element of C: " << C[n * n - 1] << std::endl;
}

int main() {
    const std::size_t n = 1000;
    
    std::vector<double> A(n * n), B(n * n), C(n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A[i * n + j] = rand() % 1000;
            B[i * n + j] = rand() % 1000;
        }
    }

    std::cout << "Matrix dimension: " << n << "x" << n << std::endl;

    printResults("mmul1", A.data(), B.data(), C.data(), n, mmul1);
    std::fill(C.begin(), C.end(), 0.0);
    printResults("mmul2", A.data(), B.data(), C.data(), n, mmul2);
    std::fill(C.begin(), C.end(), 0.0);
    printResults("mmul3", A.data(), B.data(), C.data(), n, mmul3);
    std::fill(C.begin(), C.end(), 0.0);
    printResults("mmul4", A, B, C, n, mmul4);
    
    return 0;
}