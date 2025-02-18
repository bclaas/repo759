#include <iostream>
#include <cstdlib>
#include <ctime>
#include "scan.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return 1;
    }

    std::size_t n = std::atoi(argv[1]);
    if (n == 0) {
        std::cerr << "Array size must be greater than zero." << std::endl;
        return 1;
    }

    float* arr = new float[n];
    float* output = new float[n];

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    clock_t start = clock();
    scan(arr, output, n);
    clock_t end = clock();

    std::cout << "Time taken: " << (1000.0 * (end - start) / CLOCKS_PER_SEC) << " ms" << std::endl;
    std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Last element: " << output[n - 1] << std::endl;

    delete[] arr;
    delete[] output;
    return 0;
}
