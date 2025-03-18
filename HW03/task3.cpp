#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "msort.h"

int main(int argc, char* argv[]) {
    std::size_t n = std::stoi(argv[1]);
    int t = std::stoi(argv[2]);
    int ts = std::stoi(argv[3]);

    std::vector<int> arr(n);

    srand(time(0));
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = rand() % 2001 - 1000;
    }

    omp_set_num_threads(t);

    clock_t start = clock();
    msort(arr.data(), n, ts);
    clock_t end = clock();
    float duration = static_cast<float>(end - start) / CLOCKS_PER_SEC * 1000;

    std::cout << arr[0] << std::endl;
    std::cout << arr[n * n - 1] << std::endl;
    std::cout << duration << std::endl;

    return 0;
}