#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "matmul.h"

void mmul(const float* A, const float* B, float* C, std::size_t n) {
    std::size_t i, j, k;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; j++) {
                C[i * n + j] += *(A + (i * n + k)) * *(B + (k * n + j));
            }
        }
    }
}
