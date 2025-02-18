#include "matmul.h"
#include <vector>
#include <iostream>

void mmul1(const double* A, const double* B, double* C, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            *(C + (i * n + j)) = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                *(C + (i * n + j)) += *(A + (i * n + k)) * *(B + (k * n + j));
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, std::size_t n) {
    std::size_t i, j, k;
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; j++) {
                C[i * n + j] += *(A + (i * n + k)) * *(B + (k * n + j));
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, std::size_t n) {
    std::size_t j = 0, k = 0, i = 0;
    while (j < n) {
        k = 0;
        while (k < n) {
            i = 0;
            while (i < n) {
                C[i * n + j] += *(A + (i * n + k)) * *(B + (k * n + j));
                i++;
            }
            k++;
        }
        j++;
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            C.at(i * n + j) = 0.0;
            for (std::size_t k = 0; k < n; k++) {
                C.at(i * n + j) += A.at(i * n + k) * B.at(k * n + j);
            }
        }
    }
}
