#include <cstddef>
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

void convolve(const float* image, float* output, std::size_t n, const float* mask, std::size_t m) {
    #pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float gxy = 0.0f;

            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    int ii = x + i - (m - 1) / 2;
                    int jj = y + j - (m - 1) / 2;

                    float fij;
                    if (!(0 <= jj && jj < static_cast<int>(n)) && !(0 <= ii && ii < static_cast<int>(n))) {
                        fij = 0.0f;
                    } else if (!(0 <= jj && jj < static_cast<int>(n)) || !(0 <= ii && ii < static_cast<int>(n))) {
                        fij = 1.0f;
                    } else {
                        fij = image[ii * n + jj];
                    }

                    gxy += mask[i * m + j] * fij;
                }
            }

            output[x * n + y] = gxy;
        }
    }
}