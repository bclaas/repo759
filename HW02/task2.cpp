#include "convolution.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <image_size> <mask_size>" << endl;
        return 1;
    }

    size_t n = atoi(argv[1]);
    size_t m = atoi(argv[2]);

    if (n == 0 || m == 0) {
        cerr << "Image and mask sizes must be greater than zero." << endl;
        return 1;
    }

    float* image = new float[n * n];
    float* mask = new float[m * m];
    float* output = new float[n * n];

    srand(static_cast<unsigned>(time(nullptr)));
    for (size_t i = 0; i < n * n; i++) {
        image[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
    }
    for (size_t i = 0; i < m * m; i++) {
        mask[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    clock_t start = clock();
    convolve(image, output, n, mask, m);
    clock_t end = clock();

    cout << "Time taken: " << (1000.0 * (end - start) / CLOCKS_PER_SEC) << " ms" << endl;
    cout << "First element: " << output[0] << endl;
    cout << "Last element: " << output[n * n - 1] << endl;

    delete[] image;
    delete[] mask;
    delete[] output;
    
    return 0;
}
