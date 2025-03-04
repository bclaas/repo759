#include "msort.h"
#include <vector>
#include <algorithm>
#include <omp.h>

void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right) {
    std::size_t n1 = mid - left + 1;
    std::size_t n2 = right - mid;

    std::vector<int> leftArr(n1), rightArr(n2);

    for (std::size_t i = 0; i < n1; i++) {
        leftArr[i] = arr[left + i];
    }

    for (std::size_t j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
    }

    std::size_t i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
    }

    while (i < n1) {
        arr[k++] = leftArr[i++];
    }

    while (j < n2) {
        arr[k++] = rightArr[j++];
    }
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n <= threshold) {
        std::sort(arr, arr + n); // Serial sort for small arrays
        return;
    }

    std::size_t mid = n / 2;

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task shared(arr)
            msort(arr, mid, threshold); // Sort left part

            #pragma omp task shared(arr)
            msort(arr + mid, n - mid, threshold); // Sort right part

            #pragma omp taskwait // Wait for both recursive calls to complete
            merge(arr, 0, mid - 1, n - 1); // Merge the two sorted parts
        }
    }
}