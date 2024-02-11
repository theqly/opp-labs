#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define N 99984

void fill_vectors(int* a, int* b, const int size) {
    if(a != NULL && b != NULL) {
        for(size_t i = 0; i < size; ++i) {
            a[i] = i;
			b[i] = i;
        }
    }
}

long long int find_s(const int* a, const int* b, const int size) {
    long long int s = 0;
    if(a != NULL && b != NULL) {
        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < N; ++j) {
                s += a[i]*b[j];
            }
        }
    }
    return s;
}

int main() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int* a = malloc(sizeof(int) * N);
    int* b = malloc(sizeof(int) * N);
    fill_vectors(a, b, N);
    const long long int s = find_s(a, b, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("s is: %lld\n", s);
    free(a);
    free(b);
    return 0;
}