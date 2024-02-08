#include "stdio.h"
#include "stdlib.h"
#include "time.h"

int* get_random_vector(const int N) {
    int* vector = malloc(sizeof(int) * N);
    for(size_t i = 0; i < N; ++i) {
        vector[i] = rand() % N;
    }
    return vector;
}

long long int find_s(const int* a, const int* b, const int N) {
    long long int s = 0;
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j) {
            s += a[i]*b[i];
        }
    }
    return s;
}

int main() {
    srand(time(NULL));
    int N;
    scanf("%i", &N);
    int* a = get_random_vector(N);
    int* b = get_random_vector(N);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    const long long int s = find_s(a, b, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("s is: %lld", s);
    free(a);
    free(b);
    return 0;
}