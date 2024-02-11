#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

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

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int* a = malloc(sizeof(int) * N);
    int* b = malloc(sizeof(int) * N);
    double start = MPI_Wtime();
    fill_vectors(a, b, N);
    const long long int s = find_s(a, b, N);
    double end = MPI_Wtime();
    printf("Time taken: %lf sec.\n", end - start);
    printf("s is: %lld\n", s);
    free(a);
    free(b);

    MPI_Finalize();

    return 0;
}