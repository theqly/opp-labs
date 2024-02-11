#include "stdio.h"
#include "stdlib.h"
#include "time.h"
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

    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,  &size);
    MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

    const int size_of_block = N / size;

    int* a;
    int* b = malloc(sizeof(int) * N);

    long long int s, tmp;

    double start, end;

    if(rank == 0) {
        a = malloc(sizeof(int) * N);

        start = MPI_Wtime();

        fill_vectors(a, b, N);
        s = 0;
    }

    int* local = malloc(sizeof(int) * size_of_block);

    MPI_Scatter(a, size_of_block, MPI_INT, local, size_of_block, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_INT, 0, MPI_COMM_WORLD);

    tmp = find_s(local, b, size_of_block);

    MPI_Reduce(&tmp, &s, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local);

    if(rank == 0) {
        end = MPI_Wtime();
        printf("Time taken: %lf sec.\n", end - start);
        printf("s is: %lld\n", s);
        free(a);
    }

    free(b);

    MPI_Finalize();

    return 0;
}