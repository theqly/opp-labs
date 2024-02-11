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

    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,  &size);
    MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

    const int size_of_block = N / size;

    int* a = malloc(sizeof(int) * N);
    int* b = malloc(sizeof(int) * N);

    long long int s, tmp;

    double start, end;

    if(rank == 0) {

        start = MPI_Wtime();

        fill_vectors(a, b, N);

        for(size_t i = 1; i < size; ++i) {
            MPI_Send(a + size_of_block*i, size_of_block, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        s = find_s(a, b, size_of_block);

        for(size_t i = 1; i < size; ++i) {
            MPI_Recv(&tmp, 1, MPI_LONG_LONG_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            s += tmp;
        }
    } else {
        MPI_Recv(a + size_of_block*rank, size_of_block, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        tmp = find_s(a + size_of_block*rank, b, size_of_block);
        MPI_Send(&tmp, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD);
    }

    if(rank == 0) {
        end = MPI_Wtime();
        printf("Time taken: %lf sec.\n", end - start);
        printf("s is: %lld\n", s);
    }

    free(a);
    free(b);

    MPI_Finalize();

    return 0;
}