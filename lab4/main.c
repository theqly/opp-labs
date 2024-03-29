#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if(argc < 3) {
        fprintf(stderr, "Usage: ./lab3 <p1> <p2>\n");
        return 1;
    }
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm grid_comm;

    const int ndims = 2;

    const int p1 = atoi(argv[1]);
    const int p2 = atoi(argv[2]);

    const int dims[] = {p1, p2};
    const int periods[] = {0, 0};
    const int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    const int row_remain_dims[] = {1, 0};
    const int col_remain_dims[] = {0, 1};

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
    MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    int coords[ndims];
    MPI_Cart_coords(grid_comm, rank, ndims, coords);

    const int n1 = 8;
    const int n2 = 8;
    const int n3 = 8;

    double *A, *B, *C;
    const int number_of_rows = n1 / p1;
    const int number_of_cols = n3 / p2;

    double* local_A = (double*) malloc(number_of_rows * n2 * sizeof(double));
    double* local_B = (double*) malloc(n2 * number_of_cols * sizeof(double));
    double* local_C = (double*) malloc(number_of_rows * number_of_cols * sizeof(double));

    double start;
    if(rank == 0) {
        A = (double*) malloc(n1 * n2 * sizeof(double));
        B = (double*) malloc(n2 * n3 * sizeof(double));
        C = (double*) malloc(n1 * n3 * sizeof(double));
        //filling();
        start = MPI_Wtime();
    }

    if(coords[1] == 0) MPI_Scatter(A, number_of_rows * n2, MPI_DOUBLE, local_A, number_of_rows * n2, MPI_DOUBLE, 0, row_comm);

    MPI_Datatype COLS_FROM_B;
    MPI_Type_vector(n2, number_of_cols, n3, MPI_DOUBLE, &COLS_FROM_B);
    MPI_Type_commit(&COLS_FROM_B);
    ///Ы
    if(rank == 0) {
        printf("time is:\n", MPI_Wtime() - start);
        free(A); free(B); free(C);
    }

    free(local_A); free(local_B); free(local_C);


    return 0;
}
