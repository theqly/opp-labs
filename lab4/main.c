#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void local_multiply(double* local_A, double* local_B, double* local_C, const int number_of_rows, const int number_of_cols, const int n2) {
    for(int i = 0; i < number_of_rows; ++i) {
        for(int j = 0; j < number_of_cols; ++j) {
            local_C[i * number_of_cols + j] = 0;
            for(int k = 0; k < n2; ++k) {
                local_C[i * number_of_cols + j] += local_A[i * n2 + k] * local_B[k * number_of_cols + j];
            }
        }
    }
}

double sum_of_matrix_elements(const double* matrix, const int number_of_rows, const int number_of_cols) {
    double res = 0;
    for(int i = 0; i < number_of_rows; ++i) {
        for(int j = 0; j < number_of_cols; ++j) {
            res += matrix[number_of_cols * i + j];
        }
    }
    return res;
}

void print_matrix(double* matrix, const int number_of_rows, const int number_of_cols){
    for(int i = 0; i < number_of_rows; ++i) {
        printf("| ");
        for(int j = 0; j < number_of_cols; ++j) {
            printf("%lf ", matrix[number_of_cols * i + j]);
        }
        printf(" |\n");
    }
}

void filling(double* A, double* B, const int n1, const int n2, const int n3) {
    for (int i = 0; i < n1 * n2; i++) {
        A[i] = 1;
    }
    for (int i = 0; i < n2 * n3; i++) {
        B[i] = 2;
    }
}

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
    const int row_remain_dims[] = {0, 1};
    const int col_remain_dims[] = {1, 0};

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
    MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);

    int col_size;
    MPI_Comm_size(col_comm, &col_size);
    int row_size;
    MPI_Comm_size(row_comm, &row_size);
    if(rank == 0) printf("col size is %d and row size is %d\n", col_size, row_size);

    int coords[ndims];
    MPI_Cart_coords(grid_comm, rank, ndims, coords);

    const int n1 = 3072;
    const int n2 = 2688;
    const int n3 = 2976;

    double *A, *B, *C;
    const int number_of_rows = n1 / p1;
    const int number_of_cols = n3 / p2;

    double* local_A = (double*) malloc(number_of_rows * n2 * sizeof(double));
    double* local_B = (double*) malloc(n2 * number_of_cols * sizeof(double));
    double* local_C = (double*) malloc(number_of_rows * number_of_cols * sizeof(double));
    memset(local_A, 1, number_of_rows * n2 * sizeof(double ));
    memset(local_B, 1, n2 * number_of_cols * sizeof(double));
    memset(local_C, 1, number_of_rows * number_of_cols * sizeof(double));

    double start;
    if(rank == 0) {
        A = (double*) malloc(n1 * n2 * sizeof(double));
        B = (double*) malloc(n2 * n3 * sizeof(double));
        C = (double*) malloc(n1 * n3 * sizeof(double));
        memset(C, 1, n1 * n3 * sizeof(double));
        filling(A, B, n1, n2, n3);
        start = MPI_Wtime();
    }

    if(coords[1] == 0) MPI_Scatter(A, number_of_rows * n2, MPI_DOUBLE, local_A, number_of_rows * n2, MPI_DOUBLE, 0, col_comm);

    MPI_Datatype COLS_FROM_B;
    MPI_Type_vector(n2, number_of_cols, n3, MPI_DOUBLE, &COLS_FROM_B);
    MPI_Type_commit(&COLS_FROM_B);
    if(rank == 0) {
        for(int i = 1; i < row_size; ++i) {
            MPI_Send(B + i * number_of_cols, 1, COLS_FROM_B, i, 0, row_comm);
        }

        for(int i = 0; i < n2; ++i) {
            for(int j = 0; j < number_of_cols; ++j) {
                local_B[i* number_of_cols + j] = B[i * n3 + j];
            }
        }

    } else if(coords[0] == 0) {
        MPI_Recv(local_B, number_of_cols * n2, MPI_DOUBLE, 0, 0, row_comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(local_A, number_of_rows * n2, MPI_DOUBLE, 0, row_comm);
    MPI_Bcast(local_B, n2 * number_of_cols, MPI_DOUBLE, 0, col_comm);

    local_multiply(local_A, local_B, local_C, number_of_rows, number_of_cols, n2);

    MPI_Datatype LOCAL_C;
    MPI_Type_vector(number_of_rows, number_of_cols, n3, MPI_DOUBLE, &LOCAL_C);
    MPI_Type_commit(&LOCAL_C);

    if(rank != 0) {
        MPI_Send(local_C, number_of_cols * number_of_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
        for(int i = 0; i < number_of_rows; ++i) {
            for(int j = 0; j < number_of_cols; ++j) {
                C[i * n3 + j] = local_C[i * number_of_cols + j];
            }
        }

        for(int i = 0; i < p1; ++i) {
            for(int j = 0; j < p2; ++j) {
                if(i == 0 && j == 0) continue;
                MPI_Recv(C + (i * n3 * number_of_rows + j * number_of_cols), 1, LOCAL_C, i * p2 + j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        printf("time is: %lf\n", MPI_Wtime() - start);
        printf("sum is %lf\n", sum_of_matrix_elements(C, n1, n3));
        //print_matrix(C, n1, n3);
        free(A); free(B); free(C);
    }


    free(local_A); free(local_B); free(local_C);

    MPI_Type_free(&COLS_FROM_B);
    MPI_Type_free(&LOCAL_C);

    MPI_Finalize();
    return 0;
}
