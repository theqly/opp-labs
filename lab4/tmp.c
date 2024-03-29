#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void fillA(int *a, const int n1, const int n2);

void fillB(int *b, const int n2, const int n3);

void printMatr(int *m, const int n1, const int n2);

void multiply(int *partA, int *partB, int *partC, const int diva, const int divb, const int n2);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm grid_comm;
    MPI_Comm rows_comm;
    MPI_Comm cols_comm;
    // 2D array of processes
    const int ndims = 2;
    // 2 x 3
    int p1 = atoi(argv[1]);
    int p2 = atoi(argv[2]);
    const int dims[] = {p1, p2};
    const int periods[] = {0, 0};
    const int reorder = 0;
    const int rows_remain[] = {1, 0};
    const int cols_remain[] = {0, 1};
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    MPI_Cart_sub(grid_comm, rows_remain, &rows_comm);
    MPI_Cart_sub(grid_comm, cols_remain, &cols_comm);

    int coords[ndims];
    MPI_Cart_coords(grid_comm, rank, ndims, coords);
    MPI_Cart_sub(grid_comm, rows_remain, &rows_comm);
    MPI_Cart_sub(grid_comm, cols_remain, &cols_comm);
    //printf("Rank %d has coordinates (%d, %d)\n", rank, coords[0], coords[1]);

    int rows_rank, rows_size;
    MPI_Comm_rank(rows_comm, &rows_rank);
    MPI_Comm_size(rows_comm, &rows_size);

    int cols_rank, cols_size;
    MPI_Comm_rank(cols_comm, &cols_rank);
    MPI_Comm_size(cols_comm, &cols_size);
    MPI_Status status;

    // KEEP THIS!!!
    int n1 = 4608;
    int n2 = 4000;
    int n3 = 4320;

    int *A;
    int *B;
    int *C;

    int diva = n1 / p1;
    int divb = n3 / p2;
    if (rank == 0) {
        A = calloc(n1 * n2, sizeof(int));
        B = calloc(n2 * n3, sizeof(int));
        C = calloc(n1 * n3, sizeof(int));
        fillA(A, n1, n2);
        fillB(B, n2, n3);
    }

    int *partA = calloc(diva * n2, sizeof(int));
    int *partB = calloc(divb * n2, sizeof(int));
    int *partC = calloc(diva * divb, sizeof(int));

    double starttime, endtime;
    if (rank == 0) {
        starttime = MPI_Wtime();
    }

    if (cols_rank == 0) MPI_Scatter(A, diva * n2, MPI_INT, partA, diva * n2, MPI_INT, 0, rows_comm);

    MPI_Datatype BCOL_TYPE;
    // count of blocks, blocklen, stride between beginning of blocks, oldtype, newtype
    MPI_Type_vector(n2, divb, n3, MPI_INT, &BCOL_TYPE);
    MPI_Type_commit(&BCOL_TYPE);

    MPI_Datatype BRECV_TYPE;
    MPI_Type_vector(1, divb * n2, 0, MPI_INT, &BRECV_TYPE);
    MPI_Type_commit(&BRECV_TYPE);


    //printf("cols size: %d\n",cols_size); for 2 4 it's 4
    if (rank == 0) {
        int c = 0;
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < divb; j++) {
                partB[c] = B[i * n3 + j];
                c++;
            }
        }
        for (int i = 1; i < cols_size; i++) {
            MPI_Send(&B[i * divb], 1, BCOL_TYPE, i, 10, cols_comm);
        }
    } else if (coords[0] == 0) {
        // MPI_Recv(partB,divb*n2,MPI_INT,0,10,cols_comm,&status);
        MPI_Recv(partB, 1, BRECV_TYPE, 0, 10, cols_comm, &status);
    }

    //printf("BEFORE (%d,%d):\n",coords[0],coords[1]);
    //printMatr(partA,diva,n2);
    //printMatr(partB,n2,divb);

    MPI_Bcast(partA, diva * n2, MPI_INT, 0, cols_comm);
    MPI_Bcast(partB, n2 * divb, MPI_INT, 0, rows_comm);

    //printf("AFTER (%d,%d):\n",coords[0],coords[1]);
    //printMatr(partA,diva,n2);
    //printMatr(partB,n2,divb);


    multiply(partA, partB, partC, diva, divb, n2);

    // if (rank == 2){
    // printf("partA:\n");
    // printMatr(partA,diva,n2);
    // printf("partB:\n");
    // printMatr(partB,n2,divb);
    // printf("partC:\n");
    // printMatr(partC,diva,divb);
    // }

    MPI_Datatype CPART_TYPE;
    MPI_Type_vector(diva, divb, n3, MPI_INT, &CPART_TYPE);
    MPI_Type_commit(&CPART_TYPE);

    if (rank != 0) {
        MPI_Send(partC, diva * divb, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        for (int i = 0; i < diva; i++) {
            for (int j = 0; j < divb; j++) {
                C[i * n3 + j] = partC[i * divb + j];
            }
        }

        for (int i = 0; i < p1; i++) {
            for (int j = 0; j < p2; j++) {
                if (i == 0 && j == 0) continue;
                // j * divb - width of a partC block
                MPI_Recv(&C[i * n3 * diva + j * divb], 1, CPART_TYPE, i * p2 + j, 12, MPI_COMM_WORLD, &status);
            }
        }
        endtime = MPI_Wtime();
        printf("Time taken: %f seconds\n", endtime - starttime);

        // printf("A:\n");
        // printMatr(A,n1,n2);

        // printf("B:\n");
        // printMatr(B,n2,n3);

        // printf("C:\n");
        // printMatr(C,n1,n3);

        free(A);
        free(B);
        free(C);
    }

    free(partA);
    free(partB);
    free(partC);

    MPI_Finalize();
    return 0;
}

void fillA(int *a, const int n1, const int n2) {
    for (int i = 0; i < n1 * n2; i++) {
        a[i] = i;
    }
}

void fillB(int *b, const int n2, const int n3) {
    for (int i = 0; i < n2 * n3; i++) {
        b[i] = i;
    }
}

void printMatr(int *m, const int n1, const int n2) {
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%d ", m[i * n2 + j]);
        }
        printf("\n");
    }
}

void multiply(int *partA, int *partB, int *partC, const int diva, const int divb, const int n2) {
    for (int i = 0; i < diva; i++) {
        for (int j = 0; j < divb; j++) {
            partC[i * divb + j] = 0;
            for (int k = 0; k < n2; k++) {
                partC[i * divb + j] += partA[i * n2 + k] * partB[k * divb + j];
            }
        }
    }
}
