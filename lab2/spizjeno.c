#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void fillA(double *a, int n);

void fillB(double *b, int n);

void print(double *vec, int n);

void fillutil(int *sendcount, int *displs, int size, int n);

void mult_matr_vec(double *matrix, double *vector, double *res, int sendcount, int n);

void mult_vect_scalar(double *vect, double scalar, double *res, int n);

double scalar(double *left, double *right, int n);

void sum(double *left, double *right, double *res, int n);

void substract(double *left, double *right, double *res, int n);

double normnosqrt(double *vec, int n);


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    // n - num of elements in vector
    // k - num of iterations in algo
    int n = atoi(argv[1]);
    int k = 100000;
    double epsilon = 0.00001;

    int rank, size;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // for vectors scatter
    int *sendcount = calloc(size, sizeof(int));
    int *displs = calloc(size, sizeof(int));
    fillutil(sendcount, displs, size, n);

    // for matrix scatter
    int *sendcmatr = calloc(size, sizeof(int));
    int *displsmatr = calloc(size, sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcmatr[i] = n * sendcount[i];
    }
    for (int i = 1; i < size; i++) {
        displsmatr[i] = displsmatr[i - 1] + sendcmatr[i - 1];
    }

    double *a, *parta, *b, *partb;
    double *xo, *ro, *zo;
    double *xk, *rk, *zk;
    double *tmpvec;

    xo = calloc(n, sizeof(double));
    ro = calloc(sendcount[rank], sizeof(double));
    zo = calloc(n, sizeof(double));

    xk = calloc(sendcount[rank], sizeof(double));
    rk = calloc(sendcount[rank], sizeof(double));
    zk = calloc(sendcount[rank], sizeof(double));

    parta = calloc(sendcmatr[rank], sizeof(double));
    partb = calloc(sendcount[rank], sizeof(double));

    tmpvec = calloc(sendcount[rank], sizeof(double));
    double starttime, endtime;

    if (rank == 0) {
        a = calloc(n * n, sizeof(double));
        b = calloc(n, sizeof(double));
        fillA(a, n);
        fillB(b, n);
        starttime = MPI_Wtime();
    }
    // bcast xo everywhere, xo is needed in (1) multiplying
    MPI_Bcast(xo, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // scattering b
    MPI_Scatterv(b, sendcount, displs, MPI_DOUBLE, partb, sendcount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Scatter partialy a between all processes
    MPI_Scatterv(a, sendcmatr, displsmatr, MPI_DOUBLE, parta, sendcmatr[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // count norm of b
    double bnsq = 0;
    double bnorm = 0;
    double partnorm = normnosqrt(partb, sendcount[rank]);
    MPI_Reduce(&partnorm, &bnsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        bnorm = sqrt(bnsq);
    }
    MPI_Bcast(&bnorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // for debugging
    for (int i = 0; i < n; i++) xo[i] = i;

    // prepare : r0 = b - Ax0
    mult_matr_vec(parta, xo, tmpvec, sendcount[rank], n);
    substract(partb, tmpvec, ro, sendcount[rank]);

    // prepare : z0 = r0
    MPI_Allgatherv(ro, sendcount[rank], MPI_DOUBLE, zo, sendcount, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    // important : ro ~ r^(k-1), xo ~ x^(k-1), etc.
    for (int i = 0; i < k; i++) {
        // 1. alpha_k = (r^(k-1),r^(k-1)) / (Az^(k-1),z^(k-1))

        // collecting numerator (scalar of (r,r)) for 1st step
        double x = scalar(ro, ro, sendcount[rank]);
        double alptop = 0;
        MPI_Allreduce(&x, &alptop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // getting part of A*z^(k-1)
        mult_matr_vec(parta, zo, tmpvec, sendcount[rank], n);
        double y = scalar(tmpvec, &zo[displs[rank]], sendcount[rank]);
        double alpdown = 0;
        MPI_Allreduce(&y, &alpdown, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alphak = alptop / alpdown;

        // 2. xk = x^(k-1) + alpha_k * z^(k-1)
        mult_vect_scalar(&zo[displs[rank]], alphak, tmpvec, sendcount[rank]);
        sum(&xo[displs[rank]], tmpvec, xk, sendcount[rank]);

        // 3. rk = r^(k-1) - alpha_k * A * z^(k-1)
        mult_matr_vec(parta, zo, tmpvec, sendcount[rank], n);
        mult_vect_scalar(tmpvec, alphak, tmpvec, sendcount[rank]);
        substract(ro, tmpvec, rk, sendcount[rank]);


        // this is unchecked because rk is 0 on first iteration!!!
        // 4. beta_k = (rk,rk) / (r^(k-1), r^(k-1))
        double part = scalar(rk, rk, sendcount[rank]);
        double bettop = 23;
        MPI_Allreduce(&part, &bettop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double betak = bettop / alptop;

        // 5. zk = rk + beta_k * z^(k-1)
        mult_vect_scalar(&zo[displs[rank]], betak, tmpvec, sendcount[rank]);
        sum(rk, tmpvec, zk, sendcount[rank]);

        // break condition
        double rnsq = 0;
        double rknorm = 0;
        double rpart = normnosqrt(rk, sendcount[rank]);
        double cond = 0;
        MPI_Reduce(&rpart, &rnsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            rknorm = sqrt(rnsq);
            cond = rknorm / bnorm;
        }
        MPI_Bcast(&cond, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (cond < epsilon) {
            // if (rank == 0) printf("i: %d\n",i);
            break;
        }

        // xk -> xo, rk -> ro, zk -> zo
        // xo, zo have n size, others sendcount[rank]
        memcpy(ro, rk, sendcount[rank] * sizeof(double));
        // gathering parts of xk from processes to xo, every process will have full xo = previous xk
        MPI_Allgatherv(xk, sendcount[rank], MPI_DOUBLE, xo, sendcount, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(zk, sendcount[rank], MPI_DOUBLE, zo, sendcount, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    // gathering partial results from xk's to xo in root process to print it later
    MPI_Gatherv(xk, sendcount[rank], MPI_DOUBLE, xo, sendcount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        endtime = MPI_Wtime();
        printf("Time taken: %f\n", endtime - starttime);
        free(a);
        free(b);
    }

    free(partb);
    free(parta);
    free(sendcount);
    free(displs);
    free(sendcmatr);
    free(displsmatr);

    free(xo);
    free(zo);
    free(ro);

    free(xk);
    free(rk);
    free(zk);

    free(tmpvec);

    MPI_Finalize();
    return 0;
}


void fillA(double *a, int n) {
    // test case
    // for (int i = 0; i < n*n; i++){
    // if (i / n == i % n) a[i] = 2.0;
    // else a[i] = 1.0;
    // }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            a[i * n + j] = (double) ((double) rand() / RAND_MAX * 10);
            a[j * n + i] = a[i * n + j];
        }
    }
}

void fillB(double *b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = n + 1;
    }
}

void print(double *vec, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
}

void fillutil(int *sendcount, int *displs, int size, int n) {
    for (int i = 0; i < n; i++) {
        (sendcount[i % size]) += 1;
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        (displs[i]) = displs[i - 1] + sendcount[i - 1];
    }
}

void substract(double *left, double *right, double *res, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = left[i] - right[i];
    }
}

void mult_matr_vec(double *matrix, double *vector, double *res, int sendcount, int n) {
    for (int i = 0; i < sendcount; i++) {
        res[i] = 0;
    }
    for (int i = 0; i < sendcount * n; i++) {
        res[i / n] += matrix[i] * vector[i % n];
    }
}

double scalar(double *left, double *right, int n) {
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += left[i] * right[i];
    }
    return res;
}

void mult_vect_scalar(double *vect, double scalar, double *res, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = vect[i] * scalar;
    }
}

void sum(double *left, double *right, double *res, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = left[i] + right[i];
    }
}

double normnosqrt(double *vec, int n) {
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += (vec[i] * vec[i]);
    }
    return res;
}
