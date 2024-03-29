#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


void vector_sum(const double* vec1, const double* vec2, double* result, const int n) {
    for(int i = 0; i < n; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}

void vector_sub(const double* vec1, const double* vec2, double* result, const int n) {
    for(int i = 0; i < n; ++i) {
        result[i] = vec1[i] - vec2[i];
    }
}

void mul_vec_on_scalar(const double* vec, const double scalar, double* result, const int n) {
    for(int i = 0; i < n; ++i) {
        result[i] = vec[i] * scalar;
    }
}

void mul_matrix_on_vec(const double* matrix, const double* vec, double* result, const int n, const int size_of_block) {
    for(int i = 0; i < size_of_block; ++i) {
        result[i] = 0;
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[i*n + j] * vec[j];
        }
    }
}

double scalar_mul(const double* vec1, const double* vec2, const int n) {
    double result = 0;
    for (int i = 0; i < n; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

double get_norm(double* vec, const int n) {
    double result = 0;
    for(int i = 0; i < n; ++i) {
        result += vec[i] * vec[i];
    }
    return sqrt(result);
}

void filling(double* a, double* b, double* x, const int n) {
    for(int i = 0; i < n; ++i) {
        srand(i);
        for(int j = 0; j < n; ++j) {
            if(i == j) a[i*n + j] = rand() % (n / 3) + n;
            else a[i*n + j] = n + 1;
        }
        b[i] = n + 1;
    }

    double* u = (double*)malloc(sizeof(double) * n);

    for(int i = 0; i < n; ++i) {
        u[i] = sin(2 * M_PI * i / n);
        x[i] = 0;
    }

    mul_matrix_on_vec(a, u, b, n, n);
    free(u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start;
    const double e = 0.00001;
    const int n = 11520;

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int size_of_block = n / size;
    const int size_of_matrix_block = (n * n) / size;

    double* a;
    double* b;

    double* local_a = (double*)malloc(sizeof(double) * size_of_matrix_block);
    double* local_b = (double*)malloc(sizeof(double) * size_of_block);

    double* x = (double*)malloc(sizeof(double) * n);
    double* r = (double*)malloc(sizeof(double) * size_of_block);
    double* z = (double*)malloc(sizeof(double) * n);

    double* local_z = (double*)malloc(sizeof(double) * size_of_block);

    double* tmp1 = (double*)malloc(sizeof(double) * size_of_block);
    double* tmp2 = (double*)malloc(sizeof(double) * size_of_block);

    if(rank == 0) {
        a = (double*)malloc(sizeof(double) * n * n);
        b = (double*)malloc(sizeof(double) * n);
        filling(a, b, x, n);
        start = MPI_Wtime();
    }

    MPI_Scatter(a, size_of_matrix_block, MPI_DOUBLE, local_a, size_of_matrix_block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, size_of_block, MPI_DOUBLE, local_b, size_of_block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double b_norm = 0;
    double local_b_norm = scalar_mul(local_b, local_b, size_of_block);
    MPI_Allreduce(&local_b_norm, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    b_norm = sqrt(b_norm);

    mul_matrix_on_vec(local_a, x, tmp1, n, size_of_block);
    vector_sub(local_b, tmp1, r, size_of_block);

    MPI_Allgather(r, size_of_block, MPI_DOUBLE, z, size_of_block, MPI_DOUBLE, MPI_COMM_WORLD);

    char match = 0;
    for(int i = 0; i < 10000; ++i) {
        mul_matrix_on_vec(local_a, z, tmp1, n, size_of_block);

        const double local_scalar_of_r = scalar_mul(r, r, size_of_block);
        double scalar_of_r = 0;
        MPI_Allreduce(&local_scalar_of_r, &scalar_of_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double local_tmp = scalar_mul(tmp1, z + size_of_block*rank, size_of_block);
        double tmp = 0;
        MPI_Allreduce(&local_tmp, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double aplha = scalar_of_r / tmp;

        mul_vec_on_scalar(z + size_of_block*rank, aplha, tmp2, size_of_block);
        vector_sum(x + size_of_block*rank, tmp2, x + size_of_block*rank, size_of_block);

        mul_vec_on_scalar(tmp1, aplha, tmp2, size_of_block);
        vector_sub(r, tmp2, r, size_of_block);

        double new_scalar_of_r = 0;
        double new_local_scalar_of_r = scalar_mul(r, r, size_of_block);
        MPI_Allreduce(&new_local_scalar_of_r, &new_scalar_of_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double condition = 0;
        const double r_norm = sqrt(new_scalar_of_r);
        //printf("r norm=%lf\n", r_norm);
        condition = r_norm / b_norm;

        if(condition < e) {
            match++;
            if(match == 3) {
                if(rank == 0) {
                    printf("done on i=%d\n", i);
                    printf("condition=%lf\n", condition);
                }
                break;
            }
        } else match = 0;

        double beta = new_scalar_of_r / scalar_of_r;

        mul_vec_on_scalar(z + size_of_block*rank, beta, tmp1, size_of_block);
        vector_sum(r, tmp1, local_z, size_of_block);

        MPI_Allgather(local_z, size_of_block, MPI_DOUBLE, z, size_of_block, MPI_DOUBLE, MPI_COMM_WORLD);
    }


    if(rank == 0) {
        if(match < 3) printf("doesnt working\n");
        else {
            printf("Time taken: %lf sec.\n", MPI_Wtime() - start);
        }
        free(a);
        free(b);
    }

    free(local_a);
    free(local_b);

    free(x);
    free(r);
    free(z);

    free(local_z);

    free(tmp1);
    free(tmp2);

    MPI_Finalize();

    return 0;
}