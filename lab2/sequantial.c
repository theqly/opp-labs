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

void mul_matrix_on_vec(const double* matrix, const double* vec, double* result, const int n) {
    for(int i = 0; i < n; ++i) {
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

    double* u = malloc(sizeof(double) * n);

    for(int i = 0; i < n; ++i) {
        u[i] = sin(2 * M_PI * i / n);
        x[i] = 0;
    }

    mul_matrix_on_vec(a, u, b, n);
    free(u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const double start = MPI_Wtime();
    const double e = 0.00001;
    const int n = 11520;

    double* a = malloc(sizeof(double) * n * n);
    double* b = malloc(sizeof(double) * n);

    double* x = malloc(sizeof(double) * n);
    double* r = malloc(sizeof(double) * n);
    double* z = malloc(sizeof(double) * n);

    double* tmp1 = malloc(sizeof(double) * n);
    double* tmp2 = malloc(sizeof(double) * n);

    filling(a, b, x, n);

    const double b_norm = get_norm(b, n);

    mul_matrix_on_vec(a, x, tmp1, n);
    vector_sub(b, tmp1, r, n);
    memcpy(z, r, n * sizeof(double));

    char match = 0;

    for(int i = 0; i < 10000; ++i) {
        mul_matrix_on_vec(a, z, tmp1, n);
        const double scalar_of_r = scalar_mul(r, r, n);
        double aplha = scalar_of_r / scalar_mul(tmp1, z, n);

        mul_vec_on_scalar(z, aplha, tmp2, n);
        vector_sum(x, tmp2, x, n);

        mul_vec_on_scalar(tmp1, aplha, tmp2, n);
        vector_sub(r, tmp2, r, n);

        //printf("r_next norm=%lf\n", get_norm(r, n));
        double condition = get_norm(r, n) / b_norm;
        if(condition < e) {
            match++;
            if(match == 3) {
                printf("done on i=%d\n", i);
                printf("condition=%lf\n", condition);
                break;
            }
        } else match = 0;

        double beta = scalar_mul(r, r, n) / scalar_of_r;

        mul_vec_on_scalar(z, beta, tmp1, n);
        vector_sum(r, tmp1, z, n);
    }

    if(match < 3) printf("doesnt working\n");
    else {
        printf("Time taken: %lf sec.\n", MPI_Wtime() - start);
    }

    free(a);
    free(b);

    free(x);
    free(r);
    free(z);

    free(tmp1);
    free(tmp2);

    MPI_Finalize();

    return 0;
}