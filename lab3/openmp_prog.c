#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "omp.h"


void MatrixOnVector(double* matrix, double* vec, double* res, int size){
#pragma omp for schedule(runtime)
    for(int i = 0; i < size; i++){
        res[i] = 0;
        for(int j = 0; j < size; j++){
            res[i] += matrix[i*size+j] * vec[j];
        }
    }
}

void VecMinusVec(double* vec1, double* vec2, double* res, int size){
#pragma omp for schedule(runtime)
    for(int i = 0; i < size; i++){
        res[i] = vec1[i] - vec2[i];
    }
}

void VectorOnScalar(double* vec, double* res, double scalar, int size){
#pragma omp for schedule(runtime)
    for(int i = 0; i < size; i++){
        res[i] = vec[i] * scalar;
    }
}

void Filling(double* matrix, double* vec, double* x_n, int size){
    for(int i = 0; i < size; i++){
        srand(i);
        for(int j = 0; j < size; j++){
            if(i == j){
                matrix[i*size+j] = (rand() % 10) + size/20;
            } else {
                matrix[i*size+j] = (rand() % 10);
            }
        }
        x_n[i] = 0;
    }

    double* u = (double*)malloc(sizeof(double) * size);

    for(int i = 0; i < size; ++i) {
        u[i] = sin(2 * M_PI * i / size)+ rand() % 10;
    }

    MatrixOnVector(matrix, u, vec, size);
    free(u);
}


int main(int argc, char **argv){
    int N = 6144;
    double epsilon = 0.00001;
    double tao;
    double ending_coeff = 0;
    double start_time;
    double end_time;

    double* A = (double*)malloc(sizeof(double) * N*N);
    double* b = (double *)malloc(sizeof(double) * N);
    double* x_n = (double *)malloc(sizeof(double) * N);
    double* y = (double *)malloc(sizeof(double) * N);

    memset(x_n, 1, sizeof(double) * N);

    double* tmp = (double *)malloc(sizeof(double) * N);
    Filling(A, b, x_n, N);

    double res_of_mul_one = 0;
    double res_of_mul_two = 0;
    double res_of_norm_one = 0;
    double res_of_norm_two = 0;
    start_time = omp_get_wtime();
#pragma omp parallel private(ending_coeff)
    {
#pragma omp for reduction(+:res_of_norm_one) schedule(runtime)
        for(int i = 0; i < N; i++){
            res_of_norm_one += b[i] * b[i];
        }
        double b_norm = sqrt(res_of_norm_one);

        for( int i = 0; i < 10000; i++ ){
            MatrixOnVector(A, x_n, tmp, N);
            VecMinusVec(tmp, b, y, N);
            res_of_norm_two = 0;
#pragma omp for reduction(+:res_of_norm_two) schedule(runtime)
            for(int j = 0; j < N; j++){
                res_of_norm_two += y[j] * y[j];
            }

            double index = sqrt(res_of_norm_two) / b_norm;
            if( index <= epsilon ){
                ending_coeff++;
                if( ending_coeff == 3 ){
                    //printf("iterations: %d", i);
                    break;
                }
            } else ending_coeff = 0;
            MatrixOnVector(A, y, tmp, N);
            res_of_mul_one = 0;
#pragma omp for reduction(+:res_of_mul_one) schedule(runtime)
            for(int j = 0; j < N; j++){
                res_of_mul_one += y[j] * tmp[j];
            }
            res_of_mul_two = 0;
#pragma omp for reduction(+:res_of_mul_two) schedule(runtime)
            for(int j = 0; j < N; j++){
                res_of_mul_two += tmp[j] * tmp[j];
            }
            tao = res_of_mul_one/res_of_mul_two;
            VectorOnScalar(y, tmp, tao, N);
            VecMinusVec(x_n, tmp, x_n, N);
        }
    }
    end_time = omp_get_wtime();
    printf("Время выполнения функции: %f секунд\n", end_time-start_time);
    free(A);
    free(b);
    free(x_n);
    free(y);
    free(tmp);
    return 0;
}