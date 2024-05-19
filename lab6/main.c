#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <pthread.h>
#include <math.h>

#define HELP_START 333
#define HELP_COUNT 444
#define SENDING_TASKS 555

#define L 2048
#define TASKS 1024

struct data{
    int size, rank;
    double result;
    int* tasks;
    int tasks_done;
    int tasks_to_do;
    pthread_mutex_t mutex;
};

void filling(struct data* data, int iteration){
    for(int i = 0; i < TASKS; ++i){
        data->tasks[i] = abs(50 - i % 100) * abs(data->rank - (iteration % data->size)) * L;
    }
}

void calc_tasks(struct data* data){
    int weight;
    for(int i = 0; i < data->tasks_to_do; ++i){
        pthread_mutex_lock(&data->mutex);
        weight = data->tasks[i];
        pthread_mutex_unlock(&data->mutex);
        for (int j = 0; j < weight; ++j) {
            data->result += sin(j);
        }
        pthread_mutex_lock(&data->mutex);
        data->tasks_done++;
        pthread_mutex_unlock(&data->mutex);
    }
    data->tasks_to_do = 0;
}

void* worker_func(void* void_data){
    struct data* data = (struct data*) void_data;
    data->tasks = malloc(sizeof(int) * TASKS);

    double start_time, end_time;
    double min_time, max_time;
    double disbalance, dolya_disbalace;

    for(int i = 0; i < 12; i++){
        pthread_mutex_lock(&data->mutex);
        filling(data, i);
        data->tasks_done = 0;
        data->tasks_to_do = TASKS;
        pthread_mutex_unlock(&data->mutex);
        start_time = MPI_Wtime();
        calc_tasks(data);
        //printf("%d own tasks done\n", data->rank);

        int working = 1;
        int to_receive = 0;
        while(working){
            int count_of_done = 0;
            for(int j = 0; j < data->size; ++j){
                if(j == data->rank) continue;

                MPI_Send(&data->rank, 1, MPI_INT, j, HELP_START, MPI_COMM_WORLD);
                MPI_Recv(&to_receive, 1, MPI_INT, j, HELP_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if(to_receive == 0) count_of_done++;

                if (count_of_done == data->size-1) {
                    working = 0;
                    break;
                }

                if(to_receive == 0) continue;
                MPI_Recv(data->tasks, to_receive, MPI_INT, j, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pthread_mutex_lock(&data->mutex);
                data->tasks_to_do = to_receive;
                pthread_mutex_unlock(&data->mutex);
                calc_tasks(data);
            }
        }

        end_time = MPI_Wtime();
        double local_time = end_time - start_time;
        MPI_Allreduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        printf("rank = %d; iteration = %d; time = %lf; tasks done = %d; result = %lf\n", data->rank, i, local_time, data->tasks_done, data->result);

        if(data->rank==0) {
            disbalance = max_time - min_time;
            dolya_disbalace = (disbalance/max_time) * 100;
            printf("disbalace: %lf and dolya: %lf at iteration %d\n", disbalance, dolya_disbalace, i);
        }
    }

    int finish = 999;
    MPI_Send(&finish, 1, MPI_INT, data->rank, HELP_START, MPI_COMM_WORLD);
    free(data->tasks);
    pthread_exit(NULL);
}

int main(int argc, char** argv){
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided != MPI_THREAD_MULTIPLE){
        fprintf(stderr, "MPI_THREAD_MULTIPLE not setted\n");
        MPI_Finalize();
        return 1;
    }
    struct data* data = calloc(1, sizeof(struct data));
    MPI_Comm_size(MPI_COMM_WORLD, &data->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &data->rank);
    pthread_mutex_init(&data->mutex, NULL);

    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0){
        fprintf(stderr, "error in attr_init\n");
        MPI_Finalize();
        return 1;
    }

    pthread_t worker;

    if(pthread_create(&worker, &attr, worker_func, data) != 0){
        fprintf(stderr, "error in creating worker\n");
        MPI_Finalize();
        return 1;
    }

    double start = MPI_Wtime();
    int message; // процесс, который закончил работу, или флаг завершения работы
    MPI_Barrier(MPI_COMM_WORLD);
    while(1){
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, HELP_START, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(message == 999) break;
        pthread_mutex_lock(&data->mutex);
        int tasks_remained = data->tasks_to_do - data->tasks_done;
        pthread_mutex_unlock(&data->mutex);
        int tasks_to_send = tasks_remained / (data->size * 2);
        if(tasks_remained >= 1 && tasks_to_send >= 1){
            pthread_mutex_lock(&data->mutex);
            data->tasks_to_do -= tasks_to_send;
            pthread_mutex_unlock(&data->mutex);
            MPI_Send(&tasks_to_send, 1, MPI_INT, message, HELP_COUNT, MPI_COMM_WORLD);
            MPI_Send(&data->tasks[data->tasks_to_do - tasks_to_send], tasks_to_send, MPI_INT, message, SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            tasks_to_send = 0;
            MPI_Send(&tasks_to_send, 1, MPI_INT, message, HELP_COUNT, MPI_COMM_WORLD);
        }

    }

    if(pthread_join(worker, NULL) != 0){
        fprintf(stderr, "error in joining worker\n");
        MPI_Finalize();
        return 1;
    }

    if(data->rank == 0) printf("main time = %lf\n", MPI_Wtime() - start);

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&data->mutex);
    free(data);
    return 0;
}