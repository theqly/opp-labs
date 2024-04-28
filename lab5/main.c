#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>

int count_neighbors(char* era, int width, int x, int y){
    int number_of_neighbors = 0;
    for(int dy = -1; dy < 2; ++dy){
        for(int dx = -1; dx < 2; ++dx){
            if(dx == 0 && dy == 0){
                continue;
            }
            int neighbor_x = x + dx;
            int neighbor_y = y + dy;
            if(neighbor_x < 0) neighbor_x = width - 1;
            else if(neighbor_x >= width) neighbor_x = 0;
            number_of_neighbors += era[neighbor_y * width + neighbor_x];
        }
    }
    return number_of_neighbors;
}


void calc_inner(char* cur_era, char* next_era, int width, int number_of_rows){
    for(int y = 2; y < number_of_rows; ++y){
        for(int x = 0; x < width; ++x){
            int number_of_neighbors = count_neighbors(cur_era, width, x, y);
            int position = y * width + x;
            if(cur_era[position] == 0){
                if(number_of_neighbors == 3) next_era[position] = 1;
                else next_era[position] = 0;
            } else {
                if(number_of_neighbors == 2 || number_of_neighbors == 3) next_era[position] = 1;
                else next_era[position] = 0;
            }
        }
    }
}

void calc_top(char* cur_era, char* next_era, int width){
    for(int x = 0; x < width; ++x){
        int number_of_neighbors = count_neighbors(cur_era, width, x, 1);
        int position = width + x;
        if(cur_era[position] == 0){
            if(number_of_neighbors == 3) next_era[position] = 1;
            else next_era[position] = 0;
        } else {
            if(number_of_neighbors == 2 || number_of_neighbors == 3) next_era[position] = 1;
            else next_era[position] = 0;
        }
    }
}

void calc_bottom(char* cur_era, char* next_era, int width, int number_of_rows){
    for(int x = 0; x < width; ++x){
        int number_of_neighbors = count_neighbors(cur_era, width, x, number_of_rows);
        int position = width * number_of_rows + x;
        if(cur_era[position] == 0){
            if(number_of_neighbors == 3) next_era[position] = 1;
            else next_era[position] = 0;
        } else {
            if(number_of_neighbors == 2 || number_of_neighbors == 3) next_era[position] = 1;
            else next_era[position] = 0;
        }
    }
}

char compare(char* era1, char* era2, int size){
    for(int i = 0; i < size; ++i){
        if(era1[i] != era2[i]) return 0;
    }
    return 1;
}

char check(char* cur_era, char** old_eras, int i, int width, int number_of_rows){
    for(int j = 0; j < i; ++j){
        if(compare(&cur_era[width], &(old_eras[j])[width], width * number_of_rows)) return 1;
    }
    return 0;
}

void print(char* era, int width, int height){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            printf("%d ", era[i * width + j]);
        }
        printf("\n");
    }
}




int main(int argc, char** argv){
    if(argc < 3){
        fprintf(stderr, "Usage: ./prog <width> <height>");
        return 1;
    }

    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char* old_eras[10000];

    const int number_of_rows = height / size;
    const int size_of_block = width * (number_of_rows + 2);

    char* cur_era = calloc(size_of_block, sizeof(char));

    if(rank == 0){
        cur_era[width + 1] = 1;
        //cur_era[width*(number_of_rows + 1) + 1] = 1;

        cur_era[width * 2 + 2] = 1;
        cur_era[3 * width] = 1;
        cur_era[3 * width + 1] = 1;
        cur_era[3 * width + 2] = 1;
    }

    int prev_process = (rank - 1 + size) % size;
    int next_process = (rank + 1) % size;

    double start;
    if(rank == 0){
        start = MPI_Wtime();
    }

    //print(cur_era, width, height + 2);
    //printf("--------------------------------\n");

    int i;
    for(i = 0; i < 10000; ++i){
        char *next_era = calloc(size_of_block, sizeof(char));
        MPI_Request r1, r2, r3, r4;

        MPI_Isend(&cur_era[width], width, MPI_CHAR, prev_process, 333, MPI_COMM_WORLD, &r1);
        MPI_Isend(&cur_era[width * number_of_rows], width, MPI_CHAR, next_process, 444, MPI_COMM_WORLD, &r2);

        MPI_Irecv(cur_era, width, MPI_CHAR, prev_process, 444, MPI_COMM_WORLD, &r3);
        MPI_Irecv(&cur_era[width * (number_of_rows + 1)], width, MPI_CHAR, next_process, 333, MPI_COMM_WORLD, &r4);

        calc_inner(cur_era, next_era, width, number_of_rows);


        MPI_Wait(&r3, MPI_STATUS_IGNORE);
        calc_top(cur_era, next_era, width);

        MPI_Wait(&r4, MPI_STATUS_IGNORE);
        calc_bottom(cur_era, next_era, width, number_of_rows);

        MPI_Wait(&r1, MPI_STATUS_IGNORE);
        MPI_Wait(&r2, MPI_STATUS_IGNORE);

        old_eras[i] = cur_era;

        char local_is_repeated = check(cur_era, old_eras, i, width, number_of_rows);
        char is_repeated;
        MPI_Allreduce(&local_is_repeated, &is_repeated, 1, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
        if(is_repeated == size) break;
        cur_era = next_era;
    }

    if(rank == 0){
        printf("i = %d\n", i);
        printf("Time taken: %lf sec.\n", MPI_Wtime() - start);
    }

    for (int j = 0; j < i + 1; ++j) {
        //print(old_eras[j], width, height + 2);
        //printf("--------------------------------\n");
        free(old_eras[j]);
    }

    MPI_Finalize();

    return 0;
}