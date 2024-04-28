#include <stdio.h>
#include <stdlib.h>
#include <openmpi-x86_64/mpi.h>

// RULES
// for living dot : < 2 or > 3 neighbours -> death
// for dead dot : 3 living neighbours -> alive

void fillsizes(int *lines, int n, int fsY);

void initFirstWhole(char *field, int sizeX);

void initFirstTwoLines(char *field, int sizeX);

void initThirdLine(char *field, int sizeX);

void printField(char *field, int fsX, int lines);

void fillostanov(char *ostanov, char *curfield, char **oldfields, int iters, int fsX, int linesperproc);

int cmpfields(char *curfield, char *oldfield, int elems);

int countNeighbours(char *field, int cx, int cy, int fsX);

void calc_without_headtail(char *curfield, char *newfield, int fsX, int linesperproc);

void calc_last_line(char *curfield, char *newfield, int fsX, int linesperproc);

void calc_first_line(char *curfield, char *newfield, int fsX);

int main(int argc, char **argv) {
    const int MAX_ITERS = 1000;
// исходные размеры поля по Х и по Y
    const int fsX = atoi(argv[1]);
    const int fsY = atoi(argv[2]);
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
// кол-во строк поля на ядро (процесс) - заполняем его
    int linesperproc[size];
    fillsizes(linesperproc, size, fsY);

// массив указателей на старые поля с предыдущих итераций
    char *oldfields[MAX_ITERS];

// выделяем памяти сколько требуется + 2 строчки - копии соседских верхних \ нижних
    char *curfield = calloc((linesperproc[rank] + 2) * fsX, sizeof(char));

// инициализация
    if (linesperproc[0] > 2) {
// все стартовые значения влезают на поле 1 ядра
        if (rank == 0) {
            initFirstWhole(&curfield[fsX], fsX);
        }
    } else if (linesperproc[0] <= 2) {
// не влезают
        if (rank == 0) {
            initFirstTwoLines(&curfield[fsX], fsX);

        }
        if (rank == 1) {
            initThirdLine(&curfield[fsX], fsX);
        }
    }

    int prev = (rank + size - 1) % size;
    int next = (rank + 1) % size;

    int bflag = 0;
    int titers = 0;

    double starttime, endtime;
    if (rank == 0) {
        starttime = MPI_Wtime();
    }

    for (int i = 0; i < MAX_ITERS; i++) {
//if (i == 1) break;
        char *newfield = calloc((linesperproc[rank] + 2) * fsX, sizeof(char));
        MPI_Request sreq1, sreq2, rreq3, rreq4, areq6;

//1. Инициализировать отправку первой строки предыдущему ядру, используя
//MPI_Isend. Запомнить параметр request для шага 8.
//2. Инициализировать отправку последней строки последующему ядру, используя
//MPI_Isend. Запомнить параметр request для шага 11.
        MPI_Isend(&curfield[fsX], fsX, MPI_CHAR, prev, 10, MPI_COMM_WORLD, &sreq1);
        MPI_Isend(&curfield[fsX * (linesperproc[rank])], fsX, MPI_CHAR, next, 20, MPI_COMM_WORLD, &sreq2);

//3. Инициировать получение от предыдущего ядра его последней строки,
//используя MPI_Irecv. Запомнить параметр request для шага 9.
//4. Инициировать получение от последующего ядра его первой строки, используя
//MPI_Irecv. Запомнить параметр request для шага 12.
        MPI_Irecv(curfield, fsX, MPI_CHAR, prev, 20, MPI_COMM_WORLD, &rreq3);
        MPI_Irecv(&curfield[fsX * (linesperproc[rank] + 1)], fsX, MPI_CHAR, next, 10, MPI_COMM_WORLD, &rreq4);

// 5. Вычислить вектор флагов останова.
// вектор с i элементами
// ostanov[i] = 1 <=> состояние на текущей итерации совпало с iой итерацией
        char ostanov[i];

        fillostanov(ostanov, curfield, oldfields, i, fsX, linesperproc[rank]);
// обмен остановами
        MPI_Iallreduce(MPI_IN_PLACE, ostanov, i, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD, &areq6);

// 7. Вычислить состояния клеток в строках, кроме первой и последней.
        calc_without_headtail(curfield, newfield, fsX, linesperproc[rank]);

// 8. Дождаться освобождения буфера отправки первой строки предыдущему ядру,
//используя MPI_Wait и сохраненный на шаге 1 параметр request.
// 9. Дождаться получения от предыдущего ядра его последней строки, используя
// MPI_Wait и сохраненный на шаге 3 параметр request.
        MPI_Status status;
        MPI_Wait(&sreq1, &status);
        MPI_Wait(&rreq3, &status);

//10. Вычислить состояния клеток в первой строке.
        calc_first_line(curfield, newfield, fsX);

// 11, 12 - wait
        MPI_Wait(&sreq2, &status);
        MPI_Wait(&rreq4, &status);

//13. Вычислить состояния клеток в последней строке.
        calc_last_line(curfield, newfield, fsX, linesperproc[rank]);

// 14 - wait
        MPI_Wait(&areq6, &status);

//15
        for (int j = 0; j < i; j++) {
            if (ostanov[j] == 1) {
                bflag = 1;
                titers = i;
                break;
            }
        }
        if (bflag == 1) break;

        oldfields[i] = curfield;
        curfield = newfield;

    }

    if (rank == 0) {
        endtime = MPI_Wtime();
//printf("finished on %d iteration\n",titers);
        printf("Time taken: %f\n seconds\n", endtime - starttime);
    }

// почистить все старые поля
    for (int i = 0; i < titers; i++) {
        free(oldfields[i]);
    }
    free(curfield);
    MPI_Finalize();
    return 0;
}

void fillostanov(char *ostanov, char *curfield, char **oldfields, int iters, int fsX, int linesperproc) {
    for (int i = 0; i < iters; i++) {
        char *oldfield = oldfields[i];
        ostanov[i] = cmpfields(&curfield[fsX], &oldfield[fsX], fsX * linesperproc);
    }
}

int cmpfields(char *curfield, char *oldfield, int elems) {
    for (int j = 0; j < elems; j++) {
        if (curfield[j] != oldfield[j]) return 0;
    }
    return 1;
}


void calc_first_line(char *curfield, char *newfield, int fsX) {
    for (int i = 0; i < fsX; i++) {
        int nbrs = countNeighbours(curfield, i, 1, fsX);
        if (curfield[fsX + i] == 0) {
            newfield[fsX + i] = (nbrs == 3) ? 1 : 0;
        } else if (curfield[fsX + i] == 1) {
            newfield[fsX + i] = (nbrs == 3 || nbrs == 2) ? 1 : 0;
        }
    }

}

void calc_last_line(char *curfield, char *newfield, int fsX, int linesperproc) {
    for (int i = 0; i < fsX; i++) {
        int nbrs = countNeighbours(curfield, i, linesperproc, fsX);
        if (curfield[fsX * linesperproc + i] == 0) {
            newfield[fsX * linesperproc + i] = (nbrs == 3) ? 1 : 0;
        } else if (curfield[fsX * linesperproc + i] == 1) {
            newfield[fsX * linesperproc + i] = (nbrs == 3 || nbrs == 2) ? 1 : 0;
        }
    }
}

void calc_without_headtail(char *curfield, char *newfield, int fsX, int linesperproc) {
// 7. Вычислить состояния клеток в строках, кроме первой и последней.
// нам нужна 3я строка - 1ая строка - копия от предыдущего ядра, 2ая - "первая" у нашего ядра
    for (int y = 2; y < linesperproc; y++) {
        for (int x = 0; x < fsX; x++) {
            int nbrs = countNeighbours(curfield, x, y, fsX);
            if (curfield[y * fsX + x] == 0) {
                newfield[y * fsX + x] = (nbrs == 3) ? 1 : 0;
            } else if (curfield[y * fsX + x] == 1) {
                newfield[y * fsX + x] = (nbrs == 3 || nbrs == 2) ? 1 : 0;
            }
        }
    }

}

int countNeighbours(char *field, int cx, int cy, int fsX) {
    int res = 0;
    for (int dy = -1; dy < 2; dy++) {
        for (int dx = -1; dx < 2; dx++) {
            if (dx == dy && dy == 0) {
                continue;
            }
            int px = cx + dx;
            int py = cy + dy;
            if (px < 0) {
                px = fsX - 1;
            } else if (px >= fsX) {
                px = 0;
            }
            res += field[py * fsX + px];
        }
    }
    return res;
}

// INITIALIZATION FUNCS

void fillsizes(int *lines, int n, int fsY) {
    for (int i = 0; i < n; i++) {
        lines[i] = 0;
    }
    for (int i = 0; i < fsY; i++) {
        lines[i % n]++;
    }
}

void initFirstTwoLines(char *field, int sizeX) {
// orig : (1,2) (2,3) (3,1) (3,2) (3,3)
// matr : (0,1) (1,2) (2,0) (2,1) (2,2)
// where first coord is a number of a line
    field[1] = 1;
    field[sizeX + 2] = 1;
}

void initThirdLine(char *field, int sizeX) {
// orig : (1,2) (2,3) (3,1) (3,2) (3,3)
// matr : (0,1) (1,2) (2,0) (2,1) (2,2)
// where first coord is a number of a line
    field[0] = 1;
    field[1] = 1;
    field[2] = 1;

}

void initFirstWhole(char *field, int sizeX) {
// orig : (1,2) (2,3) (3,1) (3,2) (3,3)
// matr : (0,1) (1,2) (2,0) (2,1) (2,2)
// where first coord is a number of a line
    field[1] = 1;
    field[sizeX + 2] = 1;
    field[sizeX * 2] = 1;
    field[sizeX * 2 + 1] = 1;
    field[sizeX * 2 + 2] = 1;

}

void printField(char *field, int fsX, int lines) {
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < fsX; j++) {
            printf("%d ", field[i * fsX + j]);
        }
        printf("\n");
    }
}
