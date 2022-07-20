#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include<iostream>

using namespace std;


typedef struct {
    int *A, *B, *C;
    int n, m;
} DataSet;

void fillDataSet(DataSet *dataSet);

void printDataSet(DataSet dataSet);

void closeDataSet(DataSet dataSet);

double add(DataSet dataSet);

double add_collapse(DataSet dataSet);

int *add_serial(DataSet dataset, double elapsed_parallel, double elapsed_collapsed);

void printC(int *C, int n, int m);

int main(int argc, char *argv[]) {

#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif

    DataSet dataSet;

    int dim = 500;

    dataSet.n = dim;
    dataSet.m = dim;

    fillDataSet(&dataSet);

    omp_set_nested(1);
    omp_set_num_threads(16);

    double elapsed_parallel = add(dataSet);

    double elapsed_collapsed = add_collapse(dataSet);

    int *C = add_serial(dataSet, elapsed_parallel, elapsed_collapsed);

    /*printC(C, dataSet.n, dataSet.m);
    printDataSet(dataSet);*/


    closeDataSet(dataSet);
    //system("PAUSE");
    return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
    int i, j;

    dataSet->A = (int *) malloc(sizeof(int) * dataSet->n * dataSet->m);
    dataSet->B = (int *) malloc(sizeof(int) * dataSet->n * dataSet->m);
    dataSet->C = (int *) malloc(sizeof(int) * dataSet->n * dataSet->m);

    srand(time(NULL));

#pragma omp parallel for
    for (i = 0; i < dataSet->n; i++) {
#pragma omp parallel for
        for (j = 0; j < dataSet->m; j++) {
            dataSet->A[i * dataSet->m + j] = rand() % 100;
            dataSet->B[i * dataSet->m + j] = rand() % 100;
        }
    }
}

void printDataSet(DataSet dataSet) {
    int i, j;

    printf("[-] Matrix A\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-4d", dataSet.A[i * dataSet.m + j]);
        }
        putchar('\n');
    }

    printf("[-] Matrix B\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-4d", dataSet.B[i * dataSet.m + j]);
        }
        putchar('\n');
    }

    printf("[-] Matrix C\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-8d", dataSet.C[i * dataSet.m + j]);
        }
        putchar('\n');
    }
}

void closeDataSet(DataSet dataSet) {
    free(dataSet.A);
    free(dataSet.B);
    free(dataSet.C);
}

double add(DataSet dataSet) {
    double start = omp_get_wtime();

    int i, j;


#pragma omp parallel for schedule(static, 128)
    for (i = 0; i < dataSet.n; i++) {
#pragma omp parallel for schedule(static, 128)
        for (j = 0; j < dataSet.m; j++) {
            dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("Elapsed Time Parallel %f\n", elapsed);

    return elapsed;
}

double add_collapse(DataSet dataSet) {
    double start = omp_get_wtime();

    int i, j;


#pragma omp parallel for collapse(2) schedule(static, 128)
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("Elapsed Time Parallel collapsed %f\n", elapsed);

    return elapsed;
}

int *add_serial(DataSet dataSet, double elapsed_parallel, double elapsed_collapsed) {
    double start = omp_get_wtime();

    int *C = (int *) malloc(sizeof(int) * dataSet.n * dataSet.m);

    int i, j;
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("Elapsed Time Serial %f\n", elapsed);

    cout << "Speed up parallel: " << elapsed / elapsed_parallel << endl;
    cout << "Speed up collapsed: " << elapsed / elapsed_collapsed << endl;


    return C;
}

void printC(int *C, int n, int m) {
    printf("[-] Matrix C\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%-8d", C[i * m + j]);
        }
        putchar('\n');
    }
}


