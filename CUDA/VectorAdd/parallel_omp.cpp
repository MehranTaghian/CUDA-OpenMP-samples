#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


void fillMat(int* v, int size);
void addMat(int* a, int* b, int* c, int size);
void printMat(int* v, int matSizeX, int matSizeY);

int main()
{
    const int size = 1048576;
    int* a;
    int* b;
    int* c;
    a = (int*)malloc(sizeof(int) * size);
    b = (int*)malloc(sizeof(int) * size);
    c = (int*)malloc(sizeof(int) * size);

    fillMat(a, size);
    fillMat(b, size);

    addMat(a, b, c, size);
    /*printMat(a, matSizeX, matSizeY);
    printMat(b, matSizeX, matSizeY);
    printMat(c, matSizeX, matSizeY);*/
    return EXIT_SUCCESS;
}

// Fills a Matrice with data
void fillMat(int* v, int size) {
    static int L = 0;
    for (int i = 0; i < size; i++) {
        v[i] = L++;
    }
}

// Adds two Matrices
void addMat(int* a, int* b, int* c, int size) {
    double begin = omp_get_wtime();
    int i;
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }

    double elapsed = omp_get_wtime() - begin;
    printf("elapsed %f\n", elapsed);
}

// Prints a Matrices to the stdout.
void printMat(int* v, int matSizeX, int matSizeY) {
    int i;
    printf("[-] Vector elements: \n");
    for (int i = 0; i < matSizeX; i++) {
        for (int j = 0; j < matSizeY; j++)
            printf("%d	", v[i * matSizeY + j]);
        printf("\n");

    }
    printf("\b\b  \n");
}
