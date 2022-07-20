#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void constantInit(float *data, int size, float val)
{
    static float L = 0.0f;

    for (int i = 0; i < size; ++i)
    {
        data[i] = ++L;
    }
}

// Prints a Matrices to the stdout.
void printMat(float * v, int matSizeX, int matSizeY) {
    int i;
    printf("[-] Vector elements: \n");
    for (int i = 0; i < matSizeX; i++) {
        for (int j = 0; j < matSizeY; j++)
            printf("%f", v[i*matSizeY + j]);
        printf("\n");

    }
    printf("\b\b  \n");
}

void multiply(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float C_element = 0;
            for (int k = 0; k < n; k++) {
                C_element += A[i*n + k] * B[k * n + j];
            }
            C[i* n + j] = C_element;
        }
}

void matrixMalt(int n) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = n * n;
    unsigned int mem_size_A = sizeof(float)* size_A;
    float *A = (float *)malloc(mem_size_A);
    unsigned int size_B = n * n;
    unsigned int mem_size_B = sizeof(float)* size_B;
    float *B = (float *)malloc(mem_size_B);
    unsigned int size_C = n * n;
    unsigned int mem_size_C = sizeof(float)* size_C;
    float *C = (float *)malloc(mem_size_C);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(A, size_A, 1.0f);
    constantInit(B, size_B, valB);

    double start = omp_get_wtime();

    multiply(A, B, C, n);

    double elapsed = omp_get_wtime() - start;

    printf("Elapsed time: %f\n", elapsed);

    //printMat(A, n, n);
    //printMat(B, n, n);
    //printMat(C, n, n);

}

int main() {
    int n = 0;
    printf("Enter N:");
    scanf("%u", &n);

    matrixMalt(n);
}