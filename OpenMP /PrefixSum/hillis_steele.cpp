#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <math.h>

#define NUM_THREADS 4

using namespace std;

void omp_check();
void fill_array(int* a, size_t n);
void prefix_sum(int* a, int index, size_t n, size_t block_size, int** flag, int** last_elements);
void print_array(int* a, size_t n);
void prefix_sum_parallel(int* a, size_t n, size_t block_size);

int main(int argc, char* argv[]) {
    // Check for correct compilation settings
    omp_check();
    // Input N
    size_t n = 256000000;
    //size_t n = 100;

    omp_set_num_threads(NUM_THREADS);

    size_t block_size = n / NUM_THREADS;


    //printf("[-] Please enter N: ");
    //scanf("%uld\n", &n);

    // Allocate memory for array
    int* a = (int*)malloc(n * sizeof a);
    // Fill array with numbers 1..n
    fill_array(a, n);

    // Print array
    //print_array(a, n);

    // Compute prefix sum
    prefix_sum_parallel(a, n, block_size);

    // Print array
    //print_array(a, n);

    // Free allocated memory
    free(a);
    return EXIT_SUCCESS;
}

void prefix_sum_parallel(int* a, size_t n, size_t block_size) {

    int num_levels = ceil(log2(NUM_THREADS));

    int** flag = (int**)malloc(NUM_THREADS * sizeof(int*));

    int** last_elements = (int**)malloc(NUM_THREADS * sizeof(int*));

    for (int i = 0; i < NUM_THREADS; i++) {
        flag[i] = (int*)malloc(num_levels * sizeof(int));
        last_elements[i] = (int*)malloc(num_levels * sizeof(int));
        for (int j = 0; j < num_levels; j++)
            flag[i][j] = 0;
    }

    double start = omp_get_wtime();

#pragma omp parallel
    {

#pragma omp for
        for (int i = 0; i < n; i += block_size) {
            prefix_sum(a, i, n, block_size, flag, last_elements);
        }

    }

    double elapsed = omp_get_wtime() - start;

    printf("Elapsed time is %f\n", elapsed);

}

void prefix_sum(int* a, int index, size_t n, size_t block_size, int** flag, int** last_elements) {

    int block_num = index / block_size;
    int num_levels = ceil(log2(NUM_THREADS));


    for (int i = index + 1; i < index + block_size; ++i) {
        a[i] = a[i] + a[i - 1];
    }

    int index_block = block_num - 1;

    int level = 1;
    int level_num = 0;

    //update last element of block before setting the flag and flushing
    last_elements[block_num][0] = a[index + block_size - 1];

#pragma omp flush
    flag[block_num][0] = 1;
#pragma omp flush(flag)


    if (block_num > 0) {
        while (index_block >= 0) {

            int level_num = log2(level);
            while (1) {
#pragma omp flush(flag)
                //printf("block: %d, index block: %d, level: %d\n", block_num, index_block, level_num);
                int flg_tmp = flag[index_block][level_num];
                if (flg_tmp) break;
            }

            int last_element_current_block_value = last_elements[index_block][level_num];
            a[index + block_size - 1] += last_element_current_block_value;

            //printf("block: %d, index block: %d, last element: %d, level num: %d\n", block_num, index_block, last_element_current_block_value, level_num);

            last_elements[block_num][level_num + 1] = a[index + block_size - 1];

#pragma omp flush
            flag[block_num][level_num + 1] = 1;
#pragma omp flush(flag)


            for (int j = index; j < index + block_size - 1; j++) {
                a[j] = a[j] + last_element_current_block_value;
            }


            index_block -= level;
            level *= 2;
        }
        level_num++;
    }

    while (level_num < num_levels) {
        level_num++;
        last_elements[block_num][level_num] = a[index + block_size - 1];
#pragma omp flush
        flag[block_num][level_num] = 1;
#pragma omp flush(flag)
    }


}

void print_array(int* a, size_t n) {
    int i;
    printf("[-] array: ");
    for (i = 0; i < n; ++i) {
        printf("%d, ", a[i]);
    }
    printf("\b\b  \n");
}

void fill_array(int* a, size_t n) {
    int i;
    for (i = 0; i < n; ++i) {
        a[i] = i + 1;
    }
}

void omp_check() {
    printf("------------ Info -------------\n");
#ifdef _DEBUG
    printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
    printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
    printf("[-] Platform: x64\n");
#elif _M_IX86
    printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86
#ifdef _OPENMP
    printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
    printf("[!] OpenMP is off.\n");
    printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
    printf("[-] Maximum threads: %d\n", omp_get_max_threads());
    printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
    printf("===============================\n");
}
