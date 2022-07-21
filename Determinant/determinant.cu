#include<iostream>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<omp.h>
#include<vector>
#include<string>
#include<Windows.h>
#include <clocale>
#include <locale>
#include<fstream>
#include<sstream>
#include<math.h>


using namespace std;

void printResult(int n, float *lower, float *upper);

void printMatrix(float *A, int n);


__global__ void determinant(double *u, int n) {
    /*
        The kernel which gets u (input matrix which would be row reduced to Upper in LU).
        For each pivot, we can do row reduction opertion in parallel for different rows. The
        __syncthreds() at the end of the first for-loop is because that pivots should be calculated
        sequentially, meaning that the calculation of each pivot requires that the previous one has already
        been calculated.
    */

    int step = blockDim.x;

    for (int i = 0; i < n; i++) {
        double pivot = u[i * n + i];
        for (int j = threadIdx.x; j >= (i + 1) && j < n; j += step) { //row-wise
            double coef = u[j * n + i] / pivot;
            for (int k = i; k < n; k++) { //column-wise in a row other than pivot
                u[j * n + k] -= u[i * n + k] * coef;
            }
        }
        __syncthreads();
    }
}

cudaError_t LUdecomposition(double *u, double *d_u, int n, cudaError_t &cudaStatus, cudaStream_t &stream) {
    /*
        Helper function which copies the data to device and runs the kernel
    */


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync(d_u, u, n * n * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int blockSize = n > 1024 ? 1024 : n;

    determinant << < 1, blockSize, 0, stream >> > (d_u, n);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync(u, d_u, n * n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel excecution failed!");
        goto Error;
    }

    Error:
    return cudaStatus;
}

float rand_FloatRange(float a, float b) {
    return ((b - a) * ((float) rand() / RAND_MAX)) + a;
}

void initialize(float *A, int n) {
    srand(time(0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i * n + j] = rand_FloatRange(0, 10);
        }
}

void printMatrix(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f, ", A[i * n + j]);
        }
        printf("\n");
    }
}

vector <string> readDirectory(const std::string &name) {
    /*
        Find the files inside the specified directory
    */

    vector <string> files;
    string pattern(name);
    pattern.append("\\*.*");

    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                string name = data.cFileName;
                files.push_back(name);
            }
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
    return files;
}

void readRow(vector<double> &row, string &line, string &word) {
    /*
        Read each row of the file given and returns a vector of double
        holding the values of the rows of the matrix.
    */

    row.clear();

    // read an entire row and
    // store it in a string variable 'line'

    // used for breaking words
    stringstream s(line);
    // read every column data of a row and
    // store it in a string variable, 'word'
    while (getline(s, word, ' ')) {
        // add all the column data
        // of a row to a vector
        row.push_back(stof(word));
    }
}

void readFile(string &directory, vector <string> &files) {
    /*
        This function is given the list of files inside the directory, then it creates
        threads in OMP. Each threads creates streams and allocates memory (MAXIMUM 4096 double)
        and after reading from files, it copies data to device and runs the helper function which runs the
        kernel at last.
    */

    int MAX_MATRIX_SIZE = 4096;
    int max_size = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;

    double start = omp_get_wtime();

    //u in LU decomposition. We need to only row reduce A to u to calculate determinant.
    double *upper[8];

    cudaError_t cudaStatus[8];
    double *d_u[8];

    cudaStream_t stream[8];

    // Choose which GPU to run on, change this on a multi-GPU system.
    //cudaSetDevice(0);

    int num_threads = files.size() > 8 ? 8 : files.size();


#pragma omp parallel num_threads(num_threads)
    {
        //printf("%d\n", omp_get_num_threads());

        int id = omp_get_thread_num();

        cudaStreamCreate(&stream[id]);

        cudaStatus[id] = cudaMalloc((void **) &d_u[id], max_size * sizeof(double));
        if (cudaStatus[id] != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            //goto Error;
        }
        cudaMallocHost((void **) &upper[id], max_size * sizeof(double));


#pragma omp for schedule(static, 1)
        for (int i = 0; i < files.size(); i++) {

            int id = omp_get_thread_num();

            string f = files[i];
            //printf("Hello from %d, file: %s\n", omp_get_thread_num(), f);
            // File pointer

            fstream fin;
            string filename = directory + "\\data_in\\" + f;
            // Open an existing file
            fin.open(filename, ios::in);

            // Read the Data from the file
            // as float Vector
            vector<double> row;
            //vector<float> results;
            string line, word;

            //file pointer
            fstream fout;
            //opens an existing csv file or creates a new file.
            fout.open(directory + "\\data_out\\" + "outputs_" + f, ios::out); // this will write to new file

            int n, j = 0;
            while (getline(fin, line)) {
                readRow(row, line, word);
                n = row.size();
                memcpy(&upper[id][j * n], &row[0], n * sizeof(double));
                j++;
            }

            //printMatrix(upper[id], n);
            //printf("\n");

            cudaStatus[id] = LUdecomposition(upper[id], d_u[id], n, cudaStatus[id], stream[id]);

            if (cudaStatus[id] != cudaSuccess) {
                fprintf(stderr, "LUdecomposition failed!");
                //goto Error;
            }
            double result = 1.0f;
            for (int i = 0; i < n; i++) {
                result *= upper[id][i * n + i];
            }

            fout << result;
            fout.close();
        }

        cudaFreeHost(upper[id]);
        cudaFree(d_u[id]);
    }

    double elapsed = omp_get_wtime() - start;
    printf("Elapsed time is: %f\n", elapsed);
}

int main() {

#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif

    string directory;
    cout << "Enter the directory of the dataset:" << endl;
    cout << "Attention: The directory should contain data_in and data_out directories" << endl;

    getline(cin, directory);

    cout << endl;
    cout << "The provided directory is:\n";
    cout << directory << endl;

    vector <string> files = readDirectory(directory + "\\data_in");

    readFile(directory, files);

    system("PAUSE");

    return 0;
}

void printResult(int n, float *lower, float *upper) {
    cout << "\nL Decomposition is as follows...\n" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << lower[i * n + j] << " ";
        }
        cout << endl;
    }
    cout << "\nU Decomposition is as follows...\n" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << upper[i * n + j] << " ";
        }
        cout << endl;
    }
}



