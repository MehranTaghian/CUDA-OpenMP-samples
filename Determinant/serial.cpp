#include<iostream>
#include<time.h>
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
#include<iterator>


using namespace std;

void printResult(int n, float *lower, float *upper);


void LUdecomposition(double *u, int n) {

    for (int i = 0; i < n; i++) {
        double pivot = u[i * n + i];
        for (int j = i + 1; j < n; j++) { //row-wise
            double coef = u[j * n + i] / pivot;
            for (int k = i; k < n; k++) { //column-wise in a row other than pivot
                u[j * n + k] -= u[i * n + k] * coef;
            }
        }
    }

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

wstring s2ws(const std::string &s) {
    int len;
    int slength = (int) s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t *buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

string ws2s(const wstring &ws) {
    std::setlocale(LC_ALL, "");
    const std::locale locale("");
    typedef std::codecvt<wchar_t, char, std::mbstate_t> converter_type;
    const converter_type &converter = std::use_facet<converter_type>(locale);
    std::vector<char> to(ws.length() * converter.max_length());
    std::mbstate_t state;
    const wchar_t *from_next;
    char *to_next;
    const converter_type::result result = converter.out(state, ws.data(), ws.data() + ws.length(), from_next, &to[0],
                                                        &to[0] + to.size(), to_next);
    if (result == converter_type::ok or result == converter_type::noconv) {
        const std::string s(&to[0], to_next);
        return s;
    }
    return "";
}


vector <string> readDirectory(const std::string &name) {
    vector <string> files;
    string pattern(name);
    pattern.append("\\*.*");
    wstring stemp = s2ws(pattern);
    LPCWSTR result = stemp.c_str();

    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(result, &data)) != INVALID_HANDLE_VALUE) {
        do {
            if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                string name = ws2s(data.cFileName);
                files.push_back(name);
            }
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
    return files;
}


void readRow(vector<double> &row, string &line, string &word) {
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
    double start = omp_get_wtime();
    for (int i = 0; i < files.size(); i++) {
        string f = files[i];
        //printf("Hello from %d, file: %s\n", omp_get_thread_num(), f);
        // File pointer
        fstream fin;
        string filename = directory + "\\data_in\\" + f;

        //cout << directory + "\\data_in\\" + f << endl;
        // Open an existing file
        fin.open(filename, ios::in);

        // Read the Data from the file
        // as float Vector
        vector<double> row;
        //vector<float> results;
        string line, word;
        //file pointer
        fstream fout;
        ostringstream out;
        //opens an existing csv file or creates a new file.
        fout.open(directory + "\\data_out\\" + "outputs_" + f, ios::out); // this will write to new file

        getline(fin, line);
        readRow(row, line, word);
        int n = row.size();

        double *upper;
        upper = (double *) malloc(n * n * sizeof(double));

        memcpy(&upper[0], &row[0], n * sizeof(double));

        int j = 1;
        while (getline(fin, line)) {
            readRow(row, line, word);
            memcpy(&upper[j * n], &row[0], n * sizeof(double));
            readRow(row, line, word);
            j++;
        }

        LUdecomposition(upper, n);

        double result = 1.0f;
        for (int i = 0; i < n; i++) {
            result *= upper[i * n + i];
        }

        fout << result;
        fout.close();
    }
    double elapsed = omp_get_wtime() - start;
    printf("Elapsed time is: %f\n", elapsed);
}

int main() {

    string directory;
    cout << "Enter the directory of the dataset:" << endl;
    cout << "Attention: The directory should contain data_in and data_out directories" << endl;

    getline(cin, directory);

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




