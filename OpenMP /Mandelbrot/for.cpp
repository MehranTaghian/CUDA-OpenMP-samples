#include <fstream>
#include <iostream>
#include <omp.h>
#include <math.h>
#include <sstream>

using namespace std;

int findMandelbrot(double cr, double ci, int max_iterations)
{
	int i = 0;
	double zr = 0.0, zi = 0.0;
	while (i < max_iterations && zr * zr + zi * zi < 4.0)
	{
		double temp = zr * zr - zi * zi + cr;
		zi = 2.0 * zr * zi + ci;
		zr = temp;
		i++;
	}

	return i;
}

double mapToReal(int x, int imageWidth, double minR, double maxR)
{
	double range = maxR - minR;
	return x * (range / imageWidth) + minR;
}

double mapToImaginary(int y, int imageHeight, double minI, double maxI)
{
	double range = maxI - minI;
	return y * (range / imageHeight) + minI;
}

int main()
{
	const int imageWidth = 4000, imageHeight = 4000, maxN = 1024;
	double minR = -1.5, maxR = 0.7, minI = -1.0, maxI = 1.0;
	double elapsedtime, starttime;

#ifndef _OPENMP
	printf("OpenMP is not supported, sorry!\n");
	getchar();
	return 0;
#endif

	string output[imageHeight];
	// Open the output file, write the PPM header...
	ofstream fout("output_image_For.ppm");
	fout << "P3" << endl; // "Magic Number" - PPM file
	fout << imageWidth << " " << imageHeight << endl; // Dimensions
	fout << "255" << endl; // Maximum value of a pixel R,G,B value...

	//omp_set_nested(1);
	omp_set_num_threads(8);

	starttime = omp_get_wtime();
	// For every pixel...
#pragma omp parallel for 
	for (int y = 0; y < imageHeight; y++) // Rows...
	{
		ostringstream sout;
		//#pragma omp parallel for //ordered
		for (int x = 0; x < imageWidth; x++) // Pixels in row (columns)...
		{
			// ... Find the real and imaginary values for c, corresponding to that
			//     x, y pixel in the image.
			double cr = mapToReal(x, imageWidth, minR, maxR);
			double ci = mapToImaginary(y, imageHeight, minI, maxI);

			// ... Find the number of iterations in the Mandelbrot formula
			//     using said c.
			int n = findMandelbrot(cr, ci, maxN);

			// ... Map the resulting number to an RGP value
			int r = ((int)(n * sinf(n)) % 256);
			int g = ((n * 3) % 256);
			int b = (n % 256);

			sout << r << " " << g << " " << b << " ";
		}
		sout << endl;
		output[y] = sout.str();
	}

	for (int i = 0; i < imageHeight; i++) {
		fout << output[i];
	}

	elapsedtime = omp_get_wtime() - starttime;
	// report elapsed time
	printf("Time Elapsed %f\n", elapsedtime);
	fout.close();

	cout << "Finished!" << endl;
	return 0;
}