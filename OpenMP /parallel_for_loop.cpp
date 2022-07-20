#include <stdio.h>
#include <math.h>
#include <omp.h>

#define num_threads 8

//Choose a very big number to iterate through
const long int verybig = 50000;

// ***********************************************************************
int main(void) {
    int i;
    long int j, sum[num_threads];
    double sumx[num_threads], sumy[num_threads], total[num_threads];
    double starttime, elapsedtime;
    // -----------------------------------------------------------------------
    // output a start message
    printf("serial timings for %ld iterations\n\n", verybig);
    // repeat experiment several times
    for (i = 0; i < 10; i++) {
        // get starting time56 x chapter 3 parallel studio xe for the impatient
        starttime = omp_get_wtime();
        // reset check sum & running total
        //int num_threads;
        for (int l = 0; l < num_threads; l++) {
            sum[l] = 0;
            total[l] = 0.0;
        }
        // work loop, do some work by looping verybig times
#pragma omp parallel for private(sumx, sumy, k) // reduction(+: sum, total)
        /*
         * The statement private(sumx, sumy, k) makes variables sumx, sumy, and k private to each
         * thread of the parallel-for block.
         * Each thread would have a local copy of sum and total where after the parallel block, the value
         * of each would be summed. This is the use of reduction(+: sum, total) statements. This way, we would not
         * need to define sum[num_threads] or total[num_threads] because one instance of sum and total would be
         * generated per thread.
         */
        for (j = 0; j < verybig; j++) {

            int id = omp_get_thread_num();
            long k;
            // increment check sum
            sum[id] += 1;
            // calculate first arithmetic series
            sumx[id] = 0.0;
            for (k = 0; k < j; k++)
                sumx[id] = sumx[id] + (double) k;
            // calculate second arithmetic series
            sumy[id] = 0.0;
            for (k = j; k > 0; k--)
                sumy[id] = sumy[id] + (double) k;
            if (sumx[id] > 0.0)total[id] = total[id] + 1.0 / sqrt(sumx[id]);
            if (sumy[id] > 0.0)total[id] = total[id] + 1.0 / sqrt(sumy[id]);
        }
        double total_result = 0.0;
        long sum_result = 0;

        for (j = 0; j < num_threads; j++) {
            total_result += total[j];
            sum_result += sum[j];
        }

        // get ending time and use it to determine elapsed time
        elapsedtime = omp_get_wtime() - starttime;
        // report elapsed time
        printf("time elapsed: %f secs, total = %lf, check sum = %ld\n",
               elapsedtime, total_result, sum_result);
        printf("number of threads %d\n", num_threads);
    }
    // return integer as required by function header
    getchar();
    return 0;
}
