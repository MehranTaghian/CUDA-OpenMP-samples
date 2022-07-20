# Parallel Programming on CPU and GPU
This repository contains some sample code that are useful in learning how to program with OpenMP and CUDA. Here I'm 
going to go through each directory and explain the purpose of each file. 

## OpenMP
In the OpenMP section, there is a sample code in `parallel_for_loop.cpp` which, as the name suggests, it's a simple 
for-loop parallelization.

In the `matrix_add.cpp` code, we have three 2D matrices A, B, and C where we want to calculate C = A + B. We do this in 
two ways: i) row-wise parallelization using a single parallel for-loop or ii) parallelize nested for-loops using the 
`collapse()` argument. The elements of A and B matrices are initialized randomly between 0 and 100. The 
`schedule({static, dynamic}, chunk_size)`
specifies how the thread allocation should be. If the schedule is `static`, it statically allocates a `chunk_size` number of
successive iterations to a single thread. In the case of `dynamic` allocation, each thread dynamically selects a number of chunks 
of iterations and runs them.

In the `Mandelbrot` section, we have the parallelization of the Mandelbrot set generation, which generates nice images.
The `serial.cpp` file contains the serial code for generating this set. Here we are trying to parallelize this code 
using OpenMP. The first step is to parallelize the main **for-loop** in the `For.cpp`. The second technique is 
**Multi-Tasking** which generates multiple tasks to do the process. 
One major problem that reduces the speed of the process is writing to file. We can 
use the `Pipeline` technique (as we did in `Pipeline.cpp`) to parallelize both: i) set generation and ii) writing to file.



## CUDA
