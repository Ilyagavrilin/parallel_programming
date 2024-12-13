#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h> // Include OpenMP header
#include "../utils.hpp"

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main(int argc, char **argv) {
    // Initialize a 2D array using std::vector
    std::vector<std::vector<double>> a(ISIZE, std::vector<double>(JSIZE, 0.0));

    // Fill the array with initial values
    for (int i = 0; i < ISIZE; ++i) {
        for (int j = 0; j < JSIZE; ++j) {
            a[i][j] = 10 * i + j;
        }
    }

    // Measure the time taken by the computation loop
    double computation_time = measureTime([&]() {
        // Parallelize the main computation loop
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < ISIZE; ++i) {
            for (int j = 0; j < JSIZE; ++j) {
                a[i][j] = std::sin(2 * a[i][j]);
            }
        }
    });
    
    // Print elapsed time
    std::cout << "Computation time: " << computation_time << " seconds" << std::endl;

    saveArrayToFile("result_parallel.txt", a);

    return 0;
}