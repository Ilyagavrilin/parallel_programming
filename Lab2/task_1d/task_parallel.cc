#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "../utils.hpp" // Utility header for time measurement and file saving

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main(int argc, char** argv) {
    // Initialize the 2D array
    std::vector<std::vector<double>> a(ISIZE, std::vector<double>(JSIZE, 0.0));

    // Measure time for the initialization
    double init_time = measureTime([&]() {
        for (size_t i = 0; i < ISIZE; ++i) {
            for (size_t j = 0; j < JSIZE; ++j) {
                a[i][j] = 10.0 * i + j;
            }
        }
    });
    //std::cerr << "Initialization time: " << init_time << " seconds" << std::endl;

    // Measure time for the computation loop
    double computation_time = measureTime([&]() {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t i = 0; i < ISIZE - 1; ++i) {
            for (size_t j = 6; j < JSIZE; ++j) {
                a[i][j] = std::sin(0.2 * a[i + 1][j - 6]);
            }
        }
    });
    std::cout << "Computation time: " << computation_time << " seconds" << std::endl;

    saveArrayToFile("result_parallel.txt", a);

    return 0;
}