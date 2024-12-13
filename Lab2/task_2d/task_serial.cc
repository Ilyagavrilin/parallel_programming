#include "../utils.hpp"
#include <iostream>
#include <cmath>
#include <vector>

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main() {
    std::vector<std::vector<double>> a(ISIZE, std::vector<double>(JSIZE));

    double init_time = measureTime([&]() {
        for (int i = 0; i < ISIZE; i++) {
            for (int j = 0; j < JSIZE; j++) {
                a[i][j] = 10 * i + j;
            }
        }
    });

    double compute_time = measureTime([&]() {
        for (int i = 8; i < ISIZE; i++) {
            for (int j = 0; j < JSIZE - 3; j++) {
                a[i][j] = sin(4 * a[i - 8][j + 3]);
            }
        }
    });
    std::cout << "Computation time: " << compute_time << " seconds" << std::endl;


    saveArrayToFile("output_serial.txt", a);

    return 0;
}