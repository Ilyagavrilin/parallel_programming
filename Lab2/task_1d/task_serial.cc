#include <iostream>
#include <cmath>
#include <vector>
#include "../utils.hpp" // Для measureTime и saveArrayToFile

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main() {
    std::vector<std::vector<double>> a(ISIZE, std::vector<double>(JSIZE));

    for (int i = 0; i < ISIZE; i++) {
        for (int j = 0; j < JSIZE; j++) {
            a[i][j] = 10 * i + j;
        }
    }

    double computation_time = measureTime([&]() {
        for (size_t i = 0; i < ISIZE - 1; ++i) {
            for (size_t j = 6; j < JSIZE; ++j) {
                a[i][j] = std::sin(0.2 * a[i + 1][j - 6]);
            }
        }
    });

    std::cout << "Computation time: " << computation_time << " seconds" << std::endl;

    saveArrayToFile("result_serial.txt", a);

    return 0;
}