#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

// Detect MPI using the presence of MPI_VERSION
#ifdef MPI_VERSION
#include <mpi.h>
#define MPI_ENABLED
#endif

// Save a 2D array to a file
template <typename T>
void saveArrayToFile(const std::string& filename, const std::vector<std::vector<T>>& array) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    for (const auto& row : array) {
        for (const auto& element : row) {
            file << std::fixed << std::setprecision(6) << element << " ";
        }
        file << "\n";
    }

    file.close();
}

// Measure time and execute a function
template <typename Func>
double measureTime(Func func) {
    double elapsed_time;

#ifdef MPI_ENABLED
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes are synchronized
    double start_time = MPI_Wtime();
    func();
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize again after execution
    double end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
#else
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
#endif

    return elapsed_time;
}

#endif // UTILS_HPP