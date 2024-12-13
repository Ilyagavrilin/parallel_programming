#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include "../utils.hpp"

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    // Determine the portion of rows each process will handle
    int rows_per_process = ISIZE / commsize;
    int extra_rows = ISIZE % commsize;

    int start_row = rank * rows_per_process + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate a local portion of the array
    std::vector<std::vector<double>> local_a(end_row - start_row, std::vector<double>(JSIZE, 0.0));

    // Initialize the local portion of the array
    for (int i = 0; i < local_a.size(); ++i) {
        for (int j = 0; j < JSIZE; ++j) {
            local_a[i][j] = 10 * (start_row + i) + j;
        }
    }

    // Measure the time taken by the computation loop
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    double start_time = MPI_Wtime();

    for (int i = 0; i < local_a.size(); ++i) {
        for (int j = 0; j < JSIZE; ++j) {
            local_a[i][j] = std::sin(2 * local_a[i][j]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize after computation
    double computation_time = MPI_Wtime() - start_time;

    // Gather the results on the root process
    std::vector<std::vector<double>> a;
    if (rank == 0) {
        a.resize(ISIZE, std::vector<double>(JSIZE, 0.0));
    }

    // Flatten the local portion for MPI_Gatherv
    std::vector<double> flat_local_a;
    for (const auto &row : local_a) {
        flat_local_a.insert(flat_local_a.end(), row.begin(), row.end());
    }

    // Gather counts and displacements
    std::vector<int> counts(commsize);
    std::vector<int> displacements(commsize);

    int offset = 0;
    for (int i = 0; i < commsize; ++i) {
        int rows = rows_per_process + (i < extra_rows ? 1 : 0);
        counts[i] = rows * JSIZE;
        displacements[i] = offset;
        offset += counts[i];
    }

    // Flatten the full array on the root process
    std::vector<double> flat_a(rank == 0 ? ISIZE * JSIZE : 0);

    MPI_Gatherv(flat_local_a.data(), flat_local_a.size(), MPI_DOUBLE,
                flat_a.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Convert flat array to 2D array on root process
    if (rank == 0) {
        for (int i = 0; i < ISIZE; ++i) {
            std::copy(flat_a.begin() + i * JSIZE, flat_a.begin() + (i + 1) * JSIZE, a[i].begin());
        }

        // Save the result and print the computation time
        saveArrayToFile("result_parallel_mpi.txt", a);
        std::cout << "Computation time: " << computation_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}