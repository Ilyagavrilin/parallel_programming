#include "../utils.hpp"
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iostream>

#ifndef ISIZE
#define ISIZE 1000
#endif
#ifndef JSIZE
#define JSIZE 1000
#endif

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<double>> a(ISIZE, std::vector<double>(JSIZE, 0));

    int diff_J = JSIZE / commsize;
    int start_J = diff_J * rank;
    int end_J = diff_J * (rank + 1);

    // Если JSIZE не делится на число процессов
    if (JSIZE % commsize) {
        if (rank < JSIZE % commsize) {
            start_J += rank;
            end_J += rank + 1;
        } else {
            start_J += JSIZE % commsize;
            end_J += JSIZE % commsize;
        }
    }

    if (rank == commsize - 1) {
        end_J = JSIZE - 3;
    }

    for (int i = 0; i < ISIZE; i++) {
        for (int j = 0; j < JSIZE; j++) {
            a[i][j] = 10 * i + j;
        }
    }

    int localArraySize = end_J - start_J;
    std::vector<double> localArray(localArraySize);
    std::vector<int> recvcnts(commsize, 0);
    std::vector<int> displs(commsize, 0);

    for (int i = 0; i < commsize; i++) {
        int local_start = diff_J * i;
        int local_end = diff_J * (i + 1);

        if (JSIZE % commsize) {
            if (i < JSIZE % commsize) {
                local_start += i;
                local_end += i + 1;
            } else {
                local_start += JSIZE % commsize;
                local_end += JSIZE % commsize;
            }
        }

        if (i == commsize - 1) {
            local_end = JSIZE - 3;
        }

        recvcnts[i] = local_end - local_start;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcnts[i - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 8; i < ISIZE; i++) {
        for (int j = start_J; j < end_J; j++) {
            localArray[j - start_J] = sin(4 * a[i - 8][j + 3]);
        }

        MPI_Allgatherv(localArray.data(), localArraySize, MPI_DOUBLE, a[i].data(),
                       recvcnts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double computation_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        std::cout << "Computation time: " << computation_time << " seconds" << std::endl;
    }

    if (rank == 0) {
        saveArrayToFile("output_parallel.txt", a);
    }

    MPI_Finalize();
    return 0;
}
