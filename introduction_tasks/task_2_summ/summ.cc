#include <iostream>
#include <omp.h>
#include <cstdlib>
#ifdef ENABLE_TIMING
#include <chrono>
#endif

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Invalid number of arguments, proceed only number to count."<< std::endl;
        return 1;
    }

    long N = std::atol(argv[1]);
    if (N <= 0) {
        std::cerr << "Count number should be more than zero" << std::endl;
        return 1;
    }

    double sum = 0.0;

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    #ifdef ENABLE_TIMING
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 1; i <= N; ++i) {
        double term = 1.0 / i;
        sum += term;
    }

    #ifdef ENABLE_TIMING
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    #endif

    #pragma omp master
    {
        std::cout << "Summ S = " << sum << std::endl;
        #ifdef ENABLE_TIMING
        std::cout << "Exectuion time: " << elapsed_seconds.count() << " seconds." << std::endl;
        #endif
    }

    return 0;
}