#include <iostream>
#include <omp.h>

#ifdef ENABLE_TIMING
#include <chrono>
#endif

int main() {
    #ifdef ENABLE_TIMING
        auto start_time = std::chrono::high_resolution_clock::now();
    #endif

    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        #pragma omp critical
        {
            std::cout << "Hello World from thread " << thread_num
                      << " out of " << num_threads << " threads." << std::endl;
        }
    }

    #ifdef ENABLE_TIMING
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
    #endif

    return 0;
}