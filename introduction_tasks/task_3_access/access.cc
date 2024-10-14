#include <iostream>
#include <omp.h>

#ifdef ENABLE_TIMING
#include <chrono>
#endif

int main() {
    unsigned long shared_var = 0;
    int num_threads = 0;

#ifdef ENABLE_TIMING
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    #pragma omp single
    {
        start_time = std::chrono::high_resolution_clock::now();
    }
#endif

    #pragma omp parallel shared(shared_var, num_threads)
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            std::cout << "Number of threads: " << num_threads << std::endl;
        }

        int thread_num = omp_get_thread_num();

        for (int i = 0; i < num_threads; ++i) {
            #pragma omp barrier
            if (i == thread_num) {
                #pragma omp critical
                {
                    shared_var += thread_num;
                    std::cout << "Thread " << thread_num << " bumped variable. Current value: " << shared_var << std::endl;
                }
            }
        }
    }
#ifdef ENABLE_TIMING
    #pragma omp single
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::cout << "Execution time: " << elapsed_seconds.count() << " seconds." << std::endl;
    }
#endif

    return 0;
}