// bitonic_sort.cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <climits>
#include <cassert>

// Set DEBUG to 1 to enable debug output
#define DEBUG 0

// OpenCL kernel for Bitonic Sort
const char *bitonic_sort_kernel = R"CLC(
__kernel void bitonic_sort_step(__global int* data, const int j, const int k, const int ascending) {
    unsigned int i = get_global_id(0);
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if ((data[i] > data[ixj]) == ascending) {
                // Swap data[i] and data[ixj]
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if ((data[i] < data[ixj]) == ascending) {
                // Swap data[i] and data[ixj]
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}
)CLC";

// Function to find the next power of two
size_t nextPowerOfTwo(size_t n) {
    if (n == 0) return 1;
    if ((n & (n - 1)) == 0) return n;
    size_t power = 1;
    while (power < n) power <<= 1;
    return power;
}

// Function to pad the array to the next power of two
void padArray(std::vector<int> &data, size_t paddedSize) {
    data.resize(paddedSize, INT_MAX);
}

// Function to generate a random array
std::vector<int> generateRandomArray(size_t size) {
    std::vector<int> data(size);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(0, 10000);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// Debug function to print the array
void printArray(const std::vector<int> &data) {
#if DEBUG
    for (const auto &val : data) std::cout << val << " ";
    std::cout << std::endl;
#endif
}

// Function to check if the array is sorted correctly
bool isSorted(const std::vector<int> &data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i - 1] > data[i]) {
#if DEBUG
            std::cerr << "Array is not sorted at index " << i - 1 << ": "
                      << data[i - 1] << " > " << data[i] << std::endl;
#endif
            return false;
        }
    }
    return true;
}

// Bitonic Sort implementation
void bitonicSort(std::vector<int> &data) {
    size_t n = data.size();
    size_t paddedSize = nextPowerOfTwo(n);
    padArray(data, paddedSize);

    cl::Context context;
    cl::Program program;

    // Initialize OpenCL context and compile the kernel
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        assert(!platforms.empty());

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
        assert(!devices.empty());

        context = cl::Context(devices[0]);
        cl::Program::Sources sources;
        sources.push_back({bitonic_sort_kernel, strlen(bitonic_sort_kernel)});

        program = cl::Program(context, sources);
        program.build({devices[0]});
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit(1);
    }

    cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0]);
    cl::Buffer buffer_data(context, CL_MEM_READ_WRITE, sizeof(int) * paddedSize);

    queue.enqueueWriteBuffer(buffer_data, CL_TRUE, 0, sizeof(int) * paddedSize, data.data());

    cl::Kernel kernel(program, "bitonic_sort_step");

    int ascending = 1; // 1 for ascending order
    for (unsigned int k = 2; k <= paddedSize; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            kernel.setArg(0, buffer_data);
            kernel.setArg(1, (int)j);
            kernel.setArg(2, (int)k);
            kernel.setArg(3, ascending);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(paddedSize), cl::NullRange);
            queue.finish();
        }
    }

    queue.enqueueReadBuffer(buffer_data, CL_TRUE, 0, sizeof(int) * paddedSize, data.data());

    // Remove padding
    data.resize(n);
}

int main(int argc, char *argv[]) {
    size_t array_size = 10; // Default size

    if (argc >= 2) {
        array_size = std::stoul(argv[1]);
    } else {
        std::cout << "No array size provided. Using default size: 10" << std::endl;
    }

    // Generate a random array
    std::vector<int> data = generateRandomArray(array_size);

#if DEBUG
    std::cout << "Original data: ";
    printArray(data);
#endif

    // Perform Bitonic Sort
    auto start = std::chrono::high_resolution_clock::now();
    bitonicSort(data);
    auto end = std::chrono::high_resolution_clock::now();

    // Check if the array is sorted
    bool sorted = isSorted(data);

    // Output results
    std::cout << "Sorting completed, correctness: " << (sorted ? "yes" : "no") << std::endl;
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

#if DEBUG
    std::cout << "Sorted data: ";
    printArray(data);
#endif

    return 0;
}