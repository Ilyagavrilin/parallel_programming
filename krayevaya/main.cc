#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>

// Константы задачи
constexpr float DOMAIN_START = -10.0f;
constexpr float DOMAIN_END = 10.0f;

// Вычисляемое во время компиляции значение sqrt(2)
constexpr float compute_sqrt(float value, float low = 0, float high = 2) {
    float mid = (low + high) / 2;
    return (high - low < 1e-6f) ? mid : (mid * mid > value ? compute_sqrt(value, low, mid) : compute_sqrt(value, mid, high));
}
constexpr float Y_BOUNDARY = compute_sqrt(2.0f);
constexpr float H_LIMIT = 1.0f / compute_sqrt(2.0f);

// Порог точности для валидации
constexpr float TOLERANCE = 0.01f;

// Макрос для обработки ошибок OpenCL
#define CHECK_CL_ERROR(err, msg)                                           \
    if (err != CL_SUCCESS) {                                              \
        throw std::runtime_error(std::string("Ошибка: ") + msg +          \
                                 " (код: " + std::to_string(err) + ")");  \
    }

/**
 * @brief Класс для автоматического управления ресурсами OpenCL
 */
class OpenCLResources {
public:
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;

    OpenCLResources(cl_device_id device, const char* kernel_source) {
        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err, "Не удалось создать контекст");

        command_queue = clCreateCommandQueue(context, device, 0, &err);
        CHECK_CL_ERROR(err, "Не удалось создать очередь команд");

        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
        CHECK_CL_ERROR(err, "Не удалось создать программу");

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("Ошибка компиляции программы: " + std::string(log.data()));
        }

        kernel = clCreateKernel(program, "solve_bvp", &err);
        CHECK_CL_ERROR(err, "Не удалось создать ядро");
    }

    ~OpenCLResources() {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }
};

/**
 * @brief Генерация значений параметра `a`.
 */
std::vector<float> generate_a_values(float a_start, float a_end, int n_a) {
    std::vector<float> a_values(n_a);
    for (int i = 0; i < n_a; ++i) {
        a_values[i] = a_start + i * (a_end - a_start) / (n_a - 1);
    }
    return a_values;
}

/**
 * @brief Проверка результатов.
 */
void validate_results(const std::vector<float>& results, const std::vector<float>& a_vals, float tol) {
    std::cout << "\nПроверка результатов:\n";
    for (size_t i = 0; i < a_vals.size(); ++i) {
        if (std::abs(results[i] - Y_BOUNDARY) > tol) {
            std::cerr << "a = " << a_vals[i] << ", y(10) = " << results[i]
                      << ": Ошибка превышает допустимый порог!" << std::endl;
        } else {
            std::cout << "a = " << a_vals[i] << ", y(10) = " << results[i]
                      << ": Проверка пройдена." << std::endl;
        }
    }
}

/**
 * @brief Основная функция для решения задачи.
 */
int main() {
    try {
        // Проверка шага
        int n_points = 4000;  // Количество точек
        float h = (DOMAIN_END - DOMAIN_START) / n_points;
        if (h >= H_LIMIT) {
            throw std::invalid_argument("Шаг превышает лимит. Увеличьте количество точек.");
        }

        // Генерация параметров `a`
        auto a_vals = generate_a_values(100.0f, 1000000.0f, 100);

        // Инициализация OpenCL
        cl_platform_id platform_id;
        cl_device_id device_id;
        cl_uint num_platforms, num_devices;
        cl_int err;

        err = clGetPlatformIDs(1, &platform_id, &num_platforms);
        CHECK_CL_ERROR(err, "Не удалось получить платформы");

        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
        CHECK_CL_ERROR(err, "Не удалось получить устройства");

        // Создание ресурсов OpenCL
        OpenCLResources cl_resources(device_id, R"(
            __kernel void solve_bvp(
                __global float* y_results, __global float* a_vals,
                const int n_points, const float h) {
                int idx = get_global_id(0);
                float a = a_vals[idx];
                float y_prev = sqrt(2.0f);
                float y_curr = sqrt(2.0f);
                float y_next;
                for (int i = 1; i < n_points - 1; ++i) {
                    float f_curr = a * (pow(y_curr, 3) - y_curr);
                    y_next = 2 * y_curr - y_prev - h * h * f_curr;
                    y_prev = y_curr;
                    y_curr = y_next;
                }
                y_results[idx] = y_curr;
            }
        )");

        // Буферы OpenCL
        cl_mem a_buffer = clCreateBuffer(cl_resources.context, CL_MEM_READ_ONLY,
                                         a_vals.size() * sizeof(float), nullptr, &err);
        CHECK_CL_ERROR(err, "Не удалось создать буфер для a_vals");

        cl_mem result_buffer = clCreateBuffer(cl_resources.context, CL_MEM_WRITE_ONLY,
                                              a_vals.size() * sizeof(float), nullptr, &err);
        CHECK_CL_ERROR(err, "Не удалось создать буфер для результатов");

        // Запись данных
        err = clEnqueueWriteBuffer(cl_resources.command_queue, a_buffer, CL_TRUE, 0,
                                   a_vals.size() * sizeof(float), a_vals.data(), 0, nullptr, nullptr);
        CHECK_CL_ERROR(err, "Не удалось записать данные в буфер a_vals");

        // Установка аргументов ядра
        err = clSetKernelArg(cl_resources.kernel, 0, sizeof(cl_mem), &result_buffer);
        CHECK_CL_ERROR(err, "Не удалось настроить аргумент 0");

        err = clSetKernelArg(cl_resources.kernel, 1, sizeof(cl_mem), &a_buffer);
        CHECK_CL_ERROR(err, "Не удалось настроить аргумент 1");

        err = clSetKernelArg(cl_resources.kernel, 2, sizeof(int), &n_points);
        CHECK_CL_ERROR(err, "Не удалось настроить аргумент 2");

        err = clSetKernelArg(cl_resources.kernel, 3, sizeof(float), &h);
        CHECK_CL_ERROR(err, "Не удалось настроить аргумент 3");

        // Запуск ядра
        size_t global_size = a_vals.size();
        err = clEnqueueNDRangeKernel(cl_resources.command_queue, cl_resources.kernel, 1, nullptr,
                                     &global_size, nullptr, 0, nullptr, nullptr);
        CHECK_CL_ERROR(err, "Не удалось выполнить ядро");

        // Считывание результатов
        std::vector<float> results(a_vals.size());
        err = clEnqueueReadBuffer(cl_resources.command_queue, result_buffer, CL_TRUE, 0,
                                  results.size() * sizeof(float), results.data(), 0, nullptr, nullptr);
        CHECK_CL_ERROR(err, "Не удалось прочитать результаты");

        // Проверка результатов
        validate_results(results, a_vals, TOLERANCE);

        // Очистка
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(result_buffer);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
