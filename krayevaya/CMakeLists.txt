# Минимальная версия CMake
cmake_minimum_required(VERSION 3.16)

# Название проекта
project(OpenCL_BVP_Solver
        VERSION 1.0
        DESCRIPTION "Solver for boundary value problems using OpenCL"
        LANGUAGES CXX)

# Установить стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Включить строгие предупреждения
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /WX)
endif()

# Путь к исходным файлам
set(SRC_FILES main.cc)

# Добавить главную цель (main executable)
add_executable(OpenCL_BVP_Solver ${SRC_FILES})

# Логгирование пути к OpenCL
message(STATUS "Finding OpenCL...")

# Найти библиотеку OpenCL
find_package(OpenCL REQUIRED)

# Подключить OpenCL к проекту
target_include_directories(OpenCL_BVP_Solver PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(OpenCL_BVP_Solver PRIVATE ${OpenCL_LIBRARIES})

# Добавить define для платформы
target_compile_definitions(OpenCL_BVP_Solver PRIVATE CL_TARGET_OPENCL_VERSION=120)

# Установить пути для исполняемого файла
set_target_properties(OpenCL_BVP_Solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Тестирование (Google Test)
option(ENABLE_TESTING "Enable testing" OFF)

if (ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# Сообщение об успешной конфигурации
message(STATUS "Configuration successful! Use 'make' to build the project.")