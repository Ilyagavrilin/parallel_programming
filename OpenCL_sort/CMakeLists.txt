cmake_minimum_required(VERSION 3.10)
project(OpenCLBitonicSort)

# Найти OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Установка C++ стандарта
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Добавление исполняемого файла
add_executable(OpenCLBitonicSort main.cc)
target_link_libraries(OpenCLBitonicSort ${OpenCL_LIBRARIES})