cmake_minimum_required(VERSION 3.16)
project(AccessOMP LANGUAGES CXX)

option(ENABLE_SANITIZERS "Turn on sanitizers" OFF)
option(ENABLE_TIMING "Turn on time measurement" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


# Флаги компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(ENABLE_WARNINGS)
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Включение санитайзеров
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        add_compile_options(-fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    else()
        message(WARNING "Sunitizers are not supported on your current compiler")
    endif()
endif()

find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} access.cc)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)



if(ENABLE_TIMING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TIMING)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(ENABLE_SANITIZERS)
    message(STATUS "Санитайзеры включены")
endif()
