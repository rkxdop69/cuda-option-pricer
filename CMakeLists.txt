cmake_minimum_required(VERSION 3.18)
project(OptionPricer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA Architecture to sm_89
set(CMAKE_CUDA_ARCHITECTURES 89)

# Source files
set(SOURCES
    main.cpp
    CPUOptionEngine.cpp
    GPUOptionEngine.cu
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)

# Link OpenMP and cuRAND
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE curand)
