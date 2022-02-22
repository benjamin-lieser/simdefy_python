cmake_minimum_required(VERSION 3.10)
project(simdefy_python)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -march=native -Wall -Wextra -Wfloat-conversion")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -march=native -Wall -Wextra -Wfloat-conversion -fsanitize=address -fsanitize=undefined")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ..)

add_subdirectory(pybind11)

pybind11_add_module(simdefy pythonBinding.cpp simd.h functions.h)