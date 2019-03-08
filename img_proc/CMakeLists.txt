cmake_minimum_required(VERSION 3.5)
project(adadoc)

set(PYBIND11_CPP_STANDARD -std=c++11)

add_subdirectory(pybind11)
#pybind11_add_module(adadoc adadoc.cpp adath.cpp ppht.cpp)
link_directories(/home/joy/anaconda3/lib)

include_directories(/home/joy/anaconda3/include)
add_library(adadoc MODULE adadoc.cpp adath.cpp ppht.cpp)

target_link_libraries(adadoc PRIVATE pybind11::module
        opencv_core
        opencv_imgproc)
set_target_properties(adadoc PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

