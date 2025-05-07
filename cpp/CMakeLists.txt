cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(CMAKE_PREFIX_PATH "/mnt/combined/home/parveen/gouri/libtorch/libtorch")

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)