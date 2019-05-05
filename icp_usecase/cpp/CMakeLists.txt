cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(icp_dataset)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (icp_dataset icp_dataset_random_trafo.cpp)
target_link_libraries (icp_dataset ${PCL_LIBRARIES})